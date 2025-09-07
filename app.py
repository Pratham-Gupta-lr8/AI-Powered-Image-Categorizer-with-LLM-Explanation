from __future__ import annotations
import os
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
import torchvision.datasets as datasets
import torchvision.models as models

from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# -----------------------------
# Config
# -----------------------------
DEVICE = torch.device("cpu") 
MODEL_WEIGHTS = "cnn_classifier.pth"  
SLM_MODEL_NAME = "microsoft/phi-1_5" 
BATCH_SIZE = 16 
EPOCHS = 6 
LR = 3e-4 
VAL_SPLIT = 0.2 
NUM_WORKERS = 0 

# -----------------------------
# Small LLM Loader
# -----------------------------

_slm_model: Optional[AutoModelForCausalLM] = None
_slm_tokenizer: Optional[AutoTokenizer] = None

def get_slm() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    global _slm_model, _slm_tokenizer
    if _slm_model is None or _slm_tokenizer is None:
        _slm_tokenizer = AutoTokenizer.from_pretrained(SLM_MODEL_NAME)
        _slm_model = AutoModelForCausalLM.from_pretrained(SLM_MODEL_NAME)
        _slm_model.eval().to(DEVICE)
    return _slm_model, _slm_tokenizer

# -----------------------------
# Transforms
# -----------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transform():
    return T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def get_eval_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

# -----------------------------
# Training
# -----------------------------

def train_cnn(gallery_folder):
    if not os.path.exists(gallery_folder):
        return f"Folder {gallery_folder} does not exist.", ""

    train_data = datasets.ImageFolder(gallery_folder, transform=get_train_transform())
    if len(train_data) == 0:
        return f"No images found in {gallery_folder}. Make sure it has subfolders (one per class).", ""

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    class_names = train_data.classes
    num_classes = len(class_names)
    
    # -----------------------------
    # Model
    # -----------------------------
    
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # -----------------------------
    # Training loop
    # -----------------------------
    
    model.train()
    for epoch in range(EPOCHS):
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss:.3f} - Acc: {acc:.2f}%")
    
    # -----------------------------
    # Save model + classes + gallery folder
    # -----------------------------
    
    torch.save({
        "model_state": model.state_dict(),
        "classes": class_names,
        "gallery_folder": gallery_folder,
    }, MODEL_WEIGHTS)

    return "Training finished successfully!", f"Final accuracy: {acc:.2f}%"

# -----------------------------
# Loader with Auto-Retrain
# -----------------------------

def load_cnn():
    checkpoint = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
    saved_classes = checkpoint["classes"]
    gallery_folder = checkpoint.get("gallery_folder", "data")

    current_ds = datasets.ImageFolder(gallery_folder, transform=None)
    current_classes = current_ds.classes

    if set(saved_classes) != set(current_classes):
        print("Category change detected. Retraining CNN on updated folders...")
        train_cnn(gallery_folder)
        checkpoint = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
        saved_classes = checkpoint["classes"]

    num_classes = len(saved_classes)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval().to(DEVICE)
    return model, saved_classes, gallery_folder

# -----------------------------
# Prediction + Explanation
# -----------------------------

def predict_and_explain(upload, max_tokens: int = 80):
    if not os.path.exists(MODEL_WEIGHTS):
        return "Please train the CNN first.", None, None

    model, classes, _ = load_cnn()

    if isinstance(upload, str):
        upload = Image.open(upload).convert("RGB")
    image = get_eval_transform()(upload).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0]
        topk = torch.topk(probs, k=min(2, len(classes)))

    pred_idx = topk.indices[0].item()
    pred_label = classes[pred_idx]
    runner_up = classes[topk.indices[1].item()] if len(topk.indices) > 1 else None

    slm, tok = get_slm()
    p_top = topk.values[0].item()
    p_runner = topk.values[1].item() if len(topk.values) > 1 else None

    prompt = (
        "You are helping a beginner understand an image classification result.\n"
        f"Predicted category: {pred_label} with probability {p_top:.2f}.\n"
        f"Runner-up: {runner_up} with probability {p_runner:.2f} if applicable.\n"
        "Write a short explanation in plain language (2–3 sentences).\n"
    )

    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = slm.generate(
            **{k: v.to(DEVICE) for k, v in inputs.items()},
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    explanation = tok.decode(out[0], skip_special_tokens=True)
    if explanation.startswith(prompt):
        explanation = explanation[len(prompt):].strip()

    scores_text = "\n".join([f"{classes[i]}: {probs[i].item():.3f}" for i in range(len(classes))])
    return f"Predicted: {pred_label}", explanation, scores_text

# -----------------------------
# Gradio UI
# -----------------------------

def build_ui():
    with gr.Blocks(title="CNN Image Categorizer + SLM Explanation (CPU)") as demo:
        gr.Markdown(
            """
            # 🧠📷 CNN Image Categorizer + SLM Explanation (CPU-only)
            1. Put your images into subfolders under a **gallery** directory (one subfolder per category).
            2. Click **Train/Load CNN** (uses pretrained ResNet18 + augmentation).
            3. Upload an image and click **Predict**.
            4. If new categories appear, CNN auto-retrains before prediction.
            """
        )

        with gr.Row():
            gallery_root = gr.Textbox(value="data", label="Gallery folder", placeholder="e.g., data")
            train_btn = gr.Button("Train/Load CNN", variant="primary")

        with gr.Row():
            train_status = gr.Textbox(label="Training status log", lines=10)
            train_progress = gr.Textbox(label="Training summary")

        with gr.Row():
            image_in = gr.Image(type="pil", label="Upload an image")
            predict_btn = gr.Button("Predict", variant="primary")

        pred_label = gr.Label(label="Prediction")
        explain_box = gr.Textbox(lines=6, label="SLM Explanation")
        score_box = gr.Textbox(lines=6, label="Category probabilities")

        train_btn.click(train_cnn, inputs=[gallery_root], outputs=[train_status, train_progress])
        predict_btn.click(
            predict_and_explain,
            inputs=[image_in],
            outputs=[pred_label, explain_box, score_box],
        )

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
