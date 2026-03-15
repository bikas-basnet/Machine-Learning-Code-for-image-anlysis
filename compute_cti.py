# compute_cti.py
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import datasets, models
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from collections import defaultdict
from tqdm import tqdm

# ================================
# CONFIG
# ================================
DATA_DIR = "data/test"
MODEL_DIR = "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 64

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load crops
with open("crop_list.txt") as f:
    CROPS = [line.strip() for line in f if line.strip()]

# ================================
# CROP NAME MAPPING (Handles messy filenames)
# ================================
CROP_NAME_MAP = {
    "cauliflower": ["cauliflower", "cauli"],
    "chili": ["chilli", "chili"],
    "large_cardamom": ["cardamom", "suffle"],
    "maize": ["maize"],
    "mung_bean": ["mung", "mungbean"],
    "onion": ["onion"],
    "kidney_bean": ["rajma", "kidney"],
    "rice": ["rice"],
    "sesame": ["sesame"],
    "strawberry": ["strawberry"],
    "Citrus": ["Citrus"]
}

def get_crop_from_filename(filename):
    name = filename.lower()
    for crop, keywords in CROP_NAME_MAP.items():
        if any(k in name for k in keywords):
            return crop
    return None

# ================================
# LOAD MODEL (Supports YOUR Hybrid + Others)
# ================================
def load_pretrained_model(model_path, num_classes):
    name = os.path.basename(model_path).lower()
    state_dict = torch.load(model_path, map_location=DEVICE)

    # === YOUR HYBRID MODEL ===
    if any(keyword in name for keyword in ["hybrid", "suffle", "maize", "mung", "onion", "chilli", "sesame", "squeez"]):
        from hybrid_model import HybridShuffleNetSqueezeNet
        model = HybridShuffleNetSqueezeNet(num_classes=num_classes).to(DEVICE)

    # === OTHER MODELS ===
    elif "mobilenet" in name or "mobile" in name:
        model = models.mobilenet_v2(weights=None).to(DEVICE)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif "resnet18" in name:
        model = models.resnet18(weights=None).to(DEVICE)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif "squeezenet" in name or "squeez" in name:
        model = models.squeezenet1_1(weights=None).to(DEVICE)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    else:
        raise ValueError(f"Unknown model type: {name}")

    # Load state dict (ignore classifier mismatch)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    model.eval()
    return model

# ================================
# EVALUATE WITH PROGRESS BAR
# ================================
def evaluate(model, dataloader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader, desc="Infer", leave=False):
            imgs = imgs.to(DEVICE)
            outs = model(imgs)
            _, pred = torch.max(outs, 1)
            preds.extend(pred.cpu().numpy())
            labels.extend(lbls.cpu().numpy())
    return accuracy_score(labels, preds)

# ================================
# MAIN: LOCO TRANSFER
# ================================
transfer_results = defaultdict(list)
transfer_matrix = pd.DataFrame(0.0, index=CROPS, columns=CROPS)

print("Starting Cross-Crop Transferability (CTI) Computation...\n")

for model_file in os.listdir(MODEL_DIR):
    if not model_file.endswith(".pth"): 
        continue
    model_path = os.path.join(MODEL_DIR, model_file)
    
    train_crop = get_crop_from_filename(model_file)
    if not train_crop:
        print(f"[SKIP] Cannot detect crop in: {model_file}")
        continue
    
    print(f"\nTraining on: {train_crop} → {model_file}")

    for test_crop in CROPS:
        if test_crop == train_crop: 
            continue
        
        test_path = os.path.join(DATA_DIR, test_crop)
        if not os.path.exists(test_path):
            print(f"  [MISSING] {test_crop} test folder")
            continue

        dataset = datasets.ImageFolder(test_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        num_classes = len(dataset.classes)

        try:
            model = load_pretrained_model(model_path, num_classes)
            acc = evaluate(model, dataloader)
            transfer_matrix.loc[train_crop, test_crop] = acc
            transfer_results[model_file].append((train_crop, test_crop, acc))
            print(f"    → {test_crop}: {acc:.4f}")
        except Exception as e:
            print(f"    [ERROR] {test_crop}: {e}")

# ================================
# COMPUTE CTI
# ================================
cti_scores = {}
for model_file, transfers in transfer_results.items():
    accs = [acc for _, _, acc in transfers]
    cti = np.mean(accs) if accs else 0.0
    cti_scores[model_file] = cti

# ================================
# SAVE RESULTS
# ================================
# 1. Transfer Matrix
transfer_matrix.to_csv("cti_transfer_matrix.csv")
print("\nTransfer matrix saved: cti_transfer_matrix.csv")

# 2. CTI Summary
cti_df = pd.DataFrame([
    {"Model": m, "CTI (%)": f"{v*100:.2f}"}
    for m, v in cti_scores.items()
]).sort_values("CTI (%)", ascending=False)
cti_df.to_csv("cti_summary.csv", index=False)
print("\n" + "="*60)
print("CROSS-CROP TRANSFERABILITY INDEX (CTI)")
print("="*60)
print(cti_df)
print("="*60)

# 3. Heatmap
plt.figure(figsize=(11, 9))
sns.heatmap(transfer_matrix, annot=True, fmt=".3f", cmap="viridis", cbar_kws={'label': 'Accuracy'})
plt.title("Cross-Crop Transfer Accuracy (Source → Target)")
plt.ylabel("Train Crop")
plt.xlabel("Test Crop")
plt.tight_layout()
plt.savefig("cti_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
print("Heatmap saved: cti_heatmap.png")
print("CTI Summary saved: cti_summary.csv")