import torch
import numpy as np
import cv2
from torchvision.models import resnet18, ResNet18_Weights
from pathlib import Path

# ========== SETUP ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = ResNet18_Weights.DEFAULT
resnet = resnet18(weights=weights)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Ohne FC-Schicht
resnet.eval().to(device)

# ImageNet Normalisierung
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ========== PREPROCESS ==========
def preprocess_image_np(img):
    """
    Schnelle Bildvorverarbeitung mit OpenCV + NumPy:
    - Resize
    - Normalisieren [0, 1]
    - Standardisieren (ImageNet)
    - in Torch-Tensor (CHW) umwandeln
    """
    if isinstance(img, (str, Path)):
        p = str(img)
        # robustes Lesen (Unicode/Windows-Pfade)
        data = np.fromfile(p, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cv2.imdecode konnte Bild nicht lesen: {p}")
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0  # HWC → CHW + Normalisierung
    img = (img - mean[:, None, None]) / std[:, None, None]         # Broadcasting auf CHW
    return torch.from_numpy(img)


# ========== EXTRACT ==========
def extract_embeddings(batch_imgs_np):
    """
    Wandelt eine Liste von NumPy-Bildern in Embedding-Vektoren (1 x 512) um.
    """
    batch_tensor = torch.stack([preprocess_image_np(img) for img in batch_imgs_np])
    batch_tensor = batch_tensor.to(device)

    with torch.no_grad():
        features = resnet(batch_tensor)  # [B, 512, 1, 1]
        features = features.squeeze(-1).squeeze(-1).cpu().numpy()  # [B, 512]

    return features
