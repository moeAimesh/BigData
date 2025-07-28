import torch
from torchvision import models, transforms

#Gerät auswählen (CUDA, wenn verfügbar)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Modell laden und auf GPU verschieben
resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # ohne FC-Schicht
resnet.eval().to(device)

#Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#Funktion zum Berechnen des Embeddings mit GPU-Support
def calc_embedding(image):
    img_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]
    with torch.no_grad():
        features = resnet(img_tensor)  # [1, 512, 1, 1]
    return features.view(-1).cpu().numpy()  # zurück auf CPU für Kompatibilität
