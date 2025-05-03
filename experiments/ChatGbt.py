import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.models import resnet18
import torchvision.transforms as transforms

# 💡 Bildvorbereitung und Modell
modell = resnet18(pretrained=True)
modell.fc = torch.nn.Identity()
modell.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def extrahiere_feature(bild):
    bild = cv2.cvtColor(bild, cv2.COLOR_BGR2RGB)
    bild = transform(bild).unsqueeze(0)
    with torch.no_grad():
        feature = modell(bild)
    return feature.squeeze().numpy()

# 📁 Bilder laden
def lade_bilder(pfad):
    bilder = []
    namen = []
    for datei in os.listdir(pfad):
        if datei.lower().endswith((".jpg", ".png", ".jpeg")):
            bild = cv2.imread(os.path.join(pfad, datei))
            if bild is not None:
                bilder.append(bild)
                namen.append(datei)
    return bilder, namen

# 🔍 Ähnlichkeit berechnen
def finde_aehnlichste_bilder(query_bild, alle_bilder, namen):
    query_feature = extrahiere_feature(query_bild).reshape(1, -1)
    feature_liste = [extrahiere_feature(b) for b in alle_bilder]
    similarities = cosine_similarity(query_feature, feature_liste)[0]
    top5 = np.argsort(similarities)[::-1][:5]
    return [(namen[i], similarities[i], alle_bilder[i]) for i in top5]

# 🚀 Hauptfunktion
def starte_suche(ordner, testbild_pfad):
    bilder, namen = lade_bilder(ordner)
    testbild = cv2.imread(testbild_pfad)
    ergebnisse = finde_aehnlichste_bilder(testbild, bilder, namen)

    for name, score, bild in ergebnisse:
        plt.imshow(cv2.cvtColor(bild, cv2.COLOR_BGR2RGB))
        plt.title(f"{name} - Ähnlichkeit: {score:.2f}")
        plt.axis("off")
        plt.show()

# Beispiel
# Passe den Ordnerpfad und das Bildpfad an!
ordner = r"C:\Users\moham\OneDrive\Dokumente\BigData\Test_bilder"
testbild = r"C:\Users\moham\OneDrive\Dokumente\BigData\Test_bilder\puffin-5404178_1920.jpg"
starte_suche(ordner, testbild)
