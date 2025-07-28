from PIL import Image
import imagehash

def calc_hash(image):
    return imagehash.average_hash(image).hash.astype(int).flatten()

# → Bild laden
image_path = r"C:\Users\moham\OneDrive\Dokumente\Big_Data\fork_repo_local\Test_bilder\abbey-1851493_1920.jpg"
img = Image.open(image_path).convert("RGB")

# → Hash berechnen
print(calc_hash(img))
