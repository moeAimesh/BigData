from PIL import Image
import imagehash

bild1 = Image.open(r"C:\Users\moham\OneDrive\Dokumente\BigData\Test_bilder\hummingbird-2139278_1920.jpg")
bild2 = Image.open(r"C:\Users\moham\OneDrive\Dokumente\BigData\Test_bilder\puffin-5404178_1920.jpg")

# verschiedene Hashes für bild1 und bild2 berechnen
phash1 = imagehash.phash(bild1)
phash2 = imagehash.phash(bild2)

avg1   = imagehash.average_hash(bild1)
avg2   = imagehash.average_hash(bild2)

col1   = imagehash.colorhash(bild1)
col2   = imagehash.colorhash(bild2)

dhash1 = imagehash.dhash(bild1)
dhash2 = imagehash.dhash(bild2)

# die Hamming-Distanz berechnen
phash_dist = phash1 - phash2       
avg_dist   = avg1 - avg2           
col_dist   = col1 - col2           
dhash_dist = dhash1 - dhash2

hashes_dist = [
    ('pHash', phash_dist),
    ('dHash', dhash_dist),
    ('AverageHash', avg_dist),
    ('ColorHash', col_dist)
]

# Durchschnitt mehrere Hamming-Distanzen
combined_dist = sum([dist for name, dist in hashes_dist]) / len(hashes_dist)

# Ähnlichkeit basierend auf den Hash-Distanzen
def similarity(hash_list):
    similar = []
    for name, dist in hash_list:
        if dist <= 5:
            similar.append(f"{name}") 
    return similar



for name, dist in hashes_dist:
    print(f"{name} Distanz: {dist}")
print("Kombinierte Distanz:", combined_dist)



# Ähnlichkeit prüfen
threshold = 15
if combined_dist < threshold:
    print("Ähnlich.")
else:
    similar_hashes = similarity(hashes_dist)
    print(f"Im Durchschnitt nicht ähnlich, jedoch sind die nachfolgenden Hash-Distanzen ähnlich: {', '.join(similar_hashes)}")