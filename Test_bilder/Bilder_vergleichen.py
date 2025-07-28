"""
Vergleicht zwei Bilder über Farben durch Histogramme (RGB oder HSV)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------------------------------- 
#  USER PARAMETER                                                        
# --------------------------------------------------------------------------- 

IMG_A   = Path(r"C:\Users\moham\OneDrive\Dokumente\Big_Data\fork_repo(local)\Test_bilder\image.png")     # Pfad zu Bild A
IMG_B   = Path(r"C:\Users\moham\OneDrive\Dokumente\Big_Data\fork_repo(local)\Test_bilder\image1.png")     # Pfad zu Bild B
ROWS    = 2                      # Anzahl Zeilen im Raster
COLS    = 3                      # Anzahl Spalten im Raster
USE_HSV = False                  # True → Hue+Saturation statt RGB
BINS    = 256                    # Histogramm‑Balken pro Kanal










def load_and_resize(path, target_size=None, use_hsv=False):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    if use_hsv:
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def split_into_tiles(img, rows, cols):
    h, w = img.shape[:2]
    img = img[: h // rows * rows, : w // cols * cols]
    tile_h, tile_w = h // rows, w // cols
    return [img[r * tile_h:(r + 1) * tile_h,
                c * tile_w:(c + 1) * tile_w]
            for r in range(rows) for c in range(cols)]


def hist_similarity(a, b, bins, channels):
    sim = 0.0
    for ch in channels:
        h1 = cv2.calcHist([a], [ch], None, [bins], [0, 256])
        h2 = cv2.calcHist([b], [ch], None, [bins], [0, 256])
        h1 = cv2.normalize(h1, h1, alpha=1, norm_type=cv2.NORM_L1).flatten().astype("float32")
        h2 = cv2.normalize(h2, h2, alpha=1, norm_type=cv2.NORM_L1).flatten().astype("float32")
        sim += cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    return sim / len(channels)





def main():
    img_a = load_and_resize(IMG_A, use_hsv=USE_HSV)
    img_b = load_and_resize(IMG_B, target_size=img_a.shape[1::-1], use_hsv=USE_HSV)

    channels = (0, 1) if USE_HSV else (0, 1, 2)
    ch_names = ("H", "S") if USE_HSV else ("R", "G", "B")

    global_score = hist_similarity(img_a, img_b, BINS, channels)
    print(f"Total similarity Score: {global_score:.4f}")

    tiles_a = split_into_tiles(img_a, ROWS, COLS)
    tiles_b = split_into_tiles(img_b, ROWS, COLS)

    fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 4, ROWS * 3), squeeze=False)
    Title = "HSV" if USE_HSV else "RGB"
    fig.suptitle(f" Compare {Title} \n Total similarity Score: {global_score:.4f}", fontsize= 30)

    for idx, (ta, tb) in enumerate(zip(tiles_a, tiles_b)):
        ax = axes[idx // COLS, idx % COLS]
        score = hist_similarity(ta, tb, BINS, channels)
        ax.set_title(f"Square {idx + 1}\n Score {score:.3f}", fontsize=20)

        for ch, col, name in zip(channels, ("r", "g", "b")[:len(channels)], ch_names):
            h_ta, _ = np.histogram(ta[..., ch].ravel(), bins=BINS, range=(0, 256), density=True)
            h_tb, _ = np.histogram(tb[..., ch].ravel(), bins=BINS, range=(0, 256), density=True)
            x = np.linspace(0, 255, BINS)
            ax.plot(x, h_ta, color=col, alpha=0.7, label=f"A-{name}")
            ax.plot(x, h_tb, color=col, alpha=0.3, label=f"B-{name}")

        ax.set_xlim(0, 255)
        ax.set_yticks([])
        ax.legend(fontsize=6, loc="upper right", ncol=2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
