import cv2
import numpy as np
import matplotlib.pyplot as plt

def ridgeline_plot(ax, data_list, colors, labels):
    """
    Erzeugt in 'ax' einen Ridgeline-Plot aus mehreren Datenlisten.
    Jede Datenliste steht für eine (Farb-)Verteilung.
    """
    bins = 256  # Anzahl der Bins für unser Histogram
    vertical_offset = 0.05  # Abstand zwischen den Kurven
    
    for i, (data, c, label) in enumerate(zip(data_list, colors, labels)):
        hist, bin_edges = np.histogram(data, bins=bins, range=(0, 256), density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        offset = i * vertical_offset
        
        # Für das zweite Bild machen wir die Kurven etwas durchsichtiger (alpha=0.5),
        # damit man sie voneinander unterscheiden kann.
        # Du kannst die Werte natürlich anpassen.
        alpha_value = 0.8 if "Bild1" in label else 0.5
        
        ax.fill_between(bin_centers,
                        offset,
                        hist + offset,
                        color=c, alpha=alpha_value,
                        label=label)
    
    ax.set_xlim(0, 255)
    ax.set_ylim(0, vertical_offset * len(data_list))
    ax.set_yticks([])  # y-Achse ausblenden
    ax.legend(loc='upper right')


def main():
    # ============================================================
    # === 1) Zwei Bilder laden (BGR-Format) und nach RGB wandeln ==
    # ============================================================
    bild1_bgr = cv2.imread(r"C:\Users\moham\OneDrive\Dokumente\BigData\Test_bilder\hummingbird-2139278_1920.jpg")
    bild2_bgr = cv2.imread(r"C:\Users\moham\OneDrive\Dokumente\BigData\Test_bilder\puffin-5404178_1920.jpg")
    
    if bild1_bgr is None or bild2_bgr is None:
        print("Fehler: Mindestens eines der beiden Bilder wurde nicht gefunden.")
        return
    
    bild1_rgb = cv2.cvtColor(bild1_bgr, cv2.COLOR_BGR2RGB)
    bild2_rgb = cv2.cvtColor(bild2_bgr, cv2.COLOR_BGR2RGB)
    
    # ============================================================
    # === 2) Beide Bilder in 4 Quadranten aufteilen            ===
    # ============================================================
    h1, w1, _ = bild1_rgb.shape
    h2, w2, _ = bild2_rgb.shape
    
    # Halbierung für Bild1
    mid_h1 = h1 // 2
    mid_w1 = w1 // 2
    
    # Halbierung für Bild2
    mid_h2 = h2 // 2
    mid_w2 = w2 // 2
    
    # Quadranten von Bild 1
    quadrants_bild1 = [
        bild1_rgb[0:mid_h1, 0:mid_w1],      # oben-links
        bild1_rgb[0:mid_h1, mid_w1:w1],     # oben-rechts
        bild1_rgb[mid_h1:h1, 0:mid_w1],     # unten-links
        bild1_rgb[mid_h1:h1, mid_w1:w1]     # unten-rechts
    ]
    
    # Quadranten von Bild 2
    quadrants_bild2 = [
        bild2_rgb[0:mid_h2, 0:mid_w2],      # oben-links
        bild2_rgb[0:mid_h2, mid_w2:w2],     # oben-rechts
        bild2_rgb[mid_h2:h2, 0:mid_w2],     # unten-links
        bild2_rgb[mid_h2:h2, mid_w2:w2]     # unten-rechts
    ]
    
    # ============================================================
    # === 3) Matplotlib-Figur mit 4 Subplots (für 4 Quadranten) ==
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Vergleich der Farbverteilungen (Ridgeline) in 4 Quadranten – Bild1 vs. Bild2")
    
    # Für einfachere Indizierung:
    quadrant_names = ["Oben-Links", "Oben-Rechts", "Unten-Links", "Unten-Rechts"]
    
    for i in range(4):
        # Hole den zugehörigen Subplot
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # Quadrant i von Bild1
        sub_img1 = quadrants_bild1[i]
        # Quadrant i von Bild2
        sub_img2 = quadrants_bild2[i]
        
        # Extrahiere die Pixelwerte
        r1 = sub_img1[..., 0].ravel()
        g1 = sub_img1[..., 1].ravel()
        b1 = sub_img1[..., 2].ravel()
        
        r2 = sub_img2[..., 0].ravel()
        g2 = sub_img2[..., 1].ravel()
        b2 = sub_img2[..., 2].ravel()
        
        # Bereite Datensätze (Bild1: R,G,B; Bild2: R,G,B)
        data_list = [r1, g1, b1, r2, g2, b2]
        color_list = ["r", "g", "b", "r", "g", "b"]
        label_list = [
            "Bild1 - Rot", "Bild1 - Grün", "Bild1 - Blau",
            "Bild2 - Rot", "Bild2 - Grün", "Bild2 - Blau"
        ]
        
        # Ridgeline-Plot in diesem Subplot erstellen
        ridgeline_plot(ax, data_list, color_list, label_list)
        
        ax.set_title(f"{quadrant_names[i]}")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
