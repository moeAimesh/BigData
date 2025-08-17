import numpy as np
from vispy import scene, app

# ---------- Datei & Anzeige ----------
NPZ_PATH = r"C:\BIG_DATA\data\avg_rgb_points_from_hist_bold_mode_maxed.npz"
ALPHA = 1.0
POINT_SIZE = 4.0
BG_COLOR = "grey"  # besser als weiss weil neutral und weniger ablenkend


VOXEL_BINS = None


def voxel_downsample_rgb(pos, bins_per_axis=64):
    """
    RGB-Würfel in bins^3 Voxel; pro Voxel den Mittelwert & Count.
    Gibt positions (N,3 0..255), colors (N,3 0..1), sizes (N,) zurück.
    """
    pos = np.clip(pos, 0, 255).astype(np.float32)
    step = 256.0 / bins_per_axis
    idx = np.minimum(np.floor(pos / step).astype(np.int32), bins_per_axis - 1)
    keys = (
        idx[:, 0] + bins_per_axis * idx[:, 1] + (bins_per_axis**2) * idx[:, 2]
    ).astype(np.int64)

    order = np.argsort(keys, kind="mergesort")
    keys_sorted = keys[order]
    pos_sorted = pos[order]

    uniq, start_idx = np.unique(keys_sorted, return_index=True)
    end_idx = np.r_[start_idx[1:], keys_sorted.size]

    means = np.empty((uniq.size, 3), dtype=np.float32)
    sizes = np.empty(uniq.size, dtype=np.float32)
    for i, (s, e) in enumerate(zip(start_idx, end_idx)):
        block = pos_sorted[s:e]
        means[i] = block.mean(axis=0)
        count = e - s
        sizes[i] = 2.0 + 3.0 * np.log1p(
            count
        )  # größere Voxel dichter belegt -> größerer Marker

    colors = (means / 255.0).clip(0, 1).astype(np.float32)
    return means, colors, sizes


def main():
    # 1) Laden
    data = np.load(NPZ_PATH)
    pos = data["pos"].astype(np.float32)  # (N,3) 0..255
    colors = data["colors"].astype(np.float32)  # (N,3) 0..1
    sizes = None

    print("Geladen:", NPZ_PATH, " Punkte:", pos.shape[0])

    # 2) Optional LOD
    if VOXEL_BINS is not None:
        pos, colors, sizes = voxel_downsample_rgb(pos, bins_per_axis=VOXEL_BINS)
        print(f"LOD aktiv: bins={VOXEL_BINS}  -> Punkte: {pos.shape[0]}")

    # 3) VisPy Setup
    canvas = scene.SceneCanvas(
        keys="interactive", bgcolor=BG_COLOR, size=(1100, 800), show=True
    )
    view = canvas.central_widget.add_view()
    view.camera = "turntable"  # Maus: drehen/zoomen/pannen

    markers = scene.visuals.Markers(parent=view.scene)
    rgba = np.c_[colors, np.full((colors.shape[0], 1), ALPHA, dtype=np.float32)]

    markers.set_data(
        pos=pos,
        face_color=rgba,
        size=(sizes if sizes is not None else POINT_SIZE),
        edge_width=0.0,
    )

    # Achsenrahmen & Limits (RGB-Würfel)
    scene.visuals.XYZAxis(parent=view.scene)
    view.camera.set_range(x=(0, 255), y=(0, 255), z=(0, 255))

    # schöner Blickwinkel
    try:
        view.camera.azimuth = 35
        view.camera.elevation = 20
    except Exception:
        pass

    app.run()


if __name__ == "__main__":
    main()
