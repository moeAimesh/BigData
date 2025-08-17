# ImageRecommender.py
# Startbar ohne CLI: Mode & Parameter unten einstellen und Datei direkt starten.

import json, sys, os, math
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# ---------- BOOTSTRAP: Projekt-Root in sys.path aufnehmen ----------
ROOT = Path(__file__).resolve().parents[2]  # ...\BIG_DATA
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------- KONFIG ----------
MODE    = "color"  # "color" | "hash" | "embed" | "mix"
DB      = r"C:\BIG_DATA\data\database.db"
IMAGE   = r"Z:\CODING\UNI\BIG_DATA\data\TEST_IMAGES\Dunkelheit-und-Licht.png"
TOP_K   = 5   # für color + auch als print_k/final_k genutzt, s.u.

# Plot-Output
SAVE_FIG  = False
FIG_PATH  = str(Path(__file__).with_name("results_grid.png"))

# ---------- PROFILING ----------
PROFILE_ENABLED     = True         # Profiling an/aus
PROFILE_SORT        = "cumulative" # 'cumulative' | 'time' | 'tottime' | 'calls' ...
PROFILE_PRINT_TOP   = 40           # wie viele Zeilen in der Konsole ausgeben
PROFILE_DUMP_FILE   = False         # .prof-Datei speichern
PROFILE_SKIP_PLOT   = True         # während Profiling Plots überspringen (nur Search messen)

def _with_profile(mode_label, func, *args, **kwargs):
    if not PROFILE_ENABLED:
        return func(*args, **kwargs)
    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()
    try:
        result = func(*args, **kwargs)
    finally:
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).strip_dirs().sort_stats(PROFILE_SORT).print_stats(PROFILE_PRINT_TOP)
        print("\n" + "="*80)
        print(f"[cProfile] Sorted by: {PROFILE_SORT} | Top {PROFILE_PRINT_TOP}")
        print("="*80)
        print(s.getvalue())
        if PROFILE_DUMP_FILE:
            out = Path(__file__).with_name(f"profile_{mode_label}_2.prof")
            pr.dump_stats(str(out))
            print(f"[cProfile] Stats gespeichert: {out}")
    return result

# ---------- VISUALIZATION ----------
def _imread_rgb(path: str):
    if not path or not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def _extract_paths_and_captions(
    items,
    path_keys=("path", "image_path", "filepath"),
    score_keys=("fused_similarity", "score", "cos512", "cos", "similarity"),
    max_items=None,
):
    paths, caps = [], []
    if not items:
        return paths, caps
    seq = items if max_items is None else items[:max_items]
    for it in seq:
        if not isinstance(it, dict):
            continue
        # Pfad finden
        p = None
        for k in path_keys:
            if k in it and it[k]:
                p = str(it[k])
                break
        if not p:
            continue
        # Caption/Score (ohne Rundung)
        caption = ""
        for k in score_keys:
            if k in it and isinstance(it[k], (int, float)):
                caption = f"{k}={it[k]}"
                break
        pm = it.get("per_metric")
        if isinstance(pm, dict) and not caption and len(pm) > 0:
            mk = next(iter(pm.keys()))
            caption = f"{mk}={pm[mk]}"
        paths.append(p)
        caps.append(caption)
    return paths, caps

def show_gallery(query_path: str, result_paths, captions=None, cols=5, title="Results"):
    q = _imread_rgb(query_path)
    # nur existierende Ergebnisse nehmen
    rs = []
    caps = []
    for i, p in enumerate(result_paths):
        img = _imread_rgb(p)
        if img is not None:
            rs.append(img)
            caps.append(captions[i] if captions and i < len(captions) else "")

    n = len(rs)
    if n == 0 and q is None:
        print("[WARN] Keine anzeigbaren Bilder.")
        return

    cols = max(2, min(cols, 8))
    rows = math.ceil(max(1, n) / cols)

    fig = plt.figure(figsize=(cols * 2.6, 2 + rows * 2.6))
    gs = fig.add_gridspec(nrows=1 + rows, ncols=cols)

    # Query oben über alle Spalten
    axq = fig.add_subplot(gs[0, :])
    axq.axis("off")
    if q is not None:
        axq.imshow(q)
        axq.set_title("Query", fontsize=12)
    else:
        axq.text(0.5, 0.5, "Query not available", ha="center", va="center")

    # Ergebnisse
    for i, img in enumerate(rs):
        r = i // cols
        c = i % cols
        ax = fig.add_subplot(gs[1 + r, c])
        ax.imshow(img)
        ax.axis("off")
        if caps[i]:
            ax.set_title(caps[i], fontsize=9)

    if title:
        fig.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()

    if SAVE_FIG:
        try:
            plt.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
            print(f"[INFO] Figure gespeichert: {FIG_PATH}")
        except Exception as e:
            print(f"[WARN] Konnte Figure nicht speichern: {e}")

    plt.show()

# ---------- RUNNER ----------
def run_color(db_path: str, image_path: str, top_k: int = 10, do_plot: bool = True):
    from src.similarity import color as C
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")

    q_bgr = C.calc_histogram(img, bins=C.BINS)
    q_db  = C.swap_rb_hist(q_bgr, C.BINS)  # [B|G|R] -> [R|G|B] passend zur DB
    weights = C.auto_weights(q_db, C.BINS)

    results = C.search_color_voting(
        db_path=db_path,
        q_hist=q_db,
        hist_col=C.HIST_COL,
        metrics=C.DEFAULT_METRICS,
        weight_map=weights,
        topk=top_k,
    )

    print(json.dumps({
        "module": "color",
        "weights": weights,
        "results": results
    }, ensure_ascii=False, indent=2))

    if do_plot:
        paths, caps = _extract_paths_and_captions(results, max_items=top_k)
        show_gallery(
            query_path=image_path,
            result_paths=paths,
            captions=caps,
            cols=min(5, max(3, top_k)),
            title=f"Color Search (top {top_k})"
        )

def run_hash(db_path: str, image_path: str, final_k: int = 6, do_plot: bool = True):
    from src.similarity import hash as H
    q_ah, q_dh, q_ph = H.compute_query_hashes(image_path)
    results, w = H.search_by_hash_voting_multitables(
        db_path=db_path,
        q_ahash=q_ah, q_dhash=q_dh, q_phash=q_ph,
        weights="auto",
        topk_per_hash=300,
        final_k=final_k,
        return_weights=True
    )
    print(json.dumps({
        "module": "hash",
        "weights": {"ahash": w[0], "dhash": w[1], "phash": w[2]},
        "results": results
    }, ensure_ascii=False, indent=2))

    if do_plot:
        paths, caps = _extract_paths_and_captions(results, max_items=final_k)
        show_gallery(
            query_path=image_path,
            result_paths=paths,
            captions=caps,
            cols=min(5, max(3, final_k)),
            title=f"Hash Voting (final_k={final_k})"
        )

def run_embed(db_path: str, image_path: str, print_k: int = 5, do_plot: bool = True):
    from src.similarity import embedding as E
    out = E.search_image(Path(image_path), Path(db_path))
    top = out.get("results", {}).get("final_top_k", [])[:print_k]

    print(json.dumps({
        "module": "embedding",
        "query": out.get("query", {}),
        "results": top,
        "params": out.get("params", {})
    }, ensure_ascii=False, indent=2))

    if do_plot:
        paths, caps = _extract_paths_and_captions(top, max_items=print_k)
        show_gallery(
            query_path=image_path,
            result_paths=paths,
            captions=caps,
            cols=min(5, max(3, print_k)),
            title=f"Embedding (print_k={print_k})"
        )

def run_mix(db_path: str, image_path: str, print_k: int = 5, do_plot: bool = True):
    from src.similarity import mix as M
    out = M.search_image(Path(image_path), Path(db_path))
    top = out.get("results", {}).get("final_top_k", [])[:print_k]

    print(json.dumps({
        "module": "mix",
        "query": out.get("query", {}),
        "results": top,
        "params": out.get("params", {})
    }, ensure_ascii=False, indent=2))

    if do_plot:
        paths, caps = _extract_paths_and_captions(top, max_items=print_k)
        show_gallery(
            query_path=image_path,
            result_paths=paths,
            captions=caps,
            cols=min(5, max(3, print_k)),
            title=f"Mix (print_k={print_k})"
        )

# ---------- MAIN ----------
def main():
    # bei Profiling (typisch): Plots überspringen, damit die Messung nicht dominiert wird
    do_plot = not PROFILE_ENABLED or not PROFILE_SKIP_PLOT

    if MODE == "color":
        return _with_profile("color", run_color, DB, IMAGE, TOP_K, do_plot=do_plot)
    elif MODE == "hash":
        return _with_profile("hash", run_hash, DB, IMAGE, TOP_K, do_plot=do_plot)
    elif MODE == "embed":
        return _with_profile("embed", run_embed, DB, IMAGE, TOP_K, do_plot=do_plot)
    elif MODE == "mix":
        return _with_profile("mix", run_mix, DB, IMAGE, TOP_K, do_plot=do_plot)
    else:
        raise ValueError(f"Unbekannter MODE: {MODE}")

if __name__ == "__main__":
    main()
