# engine/imageRec.py
# -*- coding: utf-8 -*-
import sys, json, argparse
from pathlib import Path

from engine.core import Config, warmup, search_once

def main():
    p = argparse.ArgumentParser(description="Image Recommender (Dispatcher)")
    p.add_argument("--db", required=True, help="Pfad zur SQLite-DB")
    p.add_argument("--image", required=True, help="Pfad zum Query-Bild")
    p.add_argument("--mode", default="mix", choices=["mix","embedding","color","hash"], help="Suchmodus")
    p.add_argument("--top-k", type=int, default=5, help="Anzahl finaler Treffer")
    p.add_argument("--hist-col", default="color_hist", help="Histogramm-Spaltenname in der DB")
    p.add_argument("--color-bins", type=int, default=32, help="Bins pro Kanal (Default 32)")
    args = p.parse_args()

    cfg = Config(
        mode=args.mode,
        top_k=args.top_k,
        hist_col=args.hist_col,
        color_bins=args.color_bins,
    )

    db = Path(args.db)
    img = Path(args.image)

    warmup(cfg)
    out = search_once(img, db, cfg)

    # kompakte Ausgabe
    to_print = {
        "query": out.get("query", {}),
        "results": {"final_top_k": out["results"]["final_top_k"][:min(args.top_k, 5)]},
        "params": out.get("params", {})
    }
    print(json.dumps(to_print, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    sys.exit(main())



#python imageRec.py --db "C:\BIG_DATA\data\database.db" --image "...\bild.jpg" --mode color --top-k 10