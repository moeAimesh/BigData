import os
import time
import psutil
import torch
import traceback
from loader import image_generator
from image_load import fast_load
from features.color_vec import calc_histogram
from features.hash import calc_hash
from features.embedding_vec import extract_embeddings

PHOTO_FOLDER = r"D:\data\image_data" 
MAX_BATCH = 1024
STEP = 64  
LOG_FILE = "batch_test_log.txt"

def safe_log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)

def run_batch_test(batch_size, all_images):
    batch_images = []
    try:
        for i, (filename, path) in enumerate(all_images[:batch_size]):
            img = fast_load(path)
            _ = calc_histogram(img)
            _ = calc_hash(img)
            batch_images.append(img)
        start = time.time()
        _ = extract_embeddings(batch_images)
        duration = time.time() - start
        used_ram = psutil.virtual_memory().used / 1024**3
        safe_log(f"✅ BATCH_SIZE {batch_size}: {duration:.2f}s, RAM: {used_ram:.2f} GB")
        return True
    except Exception as e:
        safe_log(f"❌ BATCH_SIZE {batch_size} failed: {e}\n{traceback.format_exc()}")
        return False

def find_max_batch():
    all_images = list(image_generator(PHOTO_FOLDER))
    safe_log(f"📸 {len(all_images)} Bilder gefunden")
    for batch_size in range(STEP, MAX_BATCH + STEP, STEP):
        torch.cuda.empty_cache()
        if not run_batch_test(batch_size, all_images):
            safe_log(f"🔚 Max. stabile BATCH_SIZE: {batch_size - STEP}")
            break

if __name__ == "__main__":
    find_max_batch()
