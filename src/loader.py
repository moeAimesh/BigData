import os

def image_generator(folder):
    for root, _, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                yield fname, os.path.join(root, fname)
