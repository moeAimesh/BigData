import os

def image_generator(root_dir, valid_exts=(".jpg", ".jpeg", ".png")):
    for root, _, files in os.walk(root_dir):
        for filename in files:
            if filename.lower().endswith(valid_exts):
                full_path = os.path.join(root, filename)
                yield filename, full_path
