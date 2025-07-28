import os

def image_generator(root_dir, valid_exts=(".jpg", ".jpeg", ".png")):
    """
    Generator, der rekursiv durch alle Unterordner läuft

    Args:
        root_dir (str): Hauptverzeichnis mit Bildern und Unterordnern
        valid_exts (tuple): gültige Dateiendungen

    Yields:
        (filename, full_path)
    """
    for root, _, files in os.walk(root_dir):
        for filename in files:
            if filename.lower().endswith(valid_exts):
                full_path = os.path.join(root, filename)
                yield filename, full_path
