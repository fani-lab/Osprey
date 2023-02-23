import os

def force_open(path, *args, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    return open(path, *args, **kwargs)
