import numpy as np

def xywh2ltwh(xywh):
    """Convert center-based (x,y,w,h) to left-top-based (x,y,w,h)."""
    x, y, w, h = xywh
    return np.array([x - w / 2.0, y - h / 2.0, w, h], dtype=np.float32)
