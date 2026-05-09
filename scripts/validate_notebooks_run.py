import importlib.util
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
bd_path = ROOT / 'public' / 'image-analysis-miu-batubara' / 'block_detection.py'
cd_path = ROOT / 'public' / 'image-analysis-miu-batubara' / 'circle_detection.py'

def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

bd = load_module_from_path('bd', bd_path)
cd = load_module_from_path('cd', cd_path)

print('Loading sample-stepwedge.tiff...')
with open('sample-stepwedge.tiff','rb') as f:
    b = f.read()
img = bd._load_and_validate_image(b)
print('Loaded shape:', img.shape, 'dtype:', img.dtype)

h,w = img.shape
box = [[w//2-20, h//2-20], [w//2+20, h//2-20], [w//2+20, h//2+20], [w//2-20, h//2+20]]
mean_val, pixels = bd._mean_intensity_in_box(img, box, bd.ROI_SHRINK_RATIO)
print('Block sample mean:', mean_val, 'pixels:', len(pixels))

print('Loading sample-circle.tiff...')
with open('sample-circle.tiff','rb') as f:
    b2 = f.read()
img2 = cd._load_and_validate_image(b2)
print('Loaded circle shape:', img2.shape, 'dtype:', img2.dtype)

# sample circle mean: pick a center and radius
ch, cw = img2.shape[0]//2, img2.shape[1]//2
r = min(ch,cw)//8
shr = max(1, int(r * (1.0 - bd.ROI_SHRINK_RATIO)))
mask = np.zeros_like(img2, dtype=np.uint8)
cv2.circle(mask, (cw,ch), shr, 255, -1)
pvals = img2[mask==255]
print('Circle sample mean:', float(np.mean(pvals)), 'pixels:', pvals.size)
