"""Trim vertical whitespace only: keep the full page width, pad 20 px top/bottom."""
import sys

from PIL import Image, ImageChops

im = Image.open(sys.argv[1]).convert("RGB")
bg = Image.new("RGB", im.size, "white")
bbox = ImageChops.difference(im, bg).getbbox()  # (l, t, r, b)
pad = 20
top = max(bbox[1] - pad, 0)
bottom = min(bbox[3] + pad, im.height)
im.crop((0, top, im.width, bottom)).save(sys.argv[2])
