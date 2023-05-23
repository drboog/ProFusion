import torch
import os
import argparse
import PIL.Image
import numpy as np
from tqdm import tqdm
import joblib as jlb # parallelizing


parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str)
parser.add_argument("--dest", type=str)
parser.add_argument("--num", type=int)
parser.add_argument("--size", type=int)


opt = parser.parse_args()
path = opt.src
size = opt.size
dest_path = opt.dest
if dest_path is None:
    dest_path = path

if opt.num is None:
    num = 20
else:
    num = opt.num


print(path, dest_path)

def file_ext(fname):
    return os.path.splitext(fname)[1].lower()

PIL.Image.init()

all_fnames = {os.path.relpath(os.path.join(root, fname), start=path) for root, _dirs, files in os.walk(path) for fname in files} # relative paths to --src
image_fnames = sorted(fname for fname in all_fnames if file_ext(fname) in PIL.Image.EXTENSION)

print(len(image_fnames))

def resize_img(fname):
    try:
        with open(os.path.join(path, fname), 'rb') as f:
            image = PIL.Image.open(f).convert('RGB')
            w, h = image.size
            crop = min(w, h)
            image = image.crop(((w-crop)//2, (h-crop)//2, (w+crop)//2, (h+crop)//2))
            image = image.resize((size, size), PIL.Image.LANCZOS)
            os.makedirs(os.path.dirname(os.path.join(dest_path, fname)), exist_ok=True)
            image.save(os.path.join(dest_path, fname))
    except:
        pass

jlb.Parallel(n_jobs=num)(jlb.delayed(resize_img)(name) for name in tqdm(image_fnames))
