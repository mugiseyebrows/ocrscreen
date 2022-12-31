import os
import cv2
from collections import defaultdict
import numpy as np

try:
    from .common import binarize, load_data
    from .core import recognize
except ImportError:
    from common import binarize, load_data
    from core import recognize

def read_binarize_write(img_path, out_path = None):
    img = binarize(cv2.imread(img_path))
    if out_path is None:
        basename = os.path.basename(img_path)
        name, ext = os.path.splitext(basename)
        out_path = os.path.join(os.path.dirname(img_path), name + "-bin" + ext)
    cv2.imwrite(out_path, img)

"""
def pad_value(mat, h, w, v):
    res = np.zeros((h, w))
    res[:, :] = v
    res[:mat.shape[0], :mat.shape[1]] = mat
    return res
"""


def pad_left(mat, w, v):
    if mat.shape[1] == w:
        return mat
    h = mat.shape[0]
    res = np.zeros((h, w), dtype=mat.dtype)
    res[:, :] = v
    res[:mat.shape[0], :mat.shape[1]] = mat
    return res

def pad_bottom(mat, h, v):
    if mat.shape[0] == h:
        return mat
    w = mat.shape[1]
    res = np.zeros((h, w), dtype=mat.dtype)
    res[:, :] = v
    res[:mat.shape[0], :mat.shape[1]] = mat
    return res

def hstack_padded(mats, v):
    h = max([mat.shape[0] for mat in mats])
    mats = [pad_bottom(mat, h, v) for mat in mats]
    return np.hstack(mats)

def vstack_padded(mats, v):
    w = max([mat.shape[1] for mat in mats])
    mats = [pad_left(mat, w, v) for mat in mats]
    return np.vstack(mats)


def to_bank(data, path):
    os.makedirs(path, exist_ok=True)
    grouped = defaultdict(list)
    for ch, bitmap in data.bitmaps:
        grouped[ch].append(bitmap)
    for i, (ch, group) in enumerate(grouped.items()):
        group_path = os.path.join(path, "{:03d}".format(i))
        os.makedirs(group_path, exist_ok=True)
        for j, bitmap in enumerate(group):
            img_path = os.path.join(group_path, "{:03d}.png".format(j))
            cv2.imwrite(img_path, bitmap)
        txt_path = os.path.join(group_path, "id.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(ch)

def main():

    if 0:
        img_path = "/home/overloop/Pictures/sample.png"
        read_binarize_write(img_path)
        img_path = "sample-068.png"
        read_binarize_write(img_path)
    
    if 0:
        for i in [107, 61]:
            path = "/home/overloop/tesstrain/data/scr-ground-truth/sample-{:03d}.png".format(i)
            read_binarize_write(path)

    if 0:
        grouped = defaultdict(list)
        for ch, bitmap in data.bitmaps:
            grouped[ch].append(bitmap)
        
        imgs = [hstack_padded(group, 128) for ch, group in grouped.items()]
        img = vstack_padded(imgs, 128)
        cv2.imwrite("scr.png", img)

    if 0:
        img_path = "sample.png"
        read_binarize_write(img_path)

    #img = ImageGrab.grab()

    t = 1

    """
    data_path = "/home/overloop/ocrscreen/scr-bitmaps"
    data = load_data(data_path)
    data.save_as_image("bitmaps.png")
    """

    """
    bank_path = "bitmaps"
    to_bank(data, bank_path)
    """



if __name__ == "__main__":
    main()