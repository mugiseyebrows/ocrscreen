import cv2
#import pickle
from collections import defaultdict
import os
import numpy as np
import json

COLOR_WHITE = 255

try:
    from .core import count_black_pixels
except ImportError:
    from core import count_black_pixels

def binarize(img, threshold = 128):
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_out = cv2.threshold(im_gray, threshold, 255, cv2.THRESH_BINARY)
    return img_out

def pad_left(mat, w, v):
    if mat.shape[1] == w:
        return mat
    h = mat.shape[0]
    res = np.zeros((h, w), dtype=mat.dtype)
    res[:, :] = v
    dw = w - mat.shape[1]
    res[:mat.shape[0], dw:] = mat
    return res

def pad_right(mat, w, v):
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

def pad_top(mat, h, v):
    if mat.shape[0] == h:
        return mat
    w = mat.shape[1]
    res = np.zeros((h, w), dtype=mat.dtype)
    res[:, :] = v
    dh = h - mat.shape[0]
    res[dh:, :mat.shape[1]] = mat
    return res

def hstack_padded(mats, v):
    h = max([mat.shape[0] for mat in mats])
    mats = [pad_bottom(mat, h, v) for mat in mats]
    return np.hstack(mats)

def vstack_padded(mats, v):
    w = max([mat.shape[1] for mat in mats])
    mats = [pad_right(mat, w, v) for mat in mats]
    return np.vstack(mats)


class Data:
    def __init__(self, bitmaps = None, space_width = None, scores = None):
        if bitmaps is None:
            bitmaps = []
        self.black_pixels = dict()
        if space_width is None:
            space_width = 2
        self.bitmaps = bitmaps
        self.space_width = space_width
        self.scores = scores
        self.char_height = None
        if len(self.bitmaps) > 0:
            self.calculate_black_pixels()
            self.calculate_char_height()

    def _group_bitmaps(self):
        grouped = defaultdict(list)
        for ch, bitmap in self.bitmaps:
            grouped[ch].append(bitmap)
        chs = list(grouped.keys())
        chs.sort()
        return [(ch, grouped[ch]) for ch in chs]

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        grouped = self._group_bitmaps()
        for i, (ch, group) in enumerate(grouped):
            if ch in [' ', '/', '\\', ':', '.', '?', '*', '<', '>', '|', '"']:
                ch_ = "{:03d}".format(i)
            else:
                ch_ = ch
                if ch.islower():
                    ch_ += ' lc'
                elif ch.isupper():
                    ch_ += ' uc'
            group_path = os.path.join(path, ch_)
            os.makedirs(group_path, exist_ok=True)
            for j, bitmap in enumerate(group):
                img_path = os.path.join(group_path, "{:03d}.png".format(j))
                cv2.imwrite(img_path, bitmap)
            txt_path = os.path.join(group_path, "id.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(ch)
        
        json_path = os.path.join(path, "data.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "scores": self.scores,
                "space_width": self.space_width,
                "char_height": self.char_height
            }, f, indent=1, ensure_ascii=False) 

        img_path = os.path.join(path, "bitmaps.png")
        self.save_as_image(img_path)

    def save_as_image(self, img_path):
        imgs = [hstack_padded(group, 128) for ch, group in self._group_bitmaps()]
        cv2.imwrite(img_path, vstack_padded(imgs, 128))

    def calculate_char_height(self):
        char_height = 0
        for ch, bitmap in self.bitmaps:
            char_height = max(char_height, bitmap.shape[0])
        self.char_height = char_height

    def calculate_black_pixels(self):
        black_pixels = dict()
        for ch, bitmap in self.bitmaps:
            black_pixels[ch] = count_black_pixels(bitmap)
        self.black_pixels = black_pixels

    def load(self, data_path):
        for d in os.listdir(data_path):
            dirpath = os.path.join(data_path, d)
            if not os.path.isdir(dirpath):
                continue
            bitmaps = []
            ch = None
            for n in os.listdir(dirpath):
                filename = os.path.join(dirpath, n)
                ext = os.path.splitext(n)[1]
                if ext == '.png':
                    bitmap = cv2.imread(filename, flags=cv2.IMREAD_GRAYSCALE)
                    #print("shape", bitmap.shape)
                    bitmaps.append(bitmap)
                elif n == 'id.txt':
                    with open(filename, encoding='utf-8') as f:
                        ch = f.read()
            for bitmap in bitmaps:
                self.bitmaps.append((ch, bitmap))
            """
            if len(bitmaps) > 0:
                self.black_pixels[ch] = count_black_pixels(bitmap)
            """
        
        self.calculate_char_height()
        self.calculate_black_pixels()

        try:
            json_path = os.path.join(data_path, "data.json")
            with open(json_path, encoding='utf-8') as f:
                data = json.load(f)
            self.scores = data['scores']
            self.space_width = data['space_width']
            #self.char_height = data['char_height']
        except Exception as e:
            print(e)


def save_data(path, data):
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    """
    data.save(path)

def load_data(path) -> Data:
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data
    """
    data = Data()
    data.load(path)
    return data