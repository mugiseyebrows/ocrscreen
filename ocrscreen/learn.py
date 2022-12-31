import os
import cv2
import numpy as np
import time
import argparse
from collections import defaultdict
import hashlib

try:
    from .core import find_character_for_width_range, find_bitmap, count_black_pixels
    from .common import Data, binarize, save_data, pad_left, pad_right, pad_top, pad_bottom, load_data, COLOR_WHITE
except ImportError:
    from core import find_character_for_width_range, find_bitmap, count_black_pixels
    from common import Data, binarize, save_data, pad_left, pad_right, pad_top, pad_bottom, load_data, COLOR_WHITE

def bounding_nonzero(img):
    gray = 255*(img < 128).astype(np.uint8)
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, w, h

def grow(x1, x2, img_width, min_width):
    if img_width <= min_width:
        x1 = 0
        x2 = img_width
        return x1, x2
    while True:
        if x2 - x1 >= min_width:
            return x1, x2
        if x1 > -1:
            x1 -= 1
        if x2 - x1 >= min_width:
            return x1, x2
        if x2 < img_width:
            x2 += 1

def to_freqs(items):
    res = {v:0 for v in items}
    b = 1 / len(items)
    for v in items:
        res[v] += b
    return res

def crop_white(img, padding = 0, min_height = 0, min_width = 0):
    x, y, w, h = bounding_nonzero(img)

    if w == 1:
        t = 1

    x1 = np.clip(x-padding, 0, None)
    x2 = x+w+padding
    y1 = np.clip(y-padding, 0, None)
    y2 = y+h+padding
    img_height, img_width = img.shape

    x1, x2 = grow(x1, x2, img_width, min_width)
    y1, y2 = grow(y1, y2, img_height, min_height)

    return img[y1:y2, x1:x2]

class Sample:
    def __init__(self, txt_path, img_path):
        with open(txt_path, encoding='utf-8') as f:
            text = f.read()
        self._text = text
        self._img = crop_white(binarize(cv2.imread(img_path)))
        self._txt_path = txt_path
        self._img_path = img_path

    def estimate_position(self, c):
        text = self._text
        if c not in text:
            return None
        img_w = self._img.shape[1]
        char_width = img_w / len(text)
        p = text.index(c) * char_width
        dp = char_width * 3
        p1 = np.clip(int(p - dp), 0, img_w)
        p2 = np.clip(int(p + dp), 0, img_w)
        return p1, p2

    def image(self):
        return self._img

    def text(self):
        return self._text

    def contains(self, ch):
        return ch in self._text

    def char_width(self):
        return self._img.shape[1] / len(self._text)

class Samples:

    def __init__(self, data_path):
        samples : list[Sample] = []
        
        for n in os.listdir(data_path):
            if n.endswith('.gt.txt'):
                n_ = n
                n_ = os.path.splitext(n_)[0]
                n_ = os.path.splitext(n_)[0]
                img_n = n_ + ".png"
                img_path = os.path.join(data_path, img_n)
                txt_path = os.path.join(data_path, n)
                samples.append(Sample(txt_path, img_path))
        self._samples = samples
        self._data_path = data_path

    def find(self, ch) -> list[Sample]:
        samples = [s for s in self._samples if s.contains(ch)]
        #return random.sample(samples, 2)
        return samples

    def chars(self) -> str:
        text = []
        for s in self._samples:
            text.append(s.text())
        res = [c for c in list(set("".join(text))) if c not in [' ', '\t', '\n']]
        res.sort()
        return "".join(res)

    # not used
    def char_height(self) -> int:
        return int(np.average([s.image().shape[0] for s in self._samples]))

    def samples(self):
        return self._samples

    def bitmap_score(self, ch, bitmap, tests_max = 20):
        score = 0
        match = []

        pos_tests = 0
        neg_tests = 0

        pos_tests_max = tests_max
        neg_tests_max = tests_max

        match = []

        #nasty_chars = '-ГПФШЪЬглпс'
        nasty_chars = '-'

        for sample in self._samples:
            text = sample.text()
            image = sample.image()
            if ch in text:
                if pos_tests >= pos_tests_max:
                    continue
                x1min, x2max = sample.estimate_position(ch)
                x, y, w, h = find_bitmap(image, x1min, x2max, bitmap)
                if x is not None:
                    score += 1
                pos_tests += 1
                match.append(x is not None)
            else:
                if neg_tests >= neg_tests_max:
                    continue

                x1min, x2max = 0, image.shape[1] - 1
                x, y, w, h = find_bitmap(image, x1min, x2max, bitmap)

                
                if x is not None:
                    img_ = image[:, x-2:x+bitmap.shape[1]+2]
                    t = 1

                if x is not None:
                    if ch not in nasty_chars:
                        score -= 1
                neg_tests += 1

        return score, match

    def find_padding(self, ch, bitmap, axis = 0):
        
        if axis == 0:
            w = bitmap.shape[1]
            padded0 = bitmap
            padded1 = pad_left(bitmap, w + 1, COLOR_WHITE)
            padded2 = pad_right(bitmap, w + 1, COLOR_WHITE)
            padded3 = pad_left(padded2, w + 2, COLOR_WHITE)
        else:
            h = bitmap.shape[0]
            padded0 = bitmap
            padded1 = pad_top(bitmap, h + 1, COLOR_WHITE)
            padded2 = pad_bottom(bitmap, h + 1, COLOR_WHITE)
            padded3 = pad_top(padded2, h + 2, COLOR_WHITE)
        
        items = [(bitmap, *self.bitmap_score(ch, bitmap)) for bitmap in [padded3, padded2, padded1, padded0]]
        scores = [item[1] for item in items]
        index = scores.index(max(scores))

        return items[index][0]

def format_time(s):
    h = int(s / 3600)
    s -= h * 3600
    m = int(s / 60)
    s -= m * 60
    s = int(s)
    return "{:02d}:{:02d}:{:02d}".format(h,m,s)

def uniq_bitmaps(bitmaps):
    u = {hashlib.sha256(bytes(bitmap.flatten())).hexdigest(): bitmap for bitmap in bitmaps}
    return list(u.values())

def learn_space(samples: Samples, bitmaps):

    bitmaps_dict = {ch: bitmap for ch, bitmap in bitmaps}

    space_widths = []

    for sample in samples.find(' '):
        text = sample.text()
        ix = text.index(' ')

        if ix - 1 < 0 or ix + 1 >= len(text):
            continue

        ch1 = text[ix - 1]
        ch2 = text[ix + 1]

        if ch1 not in bitmaps_dict:
            continue
        if ch2 not in bitmaps_dict:
            continue

        x1min, x1max = sample.estimate_position(ch1)
        x2min, x2max = sample.estimate_position(ch2)

        x1, y1, w1, h1 = find_bitmap(sample.image(), x1min, x1max, bitmaps_dict[ch1])
        x2, y2, w2, h2 = find_bitmap(sample.image(), x2min, x2max, bitmaps_dict[ch2])
        if None in [x1, x2]:
            continue

        space_width = x2 - (x1 + w1)

        if space_width < 0:
            continue

        space_widths.append(space_width)

    freqs = {k:v for k,v in to_freqs(space_widths).items() if v > 0.2}

    space_width = min(freqs.keys())

    return space_width


def learn_chars(samples: Samples, char_width_min, char_width_max):

    bitmaps = []
    chars = samples.chars()
    scores = dict()

    t0 = time.time()

    for i, ch in enumerate(chars):
        try:
            #ok = False
            t1 = time.time()
            if i > 0:
                est = (t1 - t0) / (i / len(chars))
            else:
                est = 0
            print("learning {:03d} / {:03d} ({} / {}) {} ".format(i + 1, len(chars), format_time(t1 - t0), format_time(est), ch), end="")
            res = []
            found = samples.find(ch)
            for s1, s2 in zip(found, found[1:]):
                x1, x2, bitmap = find_character_for_width_range(s1, s2, ch, char_width_min, char_width_max)
                print("-" if bitmap is None else "+", end="", flush=True)
                if bitmap is None:
                    continue
                
                cropped = crop_white(bitmap, padding=0, min_width=3, min_height=5)
                res.append(cropped)

                if len(res) > 20:
                    break

            evaluated = [(bitmap, *samples.bitmap_score(ch, bitmap)) for bitmap in uniq_bitmaps(res)]
            evaluated.sort(key = lambda item: item[1], reverse=True)

            if len(evaluated) > 0:

                bitmap0, score0, match0 = evaluated[0]
                
                bitmap1 = samples.find_padding(ch, bitmap0, axis=0)
                bitmap2 = samples.find_padding(ch, bitmap1, axis=1)
    
                bitmaps.append((ch, bitmap2))

                score, match = samples.bitmap_score(ch, bitmap2)
                num = np.sum(match)
                den = len(match)
                print(" {} bitmap(s), score {} / {} ({:.2f}) ".format(1, num, den, num / den))
            else:
                print(" failed to learn {}".format(ch))

        except ValueError as e:
            print(ch, e)
    
    return bitmaps, scores

def main():
    example_text = """examples:
  ocrscreen-learn path/to/samples -o path/to/database
"""
    description = "learn character bitmaps from sample images"
    parser = argparse.ArgumentParser(description=description, prog='ocrscreen-learn', epilog=example_text, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples_path", help="path to samples")
    parser.add_argument("--char-width", nargs=2, type=int, help="minumum and maximum char width in pixels")
    parser.add_argument("-o", "--output", help="path to save data", required=True)
    
    args = parser.parse_args()
    
    samples_path = args.samples_path

    samples = Samples(samples_path)

    char_heigth = samples.char_height()

    char_width_min = 3
    char_width_max = int(char_heigth * 1.5)

    if args.char_width is not None:
        char_width_min, char_width_max = args.char_width

    bitmaps, scores = learn_chars(samples, char_width_min, char_width_max)
    space_width = learn_space(samples, bitmaps)
    data = Data(bitmaps, space_width, scores)
    save_data(args.output, data)
    print("data saved as {}".format(args.output))

if __name__ == "__main__":
    main()