import cv2
import numpy as np
import time
import sys
import argparse
from io import BytesIO

try:
    import pyscreenshot as ImageGrab
except ImportError:
    from PIL import ImageGrab
from PIL import Image

try:
    from .common import binarize, load_data
    from .core import recognize, asarray
except ImportError:
    from common import binarize, load_data
    from core import recognize, asarray

def x_y_w_ch(e):
    return e[0], e[1], e[2], e[-1]


def filter_overlayed(line, black_pixels, verbose):
    filtered = line[:]
    filtered.sort(key=lambda item: item[0])
    loop = 0
    while True:
        loop += 1
        #print("loop {}".format(loop))
        ixs_to_delete = set()
        for i in range(len(filtered)-1):
            x1, y1, w1, ch1 = x_y_w_ch(filtered[i])
            x2, y2, w2, ch2 = x_y_w_ch(filtered[i+1])
            if x1 + w1 - 2 > x2:
                if black_pixels[ch1] < black_pixels[ch2]:
                    ix = i
                    ch = ch1
                else:
                    ix = i + 1
                    ch = ch2
                if ch1 == ch2:
                    if verbose:
                        print("two bitmaps for {} matched at ({} {}), ({} {})".format(ch, x1, y1, x2, y2), file=sys.stderr)
                else:
                    if verbose:
                        print("{} at ({} {}), {} at ({} {}), removing {}".format(ch1, x1, y1, ch2, x2, y2, ch), file=sys.stderr)
                ixs_to_delete.add(ix)
        
        if len(ixs_to_delete) == 0:
            return filtered
        
        filtered = [e for i, e in enumerate(filtered) if i not in ixs_to_delete]

def split_to_lines(matched, char_height):
    matched.sort(key=lambda item: item[5])
    dists = [abs(item1[5] - item2[5]) for item1, item2 in zip(matched, matched[1:])]
    breaks = [0] + [i + 1 for i,dist in enumerate(dists) if dist > char_height / 2] + [len(matched)]
    lines = [matched[begin:end] for begin, end in zip(breaks, breaks[1:])]
    return lines

def map_to_chars(line, space_width):
    chars = []
    pos = 0
    for j, (x, y, w, h, xmid, ymid, ch) in enumerate(line):
        if j > 0 and x - pos >= space_width:
            chars.append(' ')
        chars.append(ch)
        pos = x + w
    return ''.join(chars)

def find_lines(matched, char_height, space_width, black_pixels, verbose):
    lines = [
        map_to_chars(filter_overlayed(line, black_pixels, verbose), space_width)
        for line in split_to_lines(matched, char_height)
    ]
    return lines

import itertools


class Writer:
    def __init__(self, args):
        self._args = args
        self._open()
        self._f = None

    def _open(self):
        args = self._args
        if args.output:
            self._f = open(args.output, 'w', encoding='utf-8')

    def write(self, lines):
        f = self._f
        args = self._args
        text = "\n".join(lines) + "\n"
        if f:
            f.write(text)
        else:
            if args.utf_8:
                sys.stdout.buffer.write(text.encode("utf-8"))
            else:
                print(text, end="")

    def close(self):
        f = self._f
        if f:
            f.close()
            self._f = None

def main():
    example_text = """examples:
  ocrscreen-recognize -d path/to/database -i path/to/image
  ocrscreen-recognize -d path/to/database -i path/to/image -o path/to/text
  ocrscreen-recognize -d path/to/database --screen
  ocrscreen-recognize -d path/to/database --screen --rect 10 10 640 480
"""
    description = "recognize characters in image"
    parser = argparse.ArgumentParser(description=description, prog='ocrscreen-recognize', epilog=example_text, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", "-d", required=True, help='path to database directory')
    parser.add_argument("--image", "-i", nargs="+", help='image to recognize')
    parser.add_argument("--screen", action='store_true', help='take screenshot and recognieze characters on it')
    parser.add_argument("--rect", "-r", nargs=4, type=int, help="portion of screen to recognize")
    parser.add_argument("--save-image", help='path to save screenshot')
    parser.add_argument("--save-bin", help='path to save binarized image')
    parser.add_argument("--verbose", "-v", action='store_true', help='print information about overlayed matches')
    parser.add_argument("--output", "-o", help='path to store recognized text')
    parser.add_argument("--utf-8", action='store_true', help='print to stdout in utf-8')

    # todo: invert colors
    # todo: remove background

    args = parser.parse_args()

    data = load_data(args.data)

    sources = []

    if args.image:
        for img_path in args.image:
            sources.append(img_path)
    if args.screen:
        img = ImageGrab.grab()
        if args.rect:
            x, y, w, h = args.rect
            img = img.crop((x,y,x+w,y+h))

        #t1 = time.time()
        img = asarray(img)
        #t2 = time.time()
        #print("converted in {:.2f}s".format(t2 - t1))

        sources.append(img)

    writer = Writer(args)
    
    for source in sources:
        t1 = time.time()
        if isinstance(source, str):
            img_orig = cv2.imread(img_path)
            img = binarize(img_orig)
        else:
            img_orig = source
            img = binarize(source)
        
        if args.save_image:
            cv2.imwrite(args.save_image, img_orig)

        if args.save_bin:
            cv2.imwrite(args.save_bin, img)

        matched = recognize(img, data.bitmaps)
        t2 = time.time()
        lines = find_lines(matched, data.char_height, data.space_width, data.black_pixels, args.verbose)
        t3 = time.time()
        writer.write(lines)

    writer.close()

if __name__ == "__main__":
    main()