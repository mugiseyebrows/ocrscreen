import numpy as np

####################################### LEARN

ctypedef unsigned char uint8_t

cdef has_black_pixels(uint8_t [:, :] img1, int x1, int y1, int w, int h, int n):
    cdef int cnt = 0
    cdef int x, y

    if y1 + h > img1.shape[0]:
        return False
    if x1 + w > img1.shape[1]:
        return False

    for dy in range(h):
        for dx in range(w):
            y = y1 + dy
            x = x1 + dx
            if img1[y, x] == 0:
                cnt += 1
                if cnt >= n:
                    return True
    return False

def count_black_pixels(cv_img):
    cdef uint8_t [:, :] img = cv_img
    cdef int ih, iw
    ih, iw = img.shape[:2]
    cdef int x, y
    cdef int cnt = 0
    for y in range(ih):
        for x in range(iw):
            if img[y, x] == 0:
                cnt += 1
    return cnt

cdef compare_range(uint8_t [:, :] img1, uint8_t [:, :] img2, int x1, int y1, int x2, int y2, int w, int h):
    cdef int ih1, ih2, b
    cdef int x1_, y1_, x2_, y2_
    b = 0
    if y1 + h > img1.shape[0]:
        return False
    if x1 + w > img1.shape[1]:
        return False
    if y2 + h > img2.shape[0]:
        return False
    if x2 + w > img2.shape[1]:
        return False
    for dy in range(0, h):
        for dx in range(0, w):
            y1_ = y1 + dy
            x1_ = x1 + dx
            y2_ = y2 + dy
            x2_ = x2 + dx
            if img1[y1_, x1_] != img2[y2_, x2_]:
                return False

    return True

def find_character(cv_img1, x1min, x1max, cv_img2, x2min, x2max, char_width):

    stat = []
    
    cdef uint8_t [:, :] img1 = cv_img1
    cdef uint8_t [:, :] img2 = cv_img2

    h1 = img1.shape[0]
    h2 = img2.shape[0]
    h = min(h1, h2)

    def y_shifts(img_h, h):
        return range(img_h - h + 1)

    black_thr = 3

    for x1 in range(x1min, x1max):
        for y1 in y_shifts(h1, h):
            if not has_black_pixels(img1, x1, y1, char_width, h, black_thr):
                continue
            for x2 in range(x2min, x2max):
                for y2 in y_shifts(h2, h):
                    if not has_black_pixels(img2, x2, y2, char_width, h, black_thr):
                        continue
                    if compare_range(img1, img2, x1, y1, x2, y2, char_width, h):
                        bitmap = cv_img1[y1:y1+h, x1:x1+char_width]
                        stat.append((x1, x2, bitmap))

    if len(stat) > 0:
        x1, x2, bitmap = stat[0]
        return x1, x2, bitmap
    
    return None, None, None

def find_bitmap(cv_img1, x1min, x1max, cv_bitmap):
    cdef uint8_t [:, :] img1 = cv_img1
    cdef uint8_t [:, :] bitmap = cv_bitmap
    cdef int ih = img1.shape[0]
    cdef int bh = bitmap.shape[0]
    for y in range(ih - bh + 1):
        for x in range(x1min, x1max):
            if compare(img1, x, y, bitmap):
                return x, y, bitmap.shape[1], bitmap.shape[0]
    return None, None, None, None

def find_character_for_width_range(sample1, sample2, ch, char_width_min, char_width_max):
    x1min, x1max = sample1.estimate_position(ch)
    x2min, x2max = sample2.estimate_position(ch)
    cv_img1 = sample1.image()
    cv_img2 = sample2.image()
    for char_width in reversed(range(char_width_min, char_width_max + 1)):
        x1, x2, bitmap = find_character(cv_img1, x1min, x1max, cv_img2, x2min, x2max, char_width)
        if bitmap is not None:
            return x1, x2, bitmap
    
    return None, None, None

####################################### RECOGNIZE

cdef compare(uint8_t [:, :] img, int x, int y, uint8_t [:, :] bitmap):
    cdef int dx, dy
    cdef int ih, iw
    cdef int bh, bw
    ih, iw = img.shape[:2]
    bh, bw = bitmap.shape[:2]
    if y + bh > ih or x + bw >= iw:
        return False
    for dy in range(bh):
        for dx in range(bw):
            if img[y+dy, x+dx] != bitmap[dy, dx]:
                return False
    return True

def recognize(cv_img, bitmaps):
    res = []
    cdef uint8_t [:, :] img = cv_img
    cdef size_t img_h, img_w, bitmap_h, bitmap_w, x, y, dx, dy
    cdef uint8_t [:, :] bitmap

    if isinstance(bitmaps, dict):
        bitmaps_ = bitmaps.items()
    else:
        bitmaps_ = bitmaps

    for ch, cv_bitmap in bitmaps_:
        bitmap = cv_bitmap
        img_h, img_w = img.shape[:2]
        bitmap_h, bitmap_w = bitmap.shape[:2]
        for y in range(img_h-bitmap_h):
            for x in range(img_w-bitmap_w):
                if compare(img, x, y, bitmap):
                    # x, y, w, h, xmid, ymid, ch
                    res.append((x, y, bitmap_w, bitmap_h, x + bitmap_w / 2, y + bitmap_h / 2, ch))
    return res

def asarray(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    a = np.asarray(img, dtype='uint8')
    cdef uint8_t [:,:,:] av = a
    cdef int x, y
    cdef int w, h
    cdef uint8_t r, g, b
    h, w = a.shape[:2]
    for y in range(h):
        for x in range(w):
            r = av[y,x,0]
            g = av[y,x,1]
            b = av[y,x,2]

            av[y,x,0] = b 
            av[y,x,1] = g
            av[y,x,2] = r
    return a
