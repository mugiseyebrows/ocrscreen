# ocrscreen

ocr for recognizing text on computer screen

## Install
```
pip install ocrscreen
```

## Use

To create recognition database you need to create few (tens to hundreds) training samples in form of file pairs: png image file containing image of one line of text, and text file containing same line of text. Image and text should have same name except extension: `.png` for image and `.gt.txt` for text. Each character you want to recognize should be present in at least three samples.
 
If you have samples you can convert it into database with this command:

```
ocrscreen-learn path/to/samples -o path/to/database
```

Recognintion database is a list of directories, each directory contains `.png` files (one or many) representing character and `.id` file with text containing character (to avoid filesystem limitations). Directory name and file name does not matter. This form allows easy tuning and troubleshooting recognition problems. 

When you have database you can run ocr on image

```
ocrscreen-recognize -d path/to/database -i path/to/image
```

or on screen
```
ocrscreen-recognize -d path/to/database --screen
```

or on portion of screen with --rect x y w h

```
ocrscreen-recognize -d path/to/database --screen --rect 10 10 640 480
```

Inspect `samples` and `database` directory in the sources to get better understanding of data format.

## Notes

This ocr uses black and white bitmaps as search pattern to search on binarized image and only perfect equality counts as match. It doesn't use dpi and neural networks, it cannot recognize scanned text or text in photo images, it is only for recognizing perfect digital text.

If you have linux and wayland you need to install `pyscreenshot` package.
```
pip install pyscreenshot
```