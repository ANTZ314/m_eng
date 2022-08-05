#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
Splits all larger image files in a directory into ['d'x'd'] patches.

Create directory for tiles: "mkdir tiles"

filename: the image file name
d: the tile size
dir_in: the path to the directory containing the image
dir_out: the directory where tiles will be outputted
"""

from PIL import Image
from itertools import product
import os

def tile(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size
    
    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)


def main():
    filename = "Test_Growth.jpg"
    d = 32
    dir_in  = "/home/antz/Desktop/models/0_NOTES/test"
    dir_out = "/home/antz/Desktop/models/0_NOTES/test/tiles"

    tile(filename, dir_in, dir_out, d)

if __name__ == "__main__": main()

