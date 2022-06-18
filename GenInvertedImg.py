import math
import os
from PIL import Image, ImageOps

image_paths = ["./OCR_Image1/Valid_Set/G/G-2 (10).png",
               "./OCR_Image1/Train_Set/G/G-2 (11).png",
               "./OCR_Image1/Train_Set/R/R (10).png",
               "./OCR_Image1/Train_Set/V/V (1).png",
               "./OCR_Image1/Train_Set/V/V (9).png",
               "./OCR_Image1/Valid_Set/T/T (5).png"]

for path in image_paths:
    im = Image.open(path)
    im_invert = ImageOps.invert(im)
    invert_path = path[0:len(path) - 4] + "_inverted.png"
    im_invert.save(invert_path)
