import argparse
import cv2
import numpy as np

from os import listdir
from os import makedirs
from os.path import abspath
from os.path import dirname
from os.path import splitext
from os.path import isdir
from os.path import join
from math import floor, ceil


def equalize_gray_clahe(img, clipLimit=2.0, tileGridSize=(8, 8), show_clahed=False):
    """Equalize a grayscale image using Contrast Limited Adaptive Histogram Equalization

    @param clipLimit Threshold for contrast limiting.

    @param tileGridSize Size of grid for histogram equalization. Input image
    will be divided into equally sized rectangular tiles. tileGridSize defines
    the number of tiles in row and column.
    """
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    img_clahed = clahe.apply(img)
    return img_clahed


def crop_circle(img_orig, x, y, radius, pad, show_crop=False):
    """Cut the box of the circle (x,y,r) from img_orig
    """

    # the values must be integers
    x = int(x)
    y = int(y)
    radius += pad
    radius = int(radius)
    #  if show_crop:
    #  print(f"x {x} y {y} r {radius} [{x-radius}:{x+radius}, {y-radius}:{y+radius}]")

    # if the circle goes out of the image, assume is a bad one

    left = y - radius
    if left < 0:
        return None

    right = y + radius
    if right >= img_orig.shape[0]:
        return None

    top = x - radius
    if top < 0:
        return None

    bottom = x + radius
    if bottom > img_orig.shape[1]:
        return None

    # crop the circle
    piece = img_orig[left:right, top:bottom, :]

    if show_crop:
        cv2.imshow("Piece", piece)
        cv2.waitKey(0)

    return piece


def find_coins_circles(image_name_full, hcParam, show_clahed=False):
    """Find circles from this image, using the param passed

    Return a list of circles

    Use CLAHE to equalize the image before applying HoughCircles
    """

    img_orig = cv2.imread(image_name_full, cv2.IMREAD_GRAYSCALE)

    img_orig = cv2.GaussianBlur(img_orig, (5, 5), 5)

    #  img_orig = cv2.GaussianBlur(img_orig, (3, 3), 6)
    #  cv2.imshow("Original", img_orig)
    #  cv2.waitKey(0)

    #  circles = cv2.HoughCircles(image=img_orig, **hcParam)

    img_clahed = equalize_gray_clahe(img_orig)
    scale = 800 / max(img_clahed.shape[0:2])
    img_clahed_small = cv2.resize(
        img_clahed, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )
    if show_clahed:
        cv2.imshow("Clahed", img_clahed_small)
        #  cv2.waitKey(0)

    circles = cv2.HoughCircles(image=img_clahed, **hcParam)

    #  if not circles is None:
    #  print(f"Circles found {circles.shape[1]}")
    #  print(f"With {hcParam}")

    #  print(f'shape {circles[0, :, :].shape}')
    if not circles is None:
        return circles[0, :, :]
    else:
        return None


def do_extraction(raw_folder_full, save_folder_full, pad):
    """Extract all circles from images, pad them

    Input images: all folders in raw_folder_full
    Output images: same folders, in save_folder_full
    """
    show_crop = False
    show_clahed = False
    save_results = True

    #  for label in ["10c"]:
    for label in listdir(raw_folder_full):
        print(f"\nLABEL: {label}\n")
        full_label = join(raw_folder_full, label)

        output_labeled_folder_full = abspath(join(save_folder_full, label))
        print(f"output_labeled_folder_full {output_labeled_folder_full}")
        if not isdir(output_labeled_folder_full):
            makedirs(output_labeled_folder_full)

        #  for image_name in listdir(full_label)[:2]:
        for image_name in listdir(full_label):

            #  print(f"Image name: {image_name}")
            image_name_full = join(full_label, image_name)
            #  print(f"Image name full: {image_name_full}")

            image_name_base, image_name_ext = splitext(image_name)
            #  print(f"Image name base: {image_name_base} ext {image_name_ext}")
            image_name_base_full = join(output_labeled_folder_full, image_name_base)
            print(f"Image name base full: {image_name_base_full}")

            # EXTRACT circles from image
            circles = find_coins_circles(image_name_full, getHCpar(label), show_clahed)
            print(f"\tCircles found {circles.shape[0]}")

            img_orig = cv2.imread(image_name_full)
            for i, (x, y, radius) in enumerate(circles):
                img_piece = crop_circle(img_orig, x, y, radius, pad, show_crop)

                if not img_piece is None:
                    if save_results:
                        out_name_full = f"{image_name_base_full}_{i}{image_name_ext}"
                        #  print(f"\tout_name_full {out_name_full}")
                        cv2.imwrite(out_name_full, img_piece)


def getHCpar(label):
    """Return a dict of params for HoughCircles
    """
    if label in ["2e"]:
        param = {
            "method": cv2.HOUGH_GRADIENT,
            "dp": 1.0,
            "minDist": 140,
            #  "minRadius": 170,
            "minRadius": 170,
            "maxRadius": 230,
            #  "maxRadius": 320,
            #  "param1": 140,
            "param1": 135,
            "param2": 60,
        }
    elif label in ["1e"]:
        # 1e minDist 40 minRadius 165 maxRadius 190 hcParam1 70 hcParam2 25
        param = {
            "method": cv2.HOUGH_GRADIENT,
            "dp": 1.0,
            #  "minDist": 40,
            "minDist": 200,
            "minRadius": 165,
            "maxRadius": 190,
            "param1": 70,
            "param2": 25,
        }
    elif label in ["1c"]:
        # 1c minDist 150 minRadius 125 maxRadius 160 hcParam1 65 hcParam2 30
        param = {
            "method": cv2.HOUGH_GRADIENT,
            "dp": 1.0,
            "minDist": 150,
            #  "minRadius": 125,
            #  "maxRadius": 160,
            "minRadius": 110,
            "maxRadius": 140,
            "param1": 65,
            "param2": 30,
        }
    elif label in ["2c"]:
        # 2c TODO
        param = {
            "method": cv2.HOUGH_GRADIENT,
            "dp": 1.0,
            #  "minDist": 150,
            "minDist": 200,
            #  "minRadius": 125,
            "minRadius": 135,
            "maxRadius": 160,
            "param1": 65,
            "param2": 30,
        }
    elif label in ["5c"]:
        # 5c minDist 150 minRadius 150 maxRadius 170 hcParam1 60 hcParam2 25
        param = {
            "method": cv2.HOUGH_GRADIENT,
            "dp": 1.0,
            "minDist": 150,
            "minRadius": 150,
            "maxRadius": 170,
            "param1": 60,
            "param2": 25,
        }
    elif label in ["10c"]:
        # 10c minDist 150 minRadius 140 maxRadius 160 hcParam1 100 hcParam2 41
        param = {
            "method": cv2.HOUGH_GRADIENT,
            "dp": 1.0,
            "minDist": 150,
            "minRadius": 140,
            "maxRadius": 160,
            #  "param1": 100,
            "param1": 70,
            #  "param2": 40,
            "param2": 30,
        }
    elif label in ["20c"]:
        # 20c
        param = {
            "method": cv2.HOUGH_GRADIENT,
            "dp": 1.0,
            #  "minDist": 150,
            "minDist": 200,
            "minRadius": 150,
            "maxRadius": 180,
            "param1": 65,
            "param2": 30,
        }
    elif label in ["50c"]:
        # 50c minDist 150 minRadius 165 maxRadius 200 hcParam1 80 hcParam2 45
        param = {
            "method": cv2.HOUGH_GRADIENT,
            "dp": 1.0,
            "minDist": 150,
            #  "minRadius": 165,
            "minRadius": 150,
            "maxRadius": 200,
            "param1": 80,
            "param2": 45,
        }
    else:
        # vague default params that should find something
        param = {
            "method": cv2.HOUGH_GRADIENT,
            "dp": 1.0,
            "minDist": 40,
            "minRadius": 100,
            "maxRadius": 230,
            "param1": 75,
            "param2": 40,
        }

    return param


def main():
    dir_file = abspath(dirname(__file__))

    #  datafolder = "./mydata"
    datafolder = "."

    raw_folder = "./raw/raw4"
    raw_folder_full = abspath(join(dir_file, datafolder, raw_folder))
    print(f"raw_folder_full {raw_folder_full}")

    save_folder = "./parsed"
    save_folder_full = abspath(join(dir_file, datafolder, save_folder))
    print(f"save_folder_full {save_folder_full}")

    # pad around coin when cropping
    pad = 20

    do_extraction(raw_folder_full, save_folder_full, pad)


if __name__ == "__main__":
    main()
