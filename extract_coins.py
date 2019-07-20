import argparse
import cv2
import numpy as np

from os import listdir
from os import makedirs
from os import sep
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
    #  print(f"Piece shape {piece.shape}")

    if piece.shape[0] % 2 != 0 or piece.shape[1] % 2 != 0:
        print(f"Non-even shape {piece.shape}")

    if piece.shape[0] != piece.shape[1]:
        print(f"Non-square shape {piece.shape}")

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

    #  cv2.imshow("Blurred", img_orig)
    #  cv2.waitKey(0)

    img_clahed = equalize_gray_clahe(img_orig)

    circles = cv2.HoughCircles(image=img_clahed, **hcParam)

    if show_clahed:
        img_orig = cv2.imread(image_name_full)
        if not circles is None:
            for x, y, r in circles[0, :, :]:
                cv2.circle(img_orig, (x, y), r, thickness=5, color=(0, 255, 0))
                descr = f"({int(x)}, {int(y)}): {int(r)}"
                xt = int(x - r - 10)
                yt = int(y - r - 40)
                cv2.putText(
                    img_orig,
                    descr,
                    (xt, yt),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2,
                    color=(0, 0, 255),
                    thickness=3,
                )

        scale = 1000 / max(img_orig.shape[0:2])
        img_orig_small = cv2.resize(
            img_orig, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )

        cv2.imshow("Circled", img_orig_small)
        key = cv2.waitKey(0)
        #  print(f"{key}") # key == 13 ENTER; key == 32 SPACE;
        if key == 98:  # b
            print(f"\t\t\tBad circles extracted")
        elif key == 97:  # a
            print(f"\t\t\tBad circles extracted in previous one")

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
    #  show_clahed = False
    show_clahed = True
    save_results = False
    #  save_results = True

    raw_folder_tag = raw_folder_full.split(sep)[-1]
    print(f"\nRAW folder tag: {raw_folder_tag}\n")

    #  for label in listdir(raw_folder_full):
    for label in ["2e"]:
        print(f"\nLABEL: {label}\n")
        full_label = join(raw_folder_full, label)

        output_labeled_folder_full = abspath(join(save_folder_full, label))
        print(f"output_labeled_folder_full {output_labeled_folder_full}")
        if not isdir(output_labeled_folder_full):
            makedirs(output_labeled_folder_full)

        #  for image_name in listdir(full_label)[:4]:
        for image_name in sorted(listdir(full_label)):

            #  print(f"Image name: {image_name}")
            image_name_full = join(full_label, image_name)
            #  print(f"Image name full: {image_name_full}")

            image_name_base, image_name_ext = splitext(image_name)
            #  print(f"Image name base: {image_name_base} ext {image_name_ext}")
            image_name_base_full = join(output_labeled_folder_full, image_name_base)
            print(f"Image name base full: {image_name_base_full}")

            # EXTRACT circles from image
            circles = find_coins_circles(
                image_name_full, getHCpar(label, raw_folder_tag), show_clahed
            )
            print(f"\tCircles found {circles.shape[0]}")

            img_orig = cv2.imread(image_name_full)
            for i, (x, y, radius) in enumerate(circles):
                img_piece = crop_circle(img_orig, x, y, radius, pad, show_crop)

                if not img_piece is None:
                    if save_results:
                        out_name_full = f"{image_name_base_full}_{i}{image_name_ext}"
                        #  print(f"\tout_name_full {out_name_full}")
                        cv2.imwrite(out_name_full, img_piece)
                else:
                    print(f"\tBad circle {i}")


def getHCpar(label, raw_folder_tag):
    """Return a dict of params for HoughCircles

    Different params for different photo sets
    """
    if raw_folder_tag in ["raw4"]:
        if label in ["2e"]:
            param = {
                "method": cv2.HOUGH_GRADIENT,
                "dp": 1.0,
                "minDist": 140,
                "minRadius": 150,
                "maxRadius": 230,
                #  "param1": 140,
                "param1": 135,
                "param2": 60,
            }
    elif raw_folder_tag in ["raw3"]:
        if label in ["1c"]:
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
                #  "maxRadius": 170,
                "maxRadius": 165,
                "param1": 60,
                "param2": 25,
            }
        elif label in ["20c"]:
            # 20c
            param = {
                "method": cv2.HOUGH_GRADIENT,
                "dp": 1.0,
                #  "minDist": 150,
                #  "minDist": 200,
                "minDist": 250,
                #  "minRadius": 150,
                "minRadius": 155,
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
    elif raw_folder_tag in ["raw4"]:
        if label in ["2e"]:
            param = {
                "method": cv2.HOUGH_GRADIENT,
                "dp": 1.0,
                "minDist": 140,
                "minRadius": 170,
                "maxRadius": 230,
                #  "param1": 140,
                "param1": 135,
                "param2": 60,
            }
        elif label in ["1e"]:
            # 1e minDist 40 minRadius 165 maxRadius 190 hcParam1 70 hcParam2 25
            param = {
                "method": cv2.HOUGH_GRADIENT,
                "dp": 1.0,
                #  "minDist": 200,
                "minDist": 210,
                #  "minRadius": 165,
                "minRadius": 155,
                "maxRadius": 190,
                "param1": 70,
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
    elif raw_folder_tag in ["raw5"]:
        if label in ["1e"]:
            param = {
                "method": cv2.HOUGH_GRADIENT,
                "dp": 1.0,
                #  "minDist": 200,
                "minDist": 210,
                "minRadius": 150,
                "maxRadius": 160,
                "param1": 70,
                "param2": 25,
            }
        elif label in ["5c"]:
            # 5c minDist 150 minRadius 150 maxRadius 170 hcParam1 60 hcParam2 25
            param = {
                "method": cv2.HOUGH_GRADIENT,
                "dp": 1.0,
                "minDist": 150,
                "minRadius": 140,
                "maxRadius": 155,
                "param1": 60,
                "param2": 25,
            }
        elif label in ["2c"]:
            # 2c TODO
            param = {
                "method": cv2.HOUGH_GRADIENT,
                "dp": 1.0,
                "minDist": 200,
                "minRadius": 125,
                #  "minRadius": 135,
                "maxRadius": 160,
                "param1": 65,
                "param2": 30,
            }
    else:
        # vague default params that might find something, but should not be used
        param = {
            "method": cv2.HOUGH_GRADIENT,
            "dp": 1.0,
            "minDist": 20,
            "minRadius": 180,
            "maxRadius": 240,
            "param1": 75,
            "param2": 40,
        }

    return param


def main():
    dir_file = abspath(dirname(__file__))

    #  datafolder = "./mydata"
    datafolder = "."

    save_folder = "./parsed"
    save_folder_full = abspath(join(dir_file, datafolder, save_folder))
    print(f"save_folder_full {save_folder_full}")

    # pad around coin when cropping
    pad = 30

    raw_folder_list = [
        #  "./raw/raw1",
        "./raw/raw2",
        #  "./raw/raw3",
        #  "./raw/raw4",
        #  "./raw/raw5",
    ]

    for raw_folder in raw_folder_list:
        raw_folder_full = abspath(join(dir_file, datafolder, raw_folder))
        print(f"raw_folder_full {raw_folder_full}")

        do_extraction(raw_folder_full, save_folder_full, pad)


if __name__ == "__main__":
    main()
