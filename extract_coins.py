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
from math import sqrt


def center_inside(gx, gy, gr, x, y):
    """True if (x,y) is inside the circle (gx,gy,gr)
    """
    return sqrt((gx - x) ** 2 + (gy - y) ** 2) < gr


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


def is_inside(x, y, r, circles):
    """True if (x,y) is inside any of circles
    """
    for gx, gy, gr in circles:
        if center_inside(gx, gy, gr, x, y):
            return True
    return False


def find_coins_circles_refined(image_name_full, hcParam, num_coins, show_clahed=False):
    """Find circles from this image, using the param passed

    Return a list of circles

    Use CLAHE to equalize the image before applying HoughCircles

    Refine the results, no overlapping circles returned, analyze only the first
    num_coins*ratio circles found. This constraint should not matter anyway, as
    soon as num_coins are found, stop iterating.
    """

    img_orig = cv2.imread(image_name_full, cv2.IMREAD_GRAYSCALE)

    img_orig = cv2.GaussianBlur(img_orig, (5, 5), 5)

    #  cv2.imshow("Blurred", img_orig)
    #  cv2.waitKey(0)

    img_clahed = equalize_gray_clahe(img_orig)

    #  print(f"Searching circles with param {hcParam}")
    circles = cv2.HoughCircles(image=img_clahed, **hcParam)

    max_circles = num_coins * 2

    good_circles = []

    if show_clahed:
        img_orig = cv2.imread(image_name_full)

    if not circles is None:
        print(f"\tCircles before refining {circles.shape[1]}")
        for x, y, r in circles[0, :max_circles, :]:

            if not is_inside(x, y, r, good_circles):
                good_circles.append((x, y, r))

                if show_clahed:
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
            else:
                if show_clahed:
                    cv2.circle(img_orig, (x, y), r, thickness=3, color=(255, 0, 0))

            if show_clahed:
                scale = 1000 / max(img_orig.shape[0:2])
                img_orig_small = cv2.resize(
                    img_orig, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                )

                cv2.imshow("Circled", img_orig_small)
                cv2.waitKey(10)

            if len(good_circles) == num_coins:
                break

    if show_clahed:
        cv2.waitKey(0)

    return good_circles


def do_extraction(raw_folder_full, save_folder_full, pad):
    """Extract all circles from images, pad them

    Input images: all folders in raw_folder_full
    Output images: same folders, in save_folder_full
    """
    show_crop = False
    show_clahed = False
    #  show_clahed = True
    save_results = False
    save_results = True

    raw_folder_tag = raw_folder_full.split(sep)[-1]
    print(f"\nRAW folder tag: {raw_folder_tag}\n")

    for label_folder in listdir(raw_folder_full):
        if not label_folder.startswith("1c"):
            continue

        label, num_coins = label_folder.split("_")
        num_coins = int(num_coins)

        print(f"\nLABEL: {label}\tNUM_COINS: {num_coins}\n")
        full_label = join(raw_folder_full, label_folder)

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
            circles = find_coins_circles_refined(
                image_name_full,
                getHCpar_wide(label, raw_folder_tag),
                num_coins,
                show_clahed=show_clahed,
            )
            print(f"\tCircles found {len(circles)}")

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


def getHCpar_wide(label, raw_folder_tag="standard"):
    """Return a dict of params for HoughCircles

    If needed, multiple param can be defined for different datasets
    """
    if label in ["1c"]:
        # 1c minDist 150 minRadius 125 maxRadius 160 hcParam1 65 hcParam2 30
        param = {
            "method": cv2.HOUGH_GRADIENT,
            "dp": 1.0,
            "minDist": 150,
            "minRadius": 100,
            "maxRadius": 140,
            "param1": 65,
            "param2": 30,
        }
    elif label in ["2c"]:
        # 2c
        param = {
            "method": cv2.HOUGH_GRADIENT,
            "dp": 1.0,
            "minDist": 250,
            "minRadius": 125,
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
            "minRadius": 140,
            "maxRadius": 180,
            "param1": 60,
            "param2": 25,
        }
    elif label in ["10c"]:
        # 10c minDist 150 minRadius 140 maxRadius 160 hcParam1 100 hcParam2 41
        param = {
            "method": cv2.HOUGH_GRADIENT,
            "dp": 1.0,
            "minDist": 150,
            "minRadius": 120,
            "maxRadius": 160,
            "param1": 70,
            "param2": 30,
        }
    elif label in ["20c"]:
        # 20c
        param = {
            "method": cv2.HOUGH_GRADIENT,
            "dp": 1.0,
            "minDist": 250,
            "minRadius": 145,
            "maxRadius": 190,
            "param1": 65,
            "param2": 30,
        }
    elif label in ["50c"]:
        # 50c minDist 150 minRadius 165 maxRadius 200 hcParam1 80 hcParam2 45
        param = {
            "method": cv2.HOUGH_GRADIENT,
            "dp": 1.0,
            "minDist": 300,
            "minRadius": 150,
            "maxRadius": 220,
            "param1": 70,
            "param2": 45,
        }
    elif label in ["1e"]:
        # 1e minDist 40 minRadius 165 maxRadius 190 hcParam1 70 hcParam2 25
        param = {
            "method": cv2.HOUGH_GRADIENT,
            "dp": 1.0,
            "minDist": 400,
            "minRadius": 145,
            "maxRadius": 200,
            "param1": 70,
            "param2": 25,
        }
    elif label in ["2e"]:
        if raw_folder_tag == "raw1":
            param = {
                "method": cv2.HOUGH_GRADIENT,
                "dp": 1.0,
                "minDist": 360,
                "minRadius": 180,
                "maxRadius": 300,
                "param1": 75,
                "param2": 40,
            }
        else:
            param = {
                "method": cv2.HOUGH_GRADIENT,
                "dp": 1.0,
                "minDist": 100,
                "minRadius": 150,
                "maxRadius": 230,
                "param1": 75,
                "param2": 40,
            }

    return param


def main():
    dir_file = abspath(dirname(__file__))

    # common root to the dataset folder
    datafolder = "."

    save_folder = "./parsed"
    save_folder_full = abspath(join(dir_file, datafolder, save_folder))
    print(f"save_folder_full {save_folder_full}")

    # pad around coin when cropping the piece
    pad = 30

    raw_folder_list = [
        #  "./raw/raw1",
        #  "./raw/raw2",
        #  "./raw/raw3",
        #  "./raw/raw4",
        #  "./raw/raw5",
        "./raw/raw6",
    ]

    for raw_folder in raw_folder_list:
        raw_folder_full = abspath(join(dir_file, datafolder, raw_folder))
        print(f"raw_folder_full {raw_folder_full}")

        do_extraction(raw_folder_full, save_folder_full, pad)


if __name__ == "__main__":
    main()
