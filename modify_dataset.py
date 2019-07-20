import argparse
import logging
import cv2

import numpy as np

from math import floor, ceil
from math import sqrt
from os import listdir
from os import makedirs
from os import sep
from os.path import abspath
from os.path import dirname
from os.path import isdir
from os.path import join
from os.path import splitext
from random import seed
from timeit import default_timer as timer


def parse_arguments():
    """Setup CLI interface
    """
    parser = argparse.ArgumentParser(description="A tool to elaborate the coin-dataset")

    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        default="./original",
        help="path to input images to use",
    )

    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="./modified",
        help="path to output folder",
    )

    parser.add_argument("-s", "--seed", type=int, default=-1, help="random seed to use")

    # last line to parse the args
    args = parser.parse_args()
    return args


def setup_logger(logLevel="DEBUG"):
    """Setup logger that outputs to console for the module
    """
    logmoduleconsole = logging.getLogger(f"{__name__}.console")
    logmoduleconsole.propagate = False
    logmoduleconsole.setLevel(logLevel)

    module_console_handler = logging.StreamHandler()

    #  log_format_module = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #  log_format_module = "%(name)s - %(levelname)s: %(message)s"
    #  log_format_module = '%(levelname)s: %(message)s'
    log_format_module = "%(message)s"

    formatter = logging.Formatter(log_format_module)
    module_console_handler.setFormatter(formatter)

    logmoduleconsole.addHandler(module_console_handler)

    logging.addLevelName(5, "TRACE")
    # use it like this
    # logmoduleconsole.log(5, 'Exceedingly verbose debug')

    return logmoduleconsole


def imshow_resized(winName, img, maxSize=800, wait_for_key=False):
    """Show the image, resized if bigger than maxSize
    """
    maxImgSize = max(img.shape[0:2])
    if maxImgSize > maxSize:
        scale = maxSize / maxImgSize
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow(winName, img)
    if wait_for_key:
        cv2.waitKey(0)


def equalize_Lab_clahe(img, clipLimit=2.0, tileGridSize=(8, 8), show=False):
    """Equalize the L channel using Contrast Limited Adaptive Histogram Equalization

    @param clipLimit Threshold for contrast limiting.

    @param tileGridSize Size of grid for histogram equalization. Input image
    will be divided into equally sized rectangular tiles. tileGridSize defines
    the number of tiles in row and column.
    """
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    L, a, b = cv2.split(img)
    L_clahed = clahe.apply(L)
    img_clahed = cv2.merge((L_clahed, a, b))
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


def mask_background(img_orig, x, y, r, pad=10, show=False):
    """Mask the background of the circle (x,y,r)
    """

    # mask the circle in the original image
    circle_mask = np.zeros((img_orig.shape[0], img_orig.shape[1]), dtype=np.uint8)
    cv2.circle(circle_mask, center=(x, y), radius=r + pad, color=(255), thickness=-1)
    if show:
        cv2.imshow("Mask", circle_mask)

    # apply the mask
    img_masked = cv2.bitwise_and(img_orig, img_orig, mask=circle_mask)
    if show:
        cv2.imshow("Masked", img_masked)
    if show:
        cv2.waitKey(0)

    return img_masked


def do_modify(args):
    """Modify the dataset

    Options:
    * equalize
    * mask
    * resize

    In this order
    # img_work = cv2.imread
    # img_work = equalize_Lab_clahe(img_work)
    # img_work = mask_background(img_work)
    # img_work = cv2.resize(img_work)
    """

    logmodify = logging.getLogger(f"{__name__}.console.modify")

    dir_file = abspath(dirname(__file__))

    input_path = args.input_path
    input_path_full = abspath(join(dir_file, input_path))
    logmodify.info(f"input_path_full {input_path_full}")

    output_path = args.output_path
    output_path_full = abspath(join(dir_file, output_path))
    logmodify.info(f"output_path_full {output_path_full}")

    equalize_flag = True
    mask_flag = True
    resize_flag = True

    show_original = True
    show_clahed = True

    for label in listdir(input_path_full):
        label_full = join(input_path_full, label)
        logmodify.info(f"\nLABEL: {label}\n")

        for image_name in sorted(listdir(label_full)[:5]):
            image_name_full = join(label_full, image_name)
            logmodify.debug(f"IMAGE: {image_name}")

            img_work = cv2.imread(image_name_full)
            if show_original:
                imshow_resized("Original", img_work)

            if equalize_flag:
                img_work = equalize_Lab_clahe(img_work)
                if show_clahed:
                    imshow_resized("Lab clahed", img_work)

            if show_original or show_clahed:
                cv2.waitKey(0)


def main():
    args = parse_arguments()

    # setup seed value
    if args.seed == -1:
        myseed = 1
        myseed = int(timer() * 1e9 % 2 ** 32)
    else:
        myseed = args.seed
    seed(myseed)
    np.random.seed(myseed)

    path_input = args.input_path

    logmoduleconsole = setup_logger()

    logmoduleconsole.info(f"python3 modify_dataset.py -s {myseed} -i {path_input}")

    do_modify(args)


if __name__ == "__main__":
    main()
