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
        "--path_input",
        type=str,
        default="./original",
        help="path to input images to use",
    )

    parser.add_argument(
        "-o",
        "--path_output",
        type=str,
        default="./modified",
        help="path to output folder",
    )

    parser.add_argument(
        "-os", "--out_size", type=int, default=64, help="size of the output images"
    )

    parser.add_argument(
        "-ne",
        "--no_equalize",
        default=False,
        action="store_true",
        help="set this option to *not* equalize the image",
    )
    parser.add_argument(
        "-nm",
        "--no_mask",
        default=False,
        action="store_true",
        help="set this option to *not* mask the image",
    )
    parser.add_argument(
        "-nr",
        "--no_resize",
        default=False,
        action="store_true",
        help="set this option to *not* resize the image",
    )

    parser.add_argument(
        "-pc",
        "--pad_crop",
        type=int,
        default=30,
        help="pad used in the creation of the dataset",
    )
    parser.add_argument(
        "-pm",
        "--pad_mask",
        type=int,
        default=5,
        help="pad to use when masking the background",
    )
    parser.add_argument(
        "-pr",
        "--pad_resize",
        type=int,
        default=5,
        help="pad to use when cropping and resizing",
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


def find_best_circle(image, pad_crop):
    """Find the best position of a coin in the image
    """
    logfindbest = logging.getLogger(f"{__name__}.console.findbest")
    logfindbest.setLevel("INFO")

    img_dim = image.shape[0]
    center = img_dim // 2
    logfindbest.debug(f"\tImage dim {img_dim} center {center}")

    # position of the circle used to create this image
    ox, oy = center, center
    orad = center - pad_crop
    logfindbest.debug(f"\tOrig circle {ox} {oy} - {orad}")

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    method = cv2.HOUGH_GRADIENT
    dp = 1.0
    minDist = img_dim
    param1 = 70
    param2 = 35
    minRadius = center - 2 * pad_crop
    maxRadius = center

    circles = cv2.HoughCircles(
        image_gray,
        method=method,
        dp=dp,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius,
    )

    if circles is None:
        return ox, oy, orad

    # keep only the first (best) circle
    bx, by, br = circles[0, 0, :].astype(int)
    logfindbest.debug(f"\tBest circle {bx} {by} - {br}")

    # average the original and best circle
    mx = (ox + bx) // 2
    my = (oy + by) // 2
    mr = (orad + br) // 2
    logfindbest.debug(f"\tMean circle {mx} {my} - {mr}")

    if False:
        # this changes the original image, enable only to tune the algorithm
        cv2.circle(image, (ox, oy), orad, thickness=1, color=(255, 0, 0))
        cv2.circle(image, (bx, by), br, thickness=1, color=(0, 255, 0))
        cv2.circle(image, (mx, my), mr, thickness=1, color=(0, 0, 255))

    return mx, my, mr


def do_modify(
    path_input_full,
    path_output_full,
    out_size,
    flag_equalize=True,
    flag_mask=True,
    flag_resize=True,
    pad_crop=30,
    pad_mask=5,
    pad_resize=5,
    interpolation_method=cv2.INTER_AREA,
    logLevel="WARN",
):
    """Modify the dataset

    Options:
    * equalize
    * mask
    * resize (fit circle to out_size)

    Both mask and resize have an optional (independent) padding
    The path_output might be a numpy array of images TODO

    In this order
    # img_work = cv2.imread
    # find_best_circle on original image? or on equalized? TODO
    # img_work = equalize_Lab_clahe(img_work)
    # img_work = mask_background(img_work)
    # img_work = cv2.resize(img_work)
    """

    logmodify = logging.getLogger(f"{__name__}.console.modify")
    logmodify.setLevel(logLevel)

    show_all = False
    show_original = False or show_all
    show_clahed = False or show_all
    show_masked = False or show_all
    show_resized = False or show_all
    save_results = True

    for label in listdir(path_input_full):
        label_full = join(path_input_full, label)
        logmodify.info(f"\nLABEL: {label}")

        path_output_label_full = abspath(join(path_output_full, label))
        logmodify.debug(f"path_output_label_full {path_output_label_full}\n")
        if save_results and not isdir(path_output_label_full):
            makedirs(path_output_label_full)

        #  for image_name in sorted(listdir(label_full)):
        for image_name in sorted(listdir(label_full)[:5]):
            image_name_full = join(label_full, image_name)
            logmodify.info(f"IMAGE: {image_name}")

            img_work = cv2.imread(image_name_full)
            if show_original:
                imshow_resized("Original", img_work)

            if flag_equalize:
                img_work = equalize_Lab_clahe(img_work)
                if show_clahed:
                    imshow_resized("Lab clahed", img_work)

            # find the best circle of the coin
            x, y, r = find_best_circle(img_work, pad_crop)

            # mask around it, with padding pad_mask
            if flag_mask:
                img_work = mask_background(img_work, x, y, r, pad_mask)
                if show_masked:
                    imshow_resized("Lab masked", img_work)

            # crop and resize around it, with padding pad_resize
            if flag_resize:
                img_work = crop_circle(img_work, x, y, r, pad_resize)
                img_work = cv2.resize(
                    img_work, (out_size, out_size), interpolation=interpolation_method
                )
                if show_resized:
                    imshow_resized("Lab resized", img_work)

            if show_original or show_clahed or show_masked or show_resized:
                cv2.waitKey(0)

            if save_results:
                out_name_full = join(path_output_label_full, image_name)
                logmodify.debug(f"\tout_name_full {out_name_full}")
                cv2.imwrite(out_name_full, img_work)


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

    path_input = args.path_input
    path_output = args.path_output

    out_size = args.out_size

    flag_equalize = not args.no_equalize
    flag_mask = not args.no_mask
    flag_resize = not args.no_resize

    pad_crop = args.pad_crop
    pad_mask = args.pad_mask
    pad_resize = args.pad_resize

    interpolation_method = cv2.INTER_AREA

    logmoduleconsole = setup_logger()

    recap = f"python3 modify_dataset.py"
    recap += f" --path_input {path_input}"
    recap += f" --path_output {path_output}"
    recap += f" --out_size {out_size}"
    recap += " --no_equalize" if args.no_equalize else ""
    recap += " --no_mask" if args.no_mask else ""
    recap += " --no_resize" if args.no_resize else ""
    recap += f" --pad_crop {pad_crop}"
    recap += f" --pad_mask {pad_mask}"
    recap += f" --pad_resize {pad_resize}"
    recap += f" --seed {myseed}"

    logmoduleconsole.info(recap)

    logLevel = "INFO"
    #  logLevel = "ERROR"

    dir_file = abspath(dirname(__file__))

    path_input_full = abspath(join(dir_file, path_input))
    logmoduleconsole.info(f"path_input_full {path_input_full}")

    path_output_full = abspath(join(dir_file, path_output))
    logmoduleconsole.info(f"path_output_full {path_output_full}")

    do_modify(
        path_input_full,
        path_output_full,
        out_size,
        flag_equalize,
        flag_mask,
        flag_resize,
        pad_crop,
        pad_mask,
        pad_resize,
        interpolation_method,
        logLevel,
    )


if __name__ == "__main__":
    main()
