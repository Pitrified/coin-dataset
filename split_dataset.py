import argparse
import logging

import numpy as np

from os import listdir
from os import makedirs
from os import sep
from os.path import abspath
from os.path import dirname
from os.path import splitext
from os.path import isdir
from os.path import join
from random import seed
from timeit import default_timer as timer
from shutil import copy2
from random import shuffle


def parse_arguments():
    """Setup CLI interface
    """
    parser = argparse.ArgumentParser(
        description="Split a labeled dataset in train/val/test"
    )

    parser.add_argument(
        "-pi",
        "--path_input",
        type=str,
        required=True,
        help="path to input dataset to use",
    )

    parser.add_argument(
        "-po", "--path_output", type=str, default=None, help="path to output folder"
    )

    parser.add_argument(
        "-vs", "--validation_size", type=int, default=0, help="size of validation set"
    )

    parser.add_argument(
        "-ts", "--test_size", type=int, required=True, help="size of test set"
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
    #  log_format_module = "%(levelname)s: %(message)s"
    log_format_module = "%(message)s"

    formatter = logging.Formatter(log_format_module)
    module_console_handler.setFormatter(formatter)

    logmoduleconsole.addHandler(module_console_handler)

    logging.addLevelName(5, "TRACE")
    # use it like this
    # logmoduleconsole.log(5, 'Exceedingly verbose debug')

    return logmoduleconsole


def do_split_dataset(
    path_input_full, path_output_full, validation_size, test_size, logLevel="WARN"
):
    """Split a dataset in train/val/test
    """

    logsplit = logging.getLogger(f"{__name__}.console.split")
    logsplit.setLevel(logLevel)

    for label in listdir(path_input_full):
        label_full = join(path_input_full, label)
        logsplit.info(f"\nLABEL: {label}")
        logsplit.debug(f"label_full             {label_full}")

        path_output_train_full = abspath(join(path_output_full, "train", label))
        logsplit.debug(f"path_output_train_full {path_output_train_full}")
        if not isdir(path_output_train_full):
            makedirs(path_output_train_full)

        path_output_val_full = abspath(join(path_output_full, "validation", label))
        logsplit.debug(f"path_output_val_full   {path_output_val_full}")
        if not isdir(path_output_val_full):
            makedirs(path_output_val_full)

        path_output_test_full = abspath(join(path_output_full, "test", label))
        logsplit.debug(f"path_output_test_full  {path_output_test_full}")
        if not isdir(path_output_test_full):
            makedirs(path_output_test_full)

        #  image_list = listdir(label_full)[:5]
        image_list = listdir(label_full)
        shuffle(image_list)

        # info on test size
        test_size_available = len(image_list)
        if test_size > test_size_available:
            logsplit.warn(
                f"{label}: Not enough images available for the test set ({test_size_available}/{test_size})"
            )
            test_size_actual = test_size_available
        else:
            test_size_actual = test_size

        # info on validation size
        val_size_available = len(image_list) - test_size
        if val_size_available < 0:
            val_size_available = 0
        if val_size_available < validation_size:
            logsplit.warn(
                f"{label}: Not enough images available for the validation set ({val_size_available}/{validation_size})"
            )
            val_size_actual = val_size_available
        else:
            val_size_actual = validation_size

        # info on train size
        train_size_available = len(image_list) - validation_size - test_size
        if train_size_available <= 0:
            train_size_available = 0
            logsplit.warn(f"{label}: Not enough images available for the training set")
            train_size_actual = 0
        else:
            train_size_actual = train_size_available

        logsplit.info(f"Splitting {len(image_list)} images for {label} in:")
        logsplit.info(
            f"\ttrain ({train_size_actual}), validation ({val_size_actual}), test ({test_size_actual})"
        )

        for i, image_name in enumerate(image_list):
            image_name_full = join(label_full, image_name)
            logsplit.log(5, f"IMAGE: {image_name}")

            if i < test_size_actual:
                copy2(image_name_full, path_output_test_full)
            elif i < test_size_actual + val_size_actual:
                copy2(image_name_full, path_output_val_full)
            else:
                copy2(image_name_full, path_output_train_full)


def main():
    logmoduleconsole = setup_logger("WARN")

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
    if args.path_output is None:
        # if path_input is "./modified/" we need "./modified"
        path_input_clean = path_input[:-1] if path_input[-1] == sep else path_input
        logmoduleconsole.debug(f"path_input_clean {path_input_clean}")
        path_output = f"{path_input_clean}_split"
    else:
        path_output = args.path_output

    validation_size = args.validation_size
    test_size = args.test_size

    recap = f"python3 split_dataset.py"
    recap += f" --path_input {path_input}"
    recap += f" --path_output {path_output}"
    recap += f" --validation_size {validation_size}"
    recap += f" --test_size {test_size}"
    recap += f" --seed {myseed}"

    logmoduleconsole.info(recap)

    dir_file = abspath(dirname(__file__))

    path_input_full = abspath(join(dir_file, path_input))
    logmoduleconsole.info(f"path_input_full {path_input_full}")

    path_output_full = abspath(join(dir_file, path_output))
    logmoduleconsole.info(f"path_output_full {path_output_full}")

    #  logLevel = "WARN"
    logLevel = "INFO"
    #  logLevel = "DEBUG"
    #  logLevel = "TRACE"

    do_split_dataset(
        path_input_full, path_output_full, validation_size, test_size, logLevel
    )


if __name__ == "__main__":
    main()
