#! /usr/bin/env python2
# coding: utf-8

""" RCN on EMNIST-letters dataset. """
__author__ = "Rui Lin"
__email__ = "rxxlin@umich.edu"

import argparse
import numpy as np
import os
from multiprocessing import Pool
from functools import partial
from scipy.misc import imresize
from scipy.ndimage import imread

from science_rcn.inference import test_image
from science_rcn.learning import train_image

np.random.seed(42)

# -- logging
import logging
import logging.handlers
LOG_LEVEL = logging.INFO

def logging_config():
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    f = logging.Formatter("[%(levelname)s]%(module)s->%(funcName)s: \t %(message)s \t --- %(asctime)s")

    h = logging.StreamHandler()
    h.setFormatter(f)
    h.setLevel(LOG_LEVEL)

    root_logger.addHandler(h)

# -- argparse
def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument(
        '--num_per_class_training',
        type=int,
        default=None,
        help="Number of training examples.",
    )
    parser.add_argument(
        '--num_per_class_testing',
        type=int,
        default=None,
        help="Number of testing examples.",
    )
    args = parser.parse_args()
    return args


# -- RCN, use the RCN MNIST example implementation, see (https://github.com/vicariousinc/science_rcn)
# End-to-end implementation of RCN is beyond our scope for now, as explained in our report.
def run(data_dir,
        num_per_class_training=None,
        num_per_class_testing=None,
        pool_shape=(25, 25), # use default
        perturb_factor=2., # use default
        ):
    LOG = logging.getLogger()

    # Multiprocessing set up
    pool = Pool(4)

    train_data, test_data = get_mnist_data_iters(
        data_dir, num_per_class_training, num_per_class_testing)

    LOG.info("Training on {} images...".format(len(train_data)))
    train_partial = partial(train_image,
                            perturb_factor=perturb_factor)
    train_results = pool.map_async(train_partial, [d[0] for d in train_data]).get(9999999)
    all_model_factors = zip(*train_results)

    LOG.info("Testing on {} images...".format(len(test_data)))
    test_partial = partial(test_image, num_candidates=5, model_factors=all_model_factors,
                           pool_shape=pool_shape) # less number of candidates
    test_results = pool.map_async(test_partial, [d[0] for d in test_data]).get(9999999)

    # Evaluate result
    correct = 0
    for test_idx, (winner_idx, _) in enumerate(test_results):
        correct += test_data[test_idx][1] == train_data[winner_idx][1]
    print "Total test accuracy = {}".format(float(correct) / len(test_results))

    return all_model_factors, test_results


def get_mnist_data_iters(data_dir, num_per_class_training, num_per_class_testing):
    if not os.path.isdir(data_dir):
        raise IOError("Can't find your data dir '{}'".format(data_dir))

    def _load_data(image_dir, num_per_class, get_filenames=False):
        loaded_data = []
        for category in sorted(os.listdir(image_dir)):
            cat_path = os.path.join(image_dir, category)
            if not os.path.isdir(cat_path) or category.startswith('.'):
                continue
            samples = sorted(os.listdir(cat_path))[:num_per_class]

            for fname in samples:
                filepath = os.path.join(cat_path, fname)
                # Resize and pad the images to (200, 200)
                image_arr = imresize(imread(filepath, "L"), (112, 112))
                img = np.pad(image_arr,
                             pad_width=tuple([(p, p) for p in (44, 44)]),
                             mode='constant', constant_values=0)
                loaded_data.append((img, category))
        return loaded_data

    train_set = _load_data(os.path.join(data_dir, 'training'),
                           num_per_class=num_per_class_training)
    test_set = _load_data(os.path.join(data_dir, 'testing'),
                          num_per_class=num_per_class_testing)
    return train_set, test_set


if __name__ == '__main__':
    logging_config()
    logger = logging.getLogger()

    args = parse_cmd()

    run(args.data_dir,
            num_per_class_training=args.num_per_class_training,
            num_per_class_testing=args.num_per_class_testing)
