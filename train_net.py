
import sys
import pprint
import logging
import argparse
import numpy as np

from retinanet.logging import setup_logging
from retinanet.train import train_model

from retinanet.config import cfg
from retinanet.config import merge_cfg_from_file
from retinanet.config import assert_and_infer_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a network with Detectron'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training (and optionally testing)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--multi-gpu-testing',
        dest='multi_gpu_testing',
        help='Use cfg.NUM_GPUS GPUs for inference',
        action='store_true'
    )
    parser.add_argument(
        '--skip-test',
        dest='skip_test',
        help='Do not test the final model',
        action='store_true'
    )
    parser.add_argument(
        'opts',
        help='See detectron/core/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    logger = setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)
    merge_cfg_from_file(args.cfg_file)
    assert_and_infer_cfg()
    logger.info('Training with config:')
    logger.info(pprint.pformat(cfg))

    # Note that while we set the numpy random seed network training will not be
    # deterministic in general. There are sources of non-determinism that cannot
    # be removed with a reasonble execution-speed tradeoff (such as certain
    # non-deterministic cudnn functions).
    np.random.seed(cfg.RNG_SEED)

    # Execute the training run
    checkpoint = train_model()
    # Test the trained model
    if not args.skip_test:
         test_model(checkpoints['final'], args.multi_gpu_testing, args.opts)


import retinanet

if __name__ == "__main__":
    main()