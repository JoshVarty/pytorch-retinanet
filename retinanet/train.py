import os
import logging

from retinanet.config import cfg
from retinanet.config import get_output_dir
from retinanet.model_builder import create
from retinanet.roidb import combined_roidb_for_training


def train_model():
    """Model training loop."""
    model, weights = create_model()

    roidb = combined_roidb_for_training(
        cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES
    )

    return 0



def create_model():
    """Build the model and look for saved checkpoints in case we can resume from one"""
    logger = logging.getLogger(__name__)
    output_dir = get_output_dir(cfg.TRAIN.DATASETS, training=True)
    weights_file = cfg.TRAIN.WEIGHTS

    if cfg.TRAIN.AUTO_RESUME:
        # TODO: Load weights
        pass

    logger.info('Building model: {}'.format(cfg.MODEL.TYPE))
    model = create()


    return model, weights_file

        

