import os
import logging
import numpy as np
import cv2

from retinanet import lr_policy
from retinanet.config import cfg
from retinanet.config import get_output_dir
from retinanet.model_builder import create
from retinanet.roidb import combined_roidb_for_training
from retinanet.loader import RoIDataLoader

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

# for i in sorted(blobs.keys()):
#   try:
#     print(i, blobs[i].shape)
#   except:
#     print(i)

class CocoDataset(Dataset):

    def __init__(self, roidb):
        self._roidb = roidb
    
    def __len__(self):
        return len(self._roidb)

    def __getitem__(self, idx):
        num_images = len(self._roidb)
        scale_inds = np.random.randint(0, high=len(cfg.TRAIN.SCALES), size = num_images)

        im = cv2.imread(self._roidb[idx]['image'])

        assert im is not None, \
            'Failed to read image \'{}\''.format(self._roidb[idx]['image'])

        if self._roidb[idx]['flipped']:
            im=im[:,::-1,:]

        target_size = cfg.TRAIN.SCALES[scale_inds[idx]]

        im, im_scale = self.prep_im(im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
    
    def prep_im(self, im, pixel_means, target_size, max_size):
        """Prepare an image for use as a network input blob. Specially:
        - Subtract per-channel pixel mean
        - Convert to float32
        - Rescale to each of the specified target size (capped at max_size)
        Returns a list of transformed images, one for each target size. Also returns
        the scale factors that were used to compute each returned image.
        """
        im = im.astype(np.float32, copy=False)
        im -= pixel_means
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than max_size
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        im = cv2.resize(
            im,
            None,
            None,
            fx=im_scale,
            fy=im_scale,
            interpolation=cv2.INTER_LINEAR
        )
        return im, im_scale





def train_model():
    """Model training loop."""
    model, weights = create_model()

    roidb = combined_roidb_for_training(
        cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES
    )

    #TODO: Implement from roidb
    coco_dataset = None

    #TODO: Implement from coco_dataset/roidb
    coco_dataloader = RoIDataLoader(roidb, num_loaders=1)

    model.training = True

    #TODO: Use same optimizer as they did
    optimizer = optim.Adam(model.parameters(), lr=0.00125)

    cuda = torch.device('cuda')
    model = model.to(cuda)

    for cur_iter in range(cfg.SOLVER.MAX_ITER):
        optimizer.zero_grad()
        
        blobs = coco_dataloader.get_next_minibatch()

        data = torch.Tensor(blobs['data'])
        data = data.to(cuda)

        # The number of foreground predictions (TODO: Ground Truth or Predicted?)
        fg_num = float(blobs['retnet_fg_num'])
        # Get labels for classification
        classification_labels = []
        for i in range(3,8):
            key = 'retnet_cls_labels_fpn' + str(i)
            x = torch.from_numpy(blobs[key]).cuda()
            classification_labels.append(x)

        # Get labels for regression
        regression_targets = []
        locations = []
        for i in range(3,8):
            key = 'retnet_roi_bbox_targets_fpn' + str(i)
            x = torch.from_numpy(blobs[key]).cuda()
            regression_targets.append(x)
            key = 'retnet_roi_fg_bbox_locs_fpn' + str(i)
            x = torch.from_numpy(blobs[key]).cuda()
            locations.append(x)

        input = (data, classification_labels, regression_targets, locations, fg_num)
        loss = model(input)
        
        lr = lr_policy.get_lr_at_iter(cur_iter)
        optimizer.lr = lr

        optimizer.step()

        print(loss)





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

        

