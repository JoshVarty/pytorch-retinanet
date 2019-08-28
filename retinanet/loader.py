# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Detectron data loader. The design is generic and abstracted away from any
details of the minibatch. A minibatch is a dictionary of blob name keys and
their associated numpy (float32 or int32) ndarray values.

Outline of the data loader design:

loader thread\
loader thread \                    / GPU 1 enqueue thread -> feed -> EnqueueOp
...           -> minibatch queue ->  ...
loader thread /                    \ GPU N enqueue thread -> feed -> EnqueueOp
loader thread/

<---------------------------- CPU -----------------------------|---- GPU ---->

A pool of loader threads construct minibatches that are put onto the shared
minibatch queue. Each GPU has an enqueue thread that pulls a minibatch off the
minibatch queue, feeds the minibatch blobs into the workspace, and then runs
an EnqueueBlobsOp to place the minibatch blobs into the GPU's blobs queue.
During each fprop the first thing the network does is run a DequeueBlobsOp
in order to populate the workspace with the blobs from a queued minibatch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import deque
from collections import OrderedDict
import logging
import numpy as np
import signal
import threading
import time
import uuid
from six.moves import queue as Queue

#from caffe2.python import core, workspace

from retinanet.config import cfg
from retinanet.minibatch import get_minibatch
from retinanet.minibatch import get_minibatch_blob_names
# from detectron.utils.coordinator import coordinated_get
# from detectron.utils.coordinator import coordinated_put
# from detectron.utils.coordinator import Coordinator
# import detectron.utils.c2 as c2_utils

logger = logging.getLogger(__name__)


class RoIDataLoader(object):
    def __init__(
        self,
        roidb,
        num_loaders=4,
        minibatch_queue_size=64,
        blobs_queue_capacity=8
    ):
        self._roidb = roidb
        self._lock = threading.Lock()
        self._perm = deque(range(len(self._roidb)))
        self._cur = 0  # _perm cursor
        # The minibatch queue holds prepared training data in host (CPU) memory
        # When training with N > 1 GPUs, each element in the minibatch queue
        # is actually a partial minibatch which contributes 1 / N of the
        # examples to the overall minibatch
        self._minibatch_queue = Queue.Queue(maxsize=minibatch_queue_size)
        self._blobs_queue_capacity = blobs_queue_capacity
        # Random queue name in case one instantiates multple RoIDataLoaders
        self._loader_id = uuid.uuid4()
        self._blobs_queue_name = 'roi_blobs_queue_{}'.format(self._loader_id)
        # Loader threads construct (partial) minibatches and put them on the
        # minibatch queue
        self._num_loaders = 1
        self._num_gpus = cfg.NUM_GPUS

        self._output_names = get_minibatch_blob_names()
        self._shuffle_roidb_inds()
        #self.create_threads()

    def get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch. Thread safe."""
        valid = False
        while not valid:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            blobs, valid = get_minibatch(minibatch_db)
        return blobs

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb. Not thread safe."""
        if cfg.TRAIN.ASPECT_GROUPING:
            widths = np.array([r['width'] for r in self._roidb])
            heights = np.array([r['height'] for r in self._roidb])
            horz = (widths >= heights)  #
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]

            horz_inds = np.random.permutation(horz_inds)
            vert_inds = np.random.permutation(vert_inds)
            mb = cfg.TRAIN.IMS_PER_BATCH
            horz_inds = horz_inds[:(len(horz_inds) // mb) * mb]
            vert_inds = vert_inds[:(len(vert_inds) // mb) * mb]
            inds = np.hstack((horz_inds, vert_inds))

            inds = np.reshape(inds, (-1, mb))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1, ))
            self._perm = inds
        else:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._perm = deque(self._perm)
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch. Thread safe."""
        with self._lock:
            # We use a deque and always take the *first* IMS_PER_BATCH items
            # followed by *rotating* the deque so that we see fresh items
            # each time. If the length of _perm is not divisible by
            # IMS_PER_BATCH, then we end up wrapping around the permutation.
            db_inds = [self._perm[i] for i in range(cfg.TRAIN.IMS_PER_BATCH)]
            self._perm.rotate(-cfg.TRAIN.IMS_PER_BATCH)
            self._cur += cfg.TRAIN.IMS_PER_BATCH
            if self._cur >= len(self._perm):
                self._shuffle_roidb_inds()
        return db_inds

    def get_output_names(self):
        return self._output_names
