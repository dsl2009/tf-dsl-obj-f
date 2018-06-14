import numpy as np
from anchor_gen import gen_anchor
import utils
class Cfg(object):

    def __init__(self,is_train):
        self.anchors_scals = [128, 256, 512]
        self.anchors_radios = [0.5, 1, 2]
        self.feature_stride = 16
        self.image_size = [512, 512]
        self.batch_size = 4
        self.RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
        self.RPN_TRAIN_ANCHORS_PER_IMAGE = 256
        self.BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
        self.RPN_NMS_THRESHOLD = 0.7
        self.feature_shape = [np.ceil(x / self.feature_stride) for x in self.image_size]
        self.total_anchors = self.feature_shape[0]*self.feature_shape[1]*9
        self.anchors = gen_anchor.generate_anchors(scales=self.anchors_scals, ratios=self.anchors_radios,
                                              shape=self.feature_shape, feature_stride=self.feature_stride)
        self.norm_anchors = utils.norm_boxes(self.anchors,self.image_size)
        if is_train:
            self.NMS_ROIS_TRAINING = 2000
        else:
            self.NMS_ROIS_INFERENCE = 1000

config_instace = Cfg(is_train=True)
