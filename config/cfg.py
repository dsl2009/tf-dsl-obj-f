import numpy as np
from anchor_gen import gen_anchor
import utils
class Cfg(object):

    def __init__(self,is_train):
        self.is_train = is_train
        #self.anchors_scals = [128, 256, 512]
        self.anchors_scals = [(16, 32, 64), (96, 156, 244), (294, 349, 420)]
        self.anchors_radios = [0.5, 1, 2]
        self.feature_stride = [8,16,32]
        self.image_size = [512, 512]
        self.num_class = 21
        self.batch_size = 8
        self.RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
        self.RPN_TRAIN_ANCHORS_PER_IMAGE = 256
        self.BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
        self.RPN_NMS_THRESHOLD = 0.7
        self.feature_shape = [(np.ceil(self.image_size[0]/x ),np.ceil(self.image_size[0]/x)) for x in self.feature_stride]
        self.total_anchors = sum(f_shape[0]*f_shape[1] for f_shape in self.feature_shape )*9
        self.anchors = gen_anchor.gen_multi_anchors(scales=self.anchors_scals, ratios=self.anchors_radios,
                                              shape=self.feature_shape, feature_stride=self.feature_stride)

        self.norm_anchors = utils.norm_boxes(self.anchors,self.image_size)
        self.VOC_CLASSES = ('back',
                       'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair',
                       'cow', 'diningtable', 'dog', 'horse',
                       'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor')

        self.TRAIN_ROIS_PER_IMAGE = 200
        self.DETECTION_MIN_CONFIDENCE = 0.6
        self.DETECTION_MAX_INSTANCES = 100
        self.DETECTION_NMS_THRESHOLD = 0.3
        self.pool_shape = 7
        self.ROI_POSITIVE_RATIO = 0.33
        if is_train:
            self.NMS_ROIS_TRAINING = 2000
        else:
            self.NMS_ROIS_TRAINING = 1000
            self.batch_size = 1

config_instace = Cfg(is_train=False)
