from config.cfg import Cfg
from anchor_gen import gen_anchor
from first_handler_anchor_gtbox import match_anchor_gtbox
from data_set.data_loader import q
cfg = Cfg()

anchors = gen_anchor.generate_anchors(scales=cfg.anchors_scals,ratios=cfg.anchors_radios,shape=cfg.feature_shape,feature_stride=cfg.feature_stride)


images,boxs,label,rpn_match,rpn_bbox =q.get()
print(rpn_bbox)
print(boxs)