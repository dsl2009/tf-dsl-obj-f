from data_set.data_voc import VOCDetection
from matplotlib import pyplot as plt
from data_set import face
from data_set import aug_utils
import random
import numpy as np
import visual
from data_set.augmentations import SSDAugmentation

from first_handler_anchor_gtbox import match_anchor_gtbox
from config.cfg import config_instace as cfg

data_set = VOCDetection('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/VOCdevkit/VOCdevkit')
#data_set = face.Face(root='/home/dsl/PycharmProjects/tf-ssd/data_set/face1.json',image_size=512)




def get_batch(batch_size,is_shuff = True,max_detect = 50,image_size=300):
    length = data_set.len()
    idx = list(range(length))
    b = 0
    index = 0
    while True:
        if True:
            if is_shuff and index==0:
                random.shuffle(idx)
                print(idx)

            img, box, lab = data_set.pull_item(idx[index])
            #img = img - 0.5
            #img = img * 2.0
            if random.randint(0,1)==1:
                img, box = aug_utils.fliplr_left_right(img,box)
            img = (img.astype(np.float32) - np.array([123.7, 116.8, 103.9]))/255

            rpn_match,rpn_box = match_anchor_gtbox.build_rpn_targets(
                anchors=cfg.anchors,
                gt_class_ids=lab,
                gt_boxes=box,
                config=cfg
            )
            print(rpn_match)
            if b== 0:

                images = np.zeros(shape=[batch_size,image_size,image_size,3],dtype=np.float32)
                boxs = np.zeros(shape=[batch_size,max_detect,4],dtype=np.float32)
                label = np.zeros(shape=[batch_size,max_detect],dtype=np.int32)
                images[b,:,:,:] = img
                boxs[b,:box.shape[0],:] = box
                label[b,:box.shape[0]] = lab
                index=index+1

                b=b+1

            else:
                images[b, :, :, :] = img
                boxs[b, :box.shape[0], :] = box
                label[b, :box.shape[0]] = lab
                index = index + 1
                b = b + 1


            if b>=batch_size:
                yield [images,boxs,label]
                b = 0
            if index>= length:
                index = 0

def get_batch_inception(batch_size,is_shuff = True,max_detect = 50,image_size=300):
    length = data_set.len()
    idx = list(range(length))
    b = 0
    index = 0
    aug = SSDAugmentation(image_size)
    while True:
        if True:
            if is_shuff and index==0:
                random.shuffle(idx)
                print(idx)

            img, box, lab = data_set.pull_item(idx[index])
            if True:

                if random.randint(0,1)==1:
                   img, box = aug_utils.fliplr_left_right(img,box)
                img = img/255.0
                img = img - 0.5
                img = img * 2.0
            else:

                img, box, lab = aug(img,box,lab)
                img = ((img + [104, 117, 123])/255-0.5)*2.0

            rpn_match, rpn_box = match_anchor_gtbox.build_rpn_targets(
                anchors=cfg.anchors,
                gt_class_ids=lab,
                gt_boxes=box,
                config=cfg
            )

            #img = (img.astype(np.float32) - np.array([123.7, 116.8, 103.9]))/255
            if b== 0:

                images = np.zeros(shape=[batch_size,image_size,image_size,3],dtype=np.float32)
                boxs = np.zeros(shape=[batch_size,max_detect,4],dtype=np.float32)
                label = np.zeros(shape=[batch_size,max_detect],dtype=np.int32)
                batch_rpn_match = np.zeros(
                    [batch_size, cfg.anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    [batch_size, cfg.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_box.dtype)

                images[b,:,:,:] = img
                boxs[b,:box.shape[0],:] = box
                label[b,:box.shape[0]] = lab
                batch_rpn_match[b] = rpn_match[:, np.newaxis]
                batch_rpn_bbox[b] = rpn_box

                index=index+1

                b=b+1

            else:
                images[b, :, :, :] = img
                boxs[b, :box.shape[0], :] = box
                label[b, :box.shape[0]] = lab
                batch_rpn_match[b] = rpn_match[:, np.newaxis]
                batch_rpn_bbox[b] = rpn_box
                index = index + 1
                b = b + 1


            if b>=batch_size:
                yield [images,boxs,label,batch_rpn_match,batch_rpn_bbox]
                b = 0
            if index>= length:
                index = 0

def tt():
    aug = SSDAugmentation(512)
    for s in range(1000):
        img, box, lab = data_set.pull_item(s)
        print(box)
        visual.display_instances(img, box * 512)

        img, box, lab = aug(img, box, lab)
        #img = img+[104, 117, 123]
        img = ((img + [104, 117, 123]) / 255 - 0.5) * 2.0
        visual.display_instances(img*255, box * 512)

