#coding=utf-8
import tensorflow as tf
from tensorflow.contrib import slim
from config.cfg import config_instace as cfg
import utils
from data_set.data_loader import q
import cv2
import numpy as np

import visual
import time
import glob
#tf.enable_eager_execution()
def model(inputs):
    source = []
    with tf.variable_scope('vgg_16',default_name=None, values=[inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d,  slim.max_pool2d]):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, kernel_size=3,stride=1, scope='pool5')
            #vbs = slim.get_trainable_variables()
            vbs = None
    return net,vbs

def rpn_graph(feature_map,anchors_per_location=9):
    shared = slim.conv2d(feature_map,512,3,activation_fn=slim.nn.relu)
    x = slim.conv2d(shared,2 * anchors_per_location,kernel_size=1,padding='VALID',activation_fn=None)
    rpn_class_logits = tf.reshape(x,shape=[tf.shape(x)[0],-1,2])
    rpn_probs = slim.nn.softmax(rpn_class_logits)
    x = slim.conv2d(shared, 4 * anchors_per_location, kernel_size=1, padding='VALID', activation_fn=None)
    rpn_bbox =tf.reshape(x,shape=[tf.shape(x)[0],-1,4])
    return [rpn_class_logits, rpn_probs, rpn_bbox]


def propsal(rpn_probs, rpn_bbox):
    scores = rpn_probs[:, :, 1]

    deltas = rpn_bbox
    deltas = deltas * np.reshape(cfg.RPN_BBOX_STD_DEV, [1, 1, 4])

    anchors = cfg.norm_anchors


    pre_nms_limit = tf.minimum(6000, cfg.total_anchors)
    ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                     name="top_anchors").indices
    window = np.array([0, 0, 1, 1], dtype=np.float32)
    result = []
    for b in range(cfg.batch_size):

        scores_tp = tf.gather(scores[b,:],ix[b,:])
        deltas_tp = tf.gather(deltas[b, :], ix[b, :])
        pre_nms_anchors_tp = tf.gather(anchors, ix[b, :])

        boxes_tp = utils.apply_box_deltas_graph(pre_nms_anchors_tp,deltas_tp)
        boxes_tp = utils.clip_boxes_graph(boxes_tp,window)

        props = utils.nms(boxes_tp,scores_tp,cfg)
        result.append(props)
    return tf.stack(result,axis=0)

def detection_target(input_proposals, input_gt_class_ids, input_gt_boxes):
    roiss = []
    roi_gt_class_idss = []
    deltass = []
    for b in range(cfg.batch_size):
        proposals = input_proposals[b,:,:]
        gt_class_ids = input_gt_class_ids[b,:]
        gt_boxes = input_gt_boxes[b,:,:]

        proposals, _ = utils.trim_zeros_graph(proposals, name="trim_proposals")
        gt_boxes, non_zeros = utils.trim_zeros(gt_boxes, name="trim_gt_boxes")
        gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                       name="trim_gt_class_ids")

        crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
        non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
        crowd_boxes = tf.gather(gt_boxes, crowd_ix)

        gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
        gt_boxes = tf.gather(gt_boxes, non_crowd_ix)



        overlaps = utils.overlaps_graph(proposals, gt_boxes)

        crowd_overlaps = utils.overlaps_graph(proposals, crowd_boxes)
        crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)


        roi_iou_max = tf.reduce_max(overlaps, axis=1)

        positive_roi_bool = (roi_iou_max >= 0.5)
        positive_indices = tf.where(positive_roi_bool)[:, 0]


        negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]


        positive_count = int(cfg.TRAIN_ROIS_PER_IMAGE *
                             cfg.ROI_POSITIVE_RATIO)
        positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
        positive_count = tf.shape(positive_indices)[0]

        r = 1.0 / cfg.ROI_POSITIVE_RATIO
        negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
        negative_indices = tf.random_shuffle(negative_indices)[:negative_count]

        positive_rois = tf.gather(proposals, positive_indices)
        negative_rois = tf.gather(proposals, negative_indices)


        # Assign positive ROIs to GT boxes.
        positive_overlaps = tf.gather(overlaps, positive_indices)

        roi_gt_box_assignment = tf.argmax(positive_overlaps, axis=1)

        roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
        roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)


        deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
        deltas /= cfg.BBOX_STD_DEV

        rois = tf.concat([positive_rois, negative_rois], axis=0)
        N = tf.shape(negative_rois)[0]
        P = tf.maximum(cfg.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
        rois = tf.pad(rois, [(0, P), (0, 0)])
        roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
        roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
        deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
        print(roi_gt_class_ids)
        roiss.append(rois)
        roi_gt_class_idss.append(roi_gt_class_ids)
        deltass.append(deltas)
    return tf.stack(roiss,axis=0),tf.stack(roi_gt_class_idss,axis=0),tf.stack(deltass,axis=0)

def fpn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True):

    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_classifier")([rois, image_meta] + feature_maps)





def run():
    images = tf.placeholder(shape=[cfg.batch_size, cfg.image_size[0], cfg.image_size[1], 3], dtype=tf.float32)
    boxs = tf.placeholder(shape=[cfg.batch_size, 50, 4], dtype=tf.float32)
    label = tf.placeholder(shape=[cfg.batch_size, 50], dtype=tf.int32)
    rpn_match = tf.placeholder(
        [cfg.batch_size, cfg.total_anchors, 1], dtype=tf.int32)
    rpn_bbox = tf.placeholder(
        [cfg.batch_size, cfg.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=tf.float32)

    net = model(images)

    rpn_class_logits, rpn_probs, rpn_bbox = rpn_graph(net)






def eager_run():

    tf.enable_eager_execution()
    for s in range(10):
        images, boxs, label, input_rpn_match, input_rpn_bbox = q.get()

        gt_boxs = utils.norm_boxes(boxes=boxs,shape=cfg.image_size)


        net,vbs = model(images)

        rpn_class_logits, rpn_probs, rpn_bbox = rpn_graph(net)
        rpn_rois = propsal(rpn_probs,rpn_bbox)

        rois, roi_gt_class_ids, deltas = detection_target(rpn_rois,label,gt_boxs)

if __name__ == '__main__':
    eager_run()
