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
    print(deltas)
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
    images, boxs, label, rpn_match, rpn_bbox = q.get()

    net,vbs = model(images)

    rpn_class_logits, rpn_probs, rpn_bbox = rpn_graph(net)
    rpn_rois = propsal(rpn_probs,rpn_bbox)
    print(rpn_rois)

if __name__ == '__main__':
    eager_run()
