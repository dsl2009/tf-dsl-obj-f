#coding=utf-8
import tensorflow as tf
from tensorflow.contrib import slim
from config.cfg import config_instace as cfg
import utils
from data_set.data_loader import q
import cv2
import numpy as np
from nets import inception_v2
import visual
from loss import losses
import glob
import time

#tf.enable_eager_execution()
def model(img):
    with slim.arg_scope(inception_v2.inception_v2_arg_scope()):
        with slim.arg_scope([slim.batch_norm],is_training=True):
            with slim.arg_scope([slim.conv2d], trainable=True):
                logits, end_point = inception_v2.inception_v2_base(img)

    c1 = end_point['Mixed_3c']
    c2 = end_point['Mixed_4e']
    c3 = end_point['Mixed_5c']
    vbs = slim.get_variables_to_restore()

    c3 = slim.conv2d(c3, 256, 1, 1, activation_fn=None)

    c2 = slim.conv2d(c2, 256, 1, 1, activation_fn=None) + tf.image.resize_bilinear(c3, size=tf.shape(c2)[1:3])

    c1 = slim.conv2d(c1, 256, 1, 1, activation_fn=None) + tf.image.resize_bilinear(c2, size=tf.shape(c1)[1:3])


    return c1,c2,c3,vbs

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

        roiss.append(rois)
        roi_gt_class_idss.append(roi_gt_class_ids)
        deltass.append(deltas)
    return tf.stack(roiss,axis=0),tf.stack(roi_gt_class_idss,axis=0),tf.stack(deltass,axis=0)

def fpn_classifier_graph(rois, feature_maps):
    x = utils.roi_align(rois, feature_maps, cfg)

    x = slim.conv2d(x,512,kernel_size=cfg.pool_shape,padding='VALID')
    x = slim.conv2d(x, 512, kernel_size=1)
    x1 = slim.dropout(x,keep_prob=1.0)
    mrcnn_class_logits = slim.fully_connected(x1,cfg.num_class)
    mrcnn_probs = slim.softmax(mrcnn_class_logits)

    x = slim.fully_connected(x,cfg.num_class*4)
    mrcnn_bbox = tf.reshape(x,shape=(-1,cfg.num_class,4))

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox

def predict(images,window):
    c1, c2, c3, vbs = model(images)
    fp = [c1, c2, c3]
    rpn_c_l = []
    r_p = []
    r_b = []
    for f in fp:
        rpn_class_logits, rpn_probs, rpn_bbox = rpn_graph(f)
        rpn_c_l.append(rpn_class_logits)
        r_p.append(rpn_probs)
        r_b.append(rpn_bbox)

    rpn_class_logits = tf.concat(rpn_c_l, axis=1)
    rpn_probs = tf.concat(r_p, axis=1)
    rpn_bbox = tf.concat(r_b, axis=1)
    rpn_rois = propsal(rpn_probs, rpn_bbox)

    mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(rpn_rois, fp)

    detections = utils.refine_detections_graph(rpn_rois,mrcnn_class, mrcnn_bbox,window,cfg)
    return detections

def loss(gt_boxs, images, input_rpn_bbox, input_rpn_match, label):
    c1, c2, c3, vbs= model(images)
    fp = [c1, c2, c3]
    rpn_c_l = []
    r_p = []
    r_b = []
    for f in fp:
        rpn_class_logits, rpn_probs, rpn_bbox = rpn_graph(f)
        rpn_c_l.append(rpn_class_logits)
        r_p.append(rpn_probs)
        r_b.append(rpn_bbox)
    rpn_class_logits = tf.concat(rpn_c_l, axis=1)
    rpn_probs = tf.concat(r_p, axis=1)
    rpn_bbox = tf.concat(r_b, axis=1)
    rpn_rois = propsal(rpn_probs, rpn_bbox)
    rois, target_class_ids, target_bbox = detection_target(rpn_rois, label, gt_boxs)
    mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(rois, fp)
    mrcnn_class_logits = tf.squeeze(mrcnn_class_logits, axis=[1, 2])
    rpn_class_loss = losses.rpn_class_loss_graph(input_rpn_match, rpn_class_logits)
    rpn_bbox_loss = losses.rpn_bbox_loss_graph(input_rpn_bbox, input_rpn_match, rpn_bbox, cfg)
    class_loss = losses.mrcnn_class_loss_graph(target_class_ids, mrcnn_class_logits)
    bbox_loss = losses.mrcnn_bbox_loss_graph(target_bbox, target_class_ids, mrcnn_bbox)
    tf.losses.add_loss(rpn_class_loss)
    tf.losses.add_loss(rpn_bbox_loss)
    tf.losses.add_loss(class_loss)
    tf.losses.add_loss(bbox_loss)
    total_loss = tf.losses.get_losses()
    tf.summary.scalar(name='rpn_class_loss', tensor=rpn_class_loss)
    tf.summary.scalar(name='rpn_bbox_loss', tensor=rpn_bbox_loss)
    tf.summary.scalar(name='class_loss', tensor=class_loss)
    tf.summary.scalar(name='bbox_loss', tensor=bbox_loss)
    sum_op = tf.summary.merge_all()
    train_tensors = tf.identity(total_loss, 'ss')
    return train_tensors, sum_op, vbs

def detect():

    ig = tf.placeholder(shape=(1, 512, 512, 3), dtype=tf.float32)
    wind = tf.placeholder(shape=(4,1),dtype=tf.float32)
    detections = predict(images=ig,window=wind)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '/home/dsl/all_check/face_detect/nn_faster_rcnn_sec/model.ckpt-32467')
        for ip in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/VOCdevkit/VOCdevkit/VOC2012/JPEGImages/*.jpg'):
            print(ip)
            img = cv2.imread(ip)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            org, window, scale, padding, crop = utils.resize_image(img, min_dim=512, max_dim=512)
            window = np.asarray(window)/cfg.image_size[0]*1.0

            window = np.reshape(window,[4,1])



            img = (org/ 255.0-0.5)*2
            img = np.expand_dims(img, axis=0)
            t = time.time()
            detects = sess.run([detections],feed_dict={ig:img,wind:window})
            print(time.time()-t)
            arr = detects[0]

            ix = np.where(np.sum(arr,axis=1)>0)
            box = arr[ix]
            boxes = box[:,0:4]
            label = box[:,4]
            score = box[:,5]
            visual.display_instances_title(org, np.asarray(boxes) * 512, class_ids=label,
                                           class_names=cfg.VOC_CLASSES, scores=score)






def run():
    pl_images = tf.placeholder(shape=[cfg.batch_size, cfg.image_size[0], cfg.image_size[1], 3], dtype=tf.float32)
    pl_gt_boxs = tf.placeholder(shape=[cfg.batch_size, 50, 4], dtype=tf.float32)
    pl_label = tf.placeholder(shape=[cfg.batch_size, 50], dtype=tf.int32)
    pl_input_rpn_match = tf.placeholder(shape=[cfg.batch_size, cfg.total_anchors, 1], dtype=tf.int32)
    pl_input_rpn_bbox = tf.placeholder(shape=[cfg.batch_size, cfg.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=tf.float32)

    train_tensors, sum_op, vbs = loss(pl_gt_boxs, pl_images, pl_input_rpn_bbox, pl_input_rpn_match, pl_label)


    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
    train_op = slim.learning.create_train_op(train_tensors, optimizer)

    saver = tf.train.Saver(vbs)

    def restore(sess):
        saver.restore(sess, '/home/dsl/all_check/face_detect/nn_faster_rcnn/model.ckpt-86737')

    sv = tf.train.Supervisor(logdir='/home/dsl/all_check/face_detect/nn_faster_rcnn_sec', summary_op=None, init_fn=restore)

    with sv.managed_session() as sess:
        for step in range(1000000000):


            images, boxs, label, input_rpn_match, input_rpn_bbox = q.get()
            gt_boxs = utils.norm_boxes(boxs,shape=cfg.image_size)

            feed_dict = {pl_images: images, pl_gt_boxs: gt_boxs,
                         pl_label: label,pl_input_rpn_bbox:input_rpn_bbox,
                         pl_input_rpn_match:input_rpn_match}
            t = time.time()
            ls = sess.run(train_op, feed_dict=feed_dict)
            if step % 10 == 0:
                print(time.time()-t)
                summaries = sess.run(sum_op, feed_dict=feed_dict)
                sv.summary_computed(sess, summaries)
                print(ls)




def eager_run():

    tf.enable_eager_execution()
    for s in range(10):
        images, boxs, label, input_rpn_match, input_rpn_bbox = q.get()
        print(input_rpn_bbox.shape)
        gt_boxs = utils.norm_boxes(boxes=boxs,shape=cfg.image_size)
        c1,c2,c3,v = model(images)
        fp = [c1,c2,c3]
        rpn_c_l = []
        r_p = []
        r_b = []
        for f in fp:
            rpn_class_logits, rpn_probs, rpn_bbox = rpn_graph(f)
            rpn_c_l.append(rpn_class_logits)
            r_p.append(rpn_probs)
            r_b.append(rpn_bbox)
        rpn_class_logits = tf.concat(rpn_c_l,axis=1)
        rpn_probs = tf.concat(r_p, axis=1)
        rpn_bbox = tf.concat(r_b, axis=1)


        rpn_rois = propsal(rpn_probs,rpn_bbox)

        rois, target_class_ids, target_bbox = detection_target(rpn_rois,label,gt_boxs)




        mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(rois,fp)

        mrcnn_class_logits = tf.squeeze(mrcnn_class_logits,axis=[1,2])

        rpn_class_loss = losses.rpn_class_loss_graph(input_rpn_match,rpn_class_logits)

        rpn_bbox_loss = losses.rpn_bbox_loss_graph(input_rpn_bbox,input_rpn_match,rpn_bbox,cfg)

        class_loss = losses.mrcnn_class_loss_graph(target_class_ids,mrcnn_class_logits)


        bbox_loss = losses.mrcnn_bbox_loss_graph(target_bbox,target_class_ids,mrcnn_bbox)

def eager_val():
    tf.enable_eager_execution()
    for s in range(10):
        images, boxs, label, input_rpn_match, input_rpn_bbox = q.get()
        predict(images)


if __name__ == '__main__':
    detect()
