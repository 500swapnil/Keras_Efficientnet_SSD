import tensorflow as tf
import numpy as np
from collections import namedtuple
from utils.misc import *
Prediction = namedtuple('Prediction', ('boxes', 'scores', 'labels'))

def batched_nms(boxes, scores, idxs, iou_threshold, top_k=100):
    """
    Performs non-maximum suppression in a batched fashion.
    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.
    Parameters
    ----------
    boxes : Tensor[N, 4]
        boxes where NMS will be performed. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    idxs : Tensor[N]
        indices of the categories for each one of the boxes.
    iou_threshold : float
        discards all overlapping boxes
        with IoU < iou_threshold
    Returns
    -------
    keep : Tensor
        int64 tensor with the indices of
        the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    if tf.size(boxes) == 0:
        return tf.convert_to_tensor([],dtype=tf.int32)
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = tf.reduce_max(boxes)
    offsets = idxs * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = tf.image.non_max_suppression(boxes_for_nms, scores, top_k, iou_threshold)
    return keep

def post_process(detections, target_transform, confidence_threshold=0.01, top_k=100, iou_threshold=0.5):
    batch_boxes = detections[:, :, 21:]
    if not tf.is_tensor(batch_boxes):
        batch_boxes = tf.convert_to_tensor(batch_boxes)
    batch_scores = tf.nn.softmax(detections[:, :, :21], axis=2)

    batch_boxes = convert_locations_to_boxes(batch_boxes, target_transform.center_form_priors, target_transform.center_variance, target_transform.size_variance)
    batch_boxes = center_form_to_corner_form(batch_boxes)
    
    batch_size = tf.shape(batch_scores)[0]
    results = []
    for image_id in range(batch_size):
        scores, boxes = batch_scores[image_id], batch_boxes[image_id]  # (N, #CLS) (N, 4)

        num_boxes = tf.shape(scores)[0]
        num_classes = tf.shape(scores)[1]
        boxes = tf.reshape(boxes, [num_boxes, 1, 4])
        boxes = tf.broadcast_to(boxes, [num_boxes, num_classes, 4])
        labels = tf.range(num_classes, dtype=tf.float32)
        labels = tf.reshape(labels, [1, num_classes])
        labels = tf.broadcast_to(labels, tf.shape(scores))

        # remove predictions with the background label
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        labels = labels[:, 1:]
        
        # batch everything, by making every class prediction be a separate instance
        boxes = tf.reshape(boxes, [-1, 4])
        scores = tf.reshape(scores, [-1])
        labels = tf.reshape(labels, [-1])
        
        # remove low scoring boxes
        low_scoring_mask = scores > confidence_threshold
        boxes, scores, labels = tf.boolean_mask(boxes, low_scoring_mask), tf.boolean_mask(scores, low_scoring_mask), tf.boolean_mask(labels, low_scoring_mask)
        
        keep = batched_nms(boxes, scores, labels, iou_threshold, top_k)
        boxes, scores, labels = tf.gather(boxes, keep), tf.gather(scores, keep), tf.gather(labels, keep)
        results.append(Prediction(boxes.numpy(), scores.numpy(), labels.numpy()))
    return results