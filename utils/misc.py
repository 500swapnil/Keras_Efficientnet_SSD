import tensorflow as tf
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np 

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'] 

@tf.function
def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    """Convert regressional location results of SSD into boxes in the form of (center_x, center_y, h, w).
    The conversion:
        $$predicted\_center * center_variance = \frac {real\_center - prior\_center} {prior\_hw}$$
        $$exp(predicted\_hw * size_variance) = \frac {real\_hw} {prior\_hw}$$
    We do it in the inverse direction here.
    Args:
        locations (batch_size, num_priors, 4): the regression output of SSD. It will contain the outputs as well.
        priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
        center_variance: a float used to change the scale of center.
        size_variance: a float used to change of scale of size.
    Returns:
        boxes:  priors: [[center_x, center_y, h, w]]. All the values
            are relative to the image size.
    """
    # priors can have one dimension less.
    if tf.rank(priors) + 1 == tf.rank(locations):
        priors = tf.expand_dims(priors, 0)
    return tf.concat([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        tf.math.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], axis=tf.rank(locations) - 1)

@tf.function
def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    # priors can have one dimension less
    if tf.rank(center_form_priors) + 1 == tf.rank(center_form_boxes):
        center_form_priors = tf.expand_dims(center_form_priors, 0)
    return tf.concat([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
        tf.math.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
    ], axis=tf.rank(center_form_boxes) - 1)

@tf.function(experimental_relax_shapes=True)
def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = tf.clip_by_value(right_bottom - left_top, 0.0, 10000)
    return hw[..., 0] * hw[..., 1]

@tf.function(experimental_relax_shapes=True)
def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = tf.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = tf.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

@tf.function
def center_form_to_corner_form(locations):
    return tf.concat([locations[..., :2] - locations[..., 2:]/2,
                     locations[..., :2] + locations[..., 2:]/2], tf.rank(locations) - 1)

@tf.function
def corner_form_to_center_form(boxes):
    return tf.concat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
         boxes[..., 2:] - boxes[..., :2]
    ], tf.rank(boxes) - 1)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, 4]
    boxes = box_scores[:, :-2]
    picked = []
    indexes = np.argsort(scores)[::-1]
    indexes = indexes[:candidate_size]
    
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        ).numpy()
        # print(indexes.shape)
        # print(iou<=iou_threshold)
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def draw_bboxes(bboxes, ax, color='red', labels=None, IMAGE_SIZE=[300, 300]):
    # image = (im - np.min(im))/np.ptp(im)
    # print(image.shape)
    if np.max(bboxes) < 10:
        bboxes[:, [0,2]] = bboxes[:, [0,2]]*IMAGE_SIZE[1]
        bboxes[:, [1,3]] = bboxes[:, [1,3]]*IMAGE_SIZE[0]
    for i, bbox in enumerate(bboxes):
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor=color, facecolor='none')
        # ax.add_patch(rect)
        ax.add_artist(rect)
        # print(int(bbox[-1]))
        if labels is not None:
            ax.text(bbox[0]+0.5,bbox[1]+0.5, CLASSES[int(labels[i] - 1)],  fontsize=20,
                horizontalalignment='left', verticalalignment='top', bbox=dict(facecolor=color, alpha=0.4))