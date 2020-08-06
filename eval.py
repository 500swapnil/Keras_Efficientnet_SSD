import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from utils.priors import *
from model.ssd import ssd
import os
from preprocessing import prepare_dataset
from utils.post_processing import post_process
from utils.voc_evaluation import eval_detection_voc
from tqdm import tqdm
from pprint import pprint

DATASET_DIR = './dataset'
IMAGE_SIZE = [300, 300]
BATCH_SIZE = 16
MODEL_NAME = 'B3'
checkpoint_filepath = './checkpoints/efficientnetb3_SSD.h5'

# train2012 = tfds.load('voc/2012', data_dir=DATASET_DIR, split='train')
# valid2012 = tfds.load('voc/2012', data_dir=DATASET_DIR, split='validation')
print("Loading Test Data..")
test_data = tfds.load('voc', data_dir=DATASET_DIR, split='test')
# number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()
number_test = 4952
print("Number of Test Files:", number_test)

with open('./labels.txt') as f:
    CLASSES = f.read().splitlines()

iou_threshold = 0.5
center_variance = 0.1
size_variance = 0.2

specs = [
                SSDSpec(38, 8, SSDBoxSizes(30, 60), [2]),
                SSDSpec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
                SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
                SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
                SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
                SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
        ]


priors = generate_ssd_priors(specs, IMAGE_SIZE[0])
target_transform = MatchPrior(priors, center_variance, size_variance, iou_threshold)

# instantiate the datasets
validation_dataset = prepare_dataset(test_data, IMAGE_SIZE, BATCH_SIZE, target_transform, train=False)

print("Building SSD Model with EfficientNet{0} backbone..".format(MODEL_NAME))
model = ssd(MODEL_NAME)

print("Loading Checkpoint..")
model.load_weights(checkpoint_filepath)

validation_steps = number_test // BATCH_SIZE + 1
print("Number of Test Batches:", validation_steps)

test_bboxes = []
test_labels = []
test_difficults = []
for sample in test_data:
    label = sample['objects']['label'].numpy()
    bbox = sample['objects']['bbox'].numpy()[:,[1, 0, 3, 2]]
    is_difficult = sample['objects']['is_difficult'].numpy()

    test_bboxes.append(bbox)
    test_labels.append(label)
    test_difficults.append(is_difficult)

print("Evaluating..")
pred_bboxes = []
pred_labels = []
pred_scores = []
for batch in tqdm(validation_dataset, total=validation_steps):
    pred = model.predict_on_batch(batch)
    predictions = post_process(pred, target_transform, confidence_threshold=0.2)
    for prediction in predictions:
        boxes, scores, labels = prediction
        pred_bboxes.append(boxes)
        pred_labels.append(labels.astype(int) - 1)
        pred_scores.append(scores)

answer = eval_detection_voc(pred_bboxes=pred_bboxes,
                   pred_labels=pred_labels, 
                   pred_scores=pred_scores, 
                   gt_bboxes=test_bboxes, 
                   gt_labels=test_labels, 
                   gt_difficults=test_difficults, 
                   use_07_metric=True)
print("*"*100)
print("Average Precisions")
ap_dict = dict(zip(CLASSES, answer['ap']))
pprint(ap_dict)
print("*"*100)
print("Mean Average Precision:", answer['map'])