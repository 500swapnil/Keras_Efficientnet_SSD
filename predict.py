import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from utils.priors import *
from model.ssd import ssd
import matplotlib.pyplot as plt
from utils.post_processing import post_process
from matplotlib.image import imread
import os
from preprocessing import prepare_for_prediction

DATASET_DIR = './dataset'
IMAGE_SIZE = [300, 300]
BATCH_SIZE = 16
MODEL_NAME = 'B3'
checkpoint_filepath = './checkpoints/efficientnetb3_SSD.h5'
INPUT_DIR = './inputs'
OUTPUT_DIR = './outputs'

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

print("Building SSD Model with EfficientNet{0} backbone..".format(MODEL_NAME))
model = ssd(MODEL_NAME)

print("Loading Checkpoint..")
model.load_weights(checkpoint_filepath)

filenames = os.listdir(INPUT_DIR)
dataset = tf.data.Dataset.list_files(INPUT_DIR + '/*', shuffle=False)
dataset = dataset.map(prepare_for_prediction)
dataset = dataset.batch(BATCH_SIZE)

pred = model.predict(dataset, verbose=1)
predictions = post_process(pred, target_transform, confidence_threshold=0.4)

# dataset = dataset.unbatch()
print("Prediction Complete")
for i, path in enumerate(filenames):
    im = imread(os.path.join(INPUT_DIR, path))  
    fig, ax = plt.subplots(1, figsize=(15, 15))
    ax.imshow(im)
    pred_boxes, pred_scores, pred_labels = predictions[i]
    draw_bboxes(pred_boxes, ax , labels=pred_labels, color='red', IMAGE_SIZE=im.shape[:2])
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, 'out_'+ path), bbox_inches='tight', pad_inches=0)
    
print("Output is saved in", OUTPUT_DIR)

# reader = tf.WholeFileReader()
# key, value = reader.read(filename_queue)

# images = tf.image.decode_jpeg(value, channels=3)
# print(images)