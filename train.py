import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from utils.priors import *
from model.ssd import ssd
from model.loss import multibox_loss
import os
from preprocessing import prepare_dataset
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint

DATASET_DIR = './dataset'
IMAGE_SIZE = [300, 300]
BATCH_SIZE = 16
MODEL_NAME = 'B0'
EPOCHS = 20
checkpoint_filepath = None # './checkpoints/efficientnetb3_SSD.h5'
base_lr = 1e-3 if checkpoint_filepath is None else 1e-5

train2012 = tfds.load('voc/2012', data_dir=DATASET_DIR, split='train')
valid2012 = tfds.load('voc/2012', data_dir=DATASET_DIR, split='validation')
print("Loading Data..")
train2007 = tfds.load("voc", data_dir=DATASET_DIR, split='train')
valid2007 = tfds.load("voc", data_dir=DATASET_DIR, split='validation')

train_data = train2007.concatenate(valid2007).concatenate(train2012).concatenate(valid2012)

number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
# number_train = 5011
print("Number of Training Files:", number_train)

test_data = tfds.load('voc', data_dir=DATASET_DIR, split='test')
number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()
# number_test = 4952
print("Number of Test Files:", number_test)

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
training_dataset = prepare_dataset(train_data, IMAGE_SIZE, BATCH_SIZE, target_transform, train=True)
validation_dataset = prepare_dataset(test_data, IMAGE_SIZE, BATCH_SIZE, target_transform, train=False)

print("Building SSD Model with EfficientNet{0} backbone..".format(MODEL_NAME))
model = ssd(MODEL_NAME)
steps_per_epoch = number_train // BATCH_SIZE
validation_steps = number_test // BATCH_SIZE
print("Number of Train Batches:", steps_per_epoch)
print("Number of Test Batches:", validation_steps)


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=1e-5, verbose=1)
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr),
    loss = multibox_loss
)

if checkpoint_filepath is not None:
    print("Continuing Training from", checkpoint_filepath)
    model.load_weights(checkpoint_filepath)
else:
    print("Training from with only base model pretrained on imagenet")


history = model.fit(training_dataset, 
                    validation_data=validation_dataset,
                    steps_per_epoch=steps_per_epoch, 
                    validation_steps=validation_steps,
                    epochs=EPOCHS,
                    callbacks=[reduce_lr,checkpoint]) 