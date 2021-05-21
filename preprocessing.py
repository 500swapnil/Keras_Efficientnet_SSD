from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf
from utils.augmentations import *

AUTO = tf.data.experimental.AUTOTUNE

@tf.function
def data_augment(image, boxes, labels):
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5) # Random Saturation
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_brightness(image, max_delta=0.15) # Random brightness
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5) # Random Contrast
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_hue(image, max_delta=0.2) # Random Hue
    image = random_lighting_noise(image)
    image, boxes = expand(image, boxes)
    image, boxes, labels = random_crop(image, boxes, labels) # Random Crop
    image, boxes = random_flip(image, boxes) # Random Flip

    return (image, boxes, labels)


def prepare_input(sample, convert_to_normal=True):
  img = tf.cast(sample['image'], tf.float32)
  # img = img - image_mean
  labels = sample['objects']['label']+1
  bbox = sample['objects']['bbox']
  if convert_to_normal:
    bbox = tf.stack([bbox[:,1], bbox[:,0], bbox[:,3], bbox[:,2]], axis=1)
  
  img = preprocess_input(img, mode='torch')
  # img = tf.image.resize(img, IMAGE_SIZE) / 255.0
  # img = tf.cast(img, tf.float32)
  # img = (img - image_mean) / image_std
  return (img, bbox, labels)


def join_target(image, bbox, labels, image_size, target_transform):
  locations, labels = target_transform(tf.cast(bbox, tf.float32), labels)
  labels = tf.one_hot(labels, 21, axis=1, dtype=tf.float32)
  targets = tf.concat([labels, locations], axis=1)
  return (tf.image.resize(image, image_size), targets)


def prepare_dataset(dataset, image_size, batch_size, target_transform, train=False):
  # dataset = dataset.cache() # This dataset fits in RAM
  dataset = dataset.map(prepare_input, num_parallel_calls=AUTO)
  
  if train:
    # Best practices for Keras:
    # Training dataset: repeat then batch
    # Evaluation dataset: do not repeat
    dataset = dataset.shuffle(1000)
    dataset = dataset.repeat()
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
  dataset = dataset.map(lambda image, boxes, labels: join_target(image, boxes, labels, image_size, target_transform), num_parallel_calls=AUTO)
  dataset = dataset.padded_batch(batch_size)
  dataset = dataset.prefetch(AUTO)
  return dataset


def prepare_for_prediction(file_path, image_size=[300, 300]):
    img = tf.io.read_file(file_path)
    img = decode_img(img, image_size)
    img = preprocess_input(img, mode='torch')
    return img
    
def decode_img(img,  image_size=[300, 300]):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    return tf.image.resize(img, image_size)
    