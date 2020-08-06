import tensorflow as tf

@tf.function
def random_lighting_noise(image):
    if tf.random.uniform([]) > 0.5:
        channels = tf.unstack(image, axis=-1)
        channels = tf.random.shuffle(channels)
        image = tf.stack([channels[0], channels[1], channels[2]], axis=-1)
    return image


@tf.function
def bbox_area(bbox):
    return (bbox[:,2] - bbox[:,0]) * (bbox[:,3] - bbox[:,1])
 
@tf.function
def random_crop(image, bbox, labels):
    # ar_ = bbox_area(bbox)
    elems = tf.convert_to_tensor([0.1, 0.3, 0.5, 0.9, 1])
    min_object_covered = tf.random.shuffle(elems)[0]
    if min_object_covered == tf.constant(1.0):
        return image, bbox, labels
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
                                          tf.shape(image),
                                          bounding_boxes=tf.expand_dims(bbox, 0),
                                          min_object_covered=min_object_covered,
                                          area_range=[0.1, 1],
                                          aspect_ratio_range=[0.5,2])
    clip_box = bbox_for_draw[0][0]
    distorted_image = tf.slice(image, begin, size)
    image_size = tf.shape(image)[:2]
    scale_factor = tf.cast(image_size / size[:2], tf.float32)
    y_min = bbox[:,1] - clip_box[0]
    x_min = bbox[:,0] - clip_box[1]
    y_max = bbox[:,3] - clip_box[0]
    x_max = bbox[:,2] - clip_box[1]
    new_bbox = tf.stack([x_min, y_min, x_max, y_max], axis=1)

    centers_x = (bbox[:,0] + bbox[:,2]) / 2
    centers_y = (bbox[:,1] + bbox[:,3]) / 2

    # new_centers = tf.stack([centers_x - clip_box[1], centers_y - clip_box[0]], axis=1)

    left_check = centers_x > clip_box[1]
    top_check = centers_y > clip_box[0]
    right_check = centers_x < clip_box[3]
    bottom_check = centers_y < clip_box[2]

    vertical_check = tf.logical_and(top_check, bottom_check)
    horizontal_check = tf.logical_and(left_check, right_check)
    mask = tf.logical_and(vertical_check, horizontal_check)
    # print(mask, tf.stack([centers_x, centers_y], axis=1), clip_box)
    new_labels = tf.boolean_mask(labels, mask)
    new_bbox = tf.boolean_mask(new_bbox, mask)
    # new_centers = tf.boolean_mask(new_centers, mask)

    scale = tf.stack([scale_factor[1], scale_factor[0], scale_factor[1], scale_factor[0]])

    scale = tf.broadcast_to(scale, tf.shape(new_bbox))
    new_bbox = tf.clip_by_value(new_bbox, 0, 1)
    new_bbox = new_bbox * scale
    # new_centers = new_centers * scale_factor[::-1]
    new_bbox = tf.clip_by_value(new_bbox, 0, 1)
    non_zero_area_mask = bbox_area(new_bbox) > tf.constant(0, tf.float32)

    new_labels = tf.boolean_mask(new_labels, non_zero_area_mask)
    new_bbox = tf.boolean_mask(new_bbox, non_zero_area_mask)

    is_empty = tf.equal(tf.size(new_labels), 0)
    image = tf.cond(is_empty, lambda: image, lambda: distorted_image)
    bbox = tf.cond(is_empty, lambda: bbox, lambda: new_bbox)
    labels = tf.cond(is_empty, lambda: labels, lambda: new_labels)
    return image, bbox, labels

@tf.function
def random_flip(image, boxes, flip_prob=tf.constant(0.5)):
    if tf.random.uniform([]) > flip_prob:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack([1 - boxes[:,2], boxes[:,1],1- boxes[:,0], boxes[:,3]], axis=1)
    return (image, boxes)

@tf.function
def get_indices_from_slice(top, left, height, width):
    a = tf.reshape(tf.range(top, top + height),[-1,1])
    b = tf.range(left,left+width)
    A = tf.reshape(tf.tile(a,[1,width]),[-1])
    B = tf.tile(b,[height])
    indices = tf.stack([A,B], axis=1)
    return indices

@tf.function
def expand(image, boxes, expand_prob = tf.constant(0.5)):
      if tf.random.uniform([]) > expand_prob:
          return image, boxes

      image_shape = tf.cast(tf.shape(image), tf.float32)
      ratio = tf.random.uniform([], 1, 4)
      left = tf.math.round(tf.random.uniform([], 0, image_shape[1]*ratio - image_shape[1]))
      top = tf.math.round(tf.random.uniform([], 0, image_shape[0]*ratio - image_shape[0]))
      new_height = tf.math.round(image_shape[0]*ratio)
      new_width = tf.math.round(image_shape[1]*ratio)
      expand_image = tf.zeros(( new_height, new_width , image_shape[2]), dtype=tf.float32)
      indices = get_indices_from_slice(int(top), int(left), int(image_shape[0]), int(image_shape[1]))
      expand_image = tf.tensor_scatter_nd_update(expand_image, indices, tf.reshape(image, [-1,3]))

      image = expand_image
      xmin = (boxes[:,0] * image_shape[1] + left) / new_width
      ymin = (boxes[:,1] * image_shape[0] + top) / new_height
      xmax = (boxes[:,2] * image_shape[1] + left) / new_width
      ymax = (boxes[:,3] * image_shape[0] + top) / new_height

      boxes = tf.stack([xmin, ymin, xmax, ymax], axis=1)
      return image, boxes