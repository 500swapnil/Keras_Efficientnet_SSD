import efficientnet.keras as efn
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, BatchNormalization, Activation, Dense, Concatenate, Flatten, Reshape, Dropout
from keras.regularizers import l2

source_layers_to_extract = {
    'B0': ['block3b_add', 'block5c_add', 'block7a_project_bn'],
    'B1': ['block3c_add', 'block5d_add', 'block7b_add'],
    'B2': ['block3c_add', 'block5d_add', 'block7b_add'],
    'B3': ['block3c_add', 'block5e_add', 'block7b_add'],
    'B4': ['block3d_add', 'block5f_add', 'block7b_add'],
    'B5': ['block3e_add', 'block5g_add', 'block7c_add'],
    'B6': ['block3f_add', 'block5h_add', 'block7c_add'],
    'B7': ['block3g_add', 'block5j_add', 'block7d_add'],
}

                     #    f   k  s    p
extra_layers_params = [[(128, 1, 1, 'same'), (256, 3, 2, 'same')],
                       [(128, 1, 1, 'same'), (256, 3, 1, 'valid')],
                       [(128, 1, 1, 'same'), (256, 3, 1, 'valid')]]

def remove_dropout(model):
    for layer in model.layers:
        if isinstance(layer, Dropout):
            layer.rate = 0
    model_copy = keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights())
    del model
    return model_copy

def add_extras(extras, x, regularization=5e-4):
    features = []
    for extra_layer in extras:
        x = add_layer(extra_layer[0], x, regularization)
        x = add_layer(extra_layer[1], x, regularization)
        features.append(x)
    return features

def add_layer(layer_params, x, regularization=5e-4):
    filters, kernel_size, stride, padding = layer_params
    x = Conv2D(filters, kernel_size, stride, padding=padding, kernel_regularizer=l2(regularization))(x)
    x = Activation('relu')(x)
    return x

def create_base_model(base_model_name, pretrained=True, IMAGE_SIZE=[300, 300]):
    if pretrained is False:
        weights = None
    else:
        weights = "imagenet"
    if base_model_name == 'B0':
        base = efn.EfficientNetB0(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])
    elif base_model_name == 'B1':
        base = efn.EfficientNetB1(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])
    elif base_model_name == 'B2':
        base = efn.EfficientNetB2(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])
    elif base_model_name == 'B3':
        base = efn.EfficientNetB3(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])
    elif base_model_name == 'B4':
        base = efn.EfficientNetB4(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])
    elif base_model_name == 'B5':
        base = efn.EfficientNetB5(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])
    elif base_model_name == 'B6':
        base = efn.EfficientNetB6(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])
    elif base_model_name == 'B7':
        base = efn.EfficientNetB7(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])
    base = remove_dropout(base)
    base.trainable = True
    return base



def create_backbone(base_model_name, pretrained=True, IMAGE_SIZE=[300, 300], regularization=5e-4):
    source_layers = []
    base = create_base_model(base_model_name, pretrained, IMAGE_SIZE)
    layer_names = source_layers_to_extract[base_model_name]
    for name in layer_names:
        source_layers.append(base.get_layer(name).output)
    
    x = source_layers[-1]
    source_layers.extend(add_extras(extra_layers_params, x))
    return base.input, source_layers


