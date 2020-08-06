import tensorflow as tf
from tensorflow import keras
from model.backbone import create_backbone
from model.multibox_head import create_multibox_head

def ssd(base_model_name, normalizations=[20, 20, 20, -1, -1, -1], num_priors=[4,6,6,6,4,4]):
    inputs, source_layers = create_backbone(base_model_name)
    output = create_multibox_head(source_layers, num_priors, normalizations)
    model = keras.Model(inputs, outputs=output)
    return model
