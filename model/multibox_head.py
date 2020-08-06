from keras.layers import Conv2D, BatchNormalization, Activation, Dense, Concatenate, Flatten, Reshape, Dropout
from keras.regularizers import l2
from keras.engine.topology import Layer
from keras.initializers import Constant
import tensorflow.keras.backend as K


class Normalize(Layer):
    """Normalization layer as described in ParseNet paper.
    # Arguments
        scale: Default feature scale.
    # Input shape
        4D tensor with shape: (samples, rows, cols, channels)
    # Output shape
        Same as input
    # References
        http://cs.unc.edu/~wliu/papers/parsenet.pdf
    # TODO
        Add possibility to have one scale for all features.
    """
    def __init__(self, scale=20, **kwargs):
        self.scale = scale
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name=self.name+'_gamma', 
                                     shape=(input_shape[-1],),
                                     initializer=Constant(self.scale), 
                                     trainable=True)
        super(Normalize, self).build(input_shape)
        
    def call(self, x, mask=None):
        return self.gamma * K.l2_normalize(x, axis=-1)

def create_multibox_head(source_layers, num_priors, normalizations, num_classes=21, regularization=5e-4):
    mbox_conf = []
    mbox_loc = []
    for i, layer in enumerate(source_layers):
        x = layer
        name = x.name.split('/')[0]
        if normalizations is not None and normalizations[i] > 0:
            x = Normalize(normalizations[i], name=name + '_norm')(x)

        x1 = Conv2D(num_priors[i] * num_classes, 3, padding='same', kernel_regularizer=l2(regularization) ,name= name + '_mbox_conf')(x)
        x1 = Flatten(name=name + '_mbox_conf_flat')(x1)
        mbox_conf.append(x1)

        x2 = Conv2D(num_priors[i] * 4, 3, padding='same', kernel_regularizer=l2(regularization) ,name= name + '_mbox_loc')(x)
        x2 = Flatten(name=name + '_mbox_loc_flat')(x2)
        mbox_loc.append(x2)
    
    mbox_loc = Concatenate(axis=1, name='mbox_loc')(mbox_loc)
    mbox_loc = Reshape((-1, 4), name='mbox_loc_final')(mbox_loc)

    mbox_conf = Concatenate(axis=1, name='mbox_conf')(mbox_conf)
    mbox_conf = Reshape((-1, num_classes), name='mbox_conf_logits')(mbox_conf)
    # mbox_conf = Activation('softmax', name='mbox_conf_final')(mbox_conf)

    predictions = Concatenate(axis=2, name='predictions')([mbox_loc, mbox_conf])


    return predictions