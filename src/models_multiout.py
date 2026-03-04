import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as layers
# for model illustration
from tensorflow.keras.utils import plot_model
# ---
# use previous defined layers in model.py
from models import reshape_output, contruct_base_conv
# ---
def simple_conv_2chs_dual(layer_filters=[50, 32, 16], bn=True, padding='same', kernel_size=3,
                pooling=True, dense_layers=[256, 11491], dense_activation='relu', input_shape=(60, 100, 10),
                dropout=0.6, activation='relu', output_shape_2ch=1151):
    """
    Create a simple convolutional neural network model.

    Args:
        layer_filters (list): List of layer filter sizes.
        bn (bool): Whether to use batch normalization.
        padding (str): Padding type.
        kernel_size (int): Convolutional kernel size.
        pooling (bool): Whether to use pooling layers.
        dense_layers (list): List of dense layer sizes.
        dense_activation (str): Activation function for dense layers.
        input_shape (tuple): Input shape of the model.
        dropout (float): Dropout rate.
        activation (str): Activation function for convolutional layers.

    Returns:
        tf.keras.Model: The constructed convolutional neural network model.

    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = contruct_base_conv(inputs, layer_filters=layer_filters,
                                 bn=bn, padding=padding,
                                 kernel_size=kernel_size, pooling=pooling,
                                 activation=activation, dropout=dropout)

# YC: previous code combine [inputshape outputshape],
# now separate them
    for neuron in dense_layers:
        x = tf.keras.layers.Dense(neuron, activation=dense_activation)(x)
#    x = tf.keras.layers.Dense(dense_layers[0], activation=dense_activation)(flatten)
#    for neuron in dense_layers[1:]:
#        x = tf.keras.layers.Dense(neuron, activation=dense_activation)(x)
# --
#    model_transfer_train = tf.keras.models.Model(inputs, x)
        output_shape = output_shape_2ch
        output1 = tf.keras.layers.Dense(output_shape, activation='selu', kernel_initializer='zeros')(x) # rainfall
        output2 = tf.keras.layers.Dense(output_shape, activation='selu', kernel_initializer='zeros')(x) # soil moisture

        output1 = reshape_output(output_shape, output1)
        output2 = reshape_output(output_shape, output2)
##### Line for two channels
        model_transfer_train = tf.keras.models.Model(inputs=inputs, outputs=[output1, output2])
##### Lines below for concatenate
#        concat = tf.keras.layers.Concatenate(axis=-1)([output1, output2])
#        model_transfer_train = tf.keras.models.Model(inputs, concat)
    return model_transfer_train



def simple_conv_2chs(layer_filters=[50, 32, 16], bn=True, padding='same', kernel_size=3,
                pooling=True, dense_layers=[256, 11491], dense_activation='relu', input_shape=(60, 100, 10),
                dropout=0.6, activation='relu', output_shape_2ch=1151):
    """
    Create a simple convolutional neural network model.

    Args:
        layer_filters (list): List of layer filter sizes.
        bn (bool): Whether to use batch normalization.
        padding (str): Padding type.
        kernel_size (int): Convolutional kernel size.
        pooling (bool): Whether to use pooling layers.
        dense_layers (list): List of dense layer sizes.
        dense_activation (str): Activation function for dense layers.
        input_shape (tuple): Input shape of the model.
        dropout (float): Dropout rate.
        activation (str): Activation function for convolutional layers.

    Returns:
        tf.keras.Model: The constructed convolutional neural network model.

    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = contruct_base_conv(inputs, layer_filters=layer_filters,
                                 bn=bn, padding=padding,
                                 kernel_size=kernel_size, pooling=pooling,
                                 activation=activation, dropout=dropout)

# YC: previous code combine [inputshape outputshape],
# now separate them
    for neuron in dense_layers:
        x = tf.keras.layers.Dense(neuron, activation=dense_activation)(x)
#    x = tf.keras.layers.Dense(dense_layers[0], activation=dense_activation)(flatten)
#    for neuron in dense_layers[1:]:
#        x = tf.keras.layers.Dense(neuron, activation=dense_activation)(x)
# --
#    model_transfer_train = tf.keras.models.Model(inputs, x)
        output_shape = output_shape_2ch//2
        output1 = tf.keras.layers.Dense(output_shape, activation='selu', kernel_initializer='zeros')(x) # rainfall
        output2 = tf.keras.layers.Dense(output_shape, activation='selu', kernel_initializer='zeros')(x) # soil moisture

        output1 = reshape_output(output_shape, output1)
        output2 = reshape_output(output_shape, output2)
        concat = tf.keras.layers.Concatenate(axis=-1)([output1, output2])
        model_transfer_train = tf.keras.models.Model(inputs, concat)
# Do I need to put a compile here?

#    output1 = tf.keras.layers.Dense(output_shape, activation='selu', kernel_initializer='zeros')(x)
#    output2 = tf.keras.layers.Dense(output_shape, activation='selu', kernel_initializer='zeros')(x)
#    output3 = tf.keras.layers.Dense(output_shape, activation='sigmoid', kernel_initializer='zeros')(x)

#    output1 = reshape_output(output_shape, output1)
#    output2 = reshape_output(output_shape, output2)
#    output3 = reshape_output(output_shape, output3)

#    concat = tf.keras.layers.Concatenate(axis=-2)([output1, output2, output3])
#    model_transfer_train = tf.keras.models.Model(inputs1, concat)
#    model_transfer_train.compile(loss=['mse'], optimizer='adam')



    return model_transfer_train

