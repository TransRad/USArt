# https://github.com/ncullen93/Unet-ants/blob/master/code/models/create_unet_model.py
import tensorflow as tf
from keras.layers import Conv2D, Input, MaxPooling2D, Conv2DTranspose, UpSampling2D, Concatenate
from keras.layers import LeakyReLU
from keras.models import Model
import numpy as np
from keras.regularizers import l1, l2
import keras.backend as K
from keras import optimizers as opt

"""KIKI network."""
from utils.compile import default_model_compile


def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smoothing_factor) / (K.sum(y_true_f) + K.sum(y_pred_f) + smoothing_factor)


def loss_dice_coefficient_error(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def create_unet_model_2d(input_image_size,
                        n_labels=1,
                        layers=3,
                        lowest_resolution=16,
                        convolution_kernel_size=(3, 3),
                        deconvolution_kernel_size=(3, 3),
                        pool_size=(2, 2),
                        mode='regression',
                        output_activation='relu',
                        init_lr=0.0001):
    """
    Create a 2D Unet model
    Example
    -------
    unet_model = create_Unet_model2D( (100,100,1), 1, 4)
    """
    layers = np.arange(layers)
    number_of_classification_labels = n_labels

    inputs = Input(shape=input_image_size)

    ## ENCODING PATH ##

    encoding_convolution_layers = []
    pool = None
    for i in range(len(layers)):
        number_of_filters = lowest_resolution * 2 ** (layers[i])

        if i == 0:
            conv = Conv2D(filters=number_of_filters,
                        kernel_size=convolution_kernel_size,
                        activation='relu',
                        padding='same')(inputs)
        else:
            conv = Conv2D(filters=number_of_filters,
                        kernel_size=convolution_kernel_size,
                        activation='relu',
                        padding='same')(pool)

        encoding_convolution_layers.append(Conv2D(filters=number_of_filters,
                                                kernel_size=convolution_kernel_size,
                                                activation='relu',
                                                padding='same')(conv))

        if i < len(layers) - 1:
            pool = MaxPooling2D(pool_size=pool_size)(encoding_convolution_layers[i])

    ## DECODING PATH ##
    outputs = encoding_convolution_layers[len(layers) - 1]
    for i in range(1, len(layers)):
        number_of_filters = lowest_resolution * 2 ** (len(layers) - layers[i] - 1)
        tmp_deconv = Conv2DTranspose(filters=number_of_filters, kernel_size=deconvolution_kernel_size,
                                    padding='same')(outputs)
        tmp_deconv = UpSampling2D(size=pool_size)(tmp_deconv)
        outputs = Concatenate(axis=3)([tmp_deconv, encoding_convolution_layers[len(layers) - i - 1]])

        outputs = Conv2D(filters=number_of_filters, kernel_size=convolution_kernel_size,
                        activation='relu', padding='same')(outputs)
        outputs = Conv2D(filters=number_of_filters, kernel_size=convolution_kernel_size,
                        activation='relu', padding='same')(outputs)

    if mode == 'classification':
        if number_of_classification_labels == 1:
            outputs = Conv2D(filters=number_of_classification_labels, kernel_size=(1, 1),
                            activation='sigmoid')(outputs)
        else:
            outputs = Conv2D(filters=number_of_classification_labels, kernel_size=(1, 1),
                            activation='softmax')(outputs)

        unet_model = Model(inputs=inputs, outputs=outputs)

        if number_of_classification_labels == 1:
            unet_model.compile(loss=loss_dice_coefficient_error,
                            optimizer=opt.Adam(lr=init_lr), metrics=[dice_coefficient])
        else:
            unet_model.compile(loss='categorical_crossentropy',
                            optimizer=opt.Adam(lr=init_lr), metrics=['accuracy', 'categorical_crossentropy'])
    elif mode == 'regression':
        outputs = Conv2D(filters=number_of_classification_labels, kernel_size=(1, 1),
                        activation=output_activation)(outputs)
        unet_model = Model(inputs=inputs, outputs=outputs)
        # unet_model.compile(loss='mse', optimizer=opt.Adam(lr=init_lr))
        default_model_compile(model=unet_model, lr=init_lr, loss='mssim')
    else:
        raise ValueError('mode must be either `classification` or `regression`')

    return unet_model


def create_unet_model_2d_leaky_relu(input_image_size,
                                    n_labels=1,
                                    layers=2,
                                    lowest_resolution=16,
                                    convolution_kernel_size=(3, 3),
                                    deconvolution_kernel_size=(3, 3),
                                    pool_size=(2, 2),
                                    init_lr=0.0001,
                                    alpha_val=0.1,
                                    activity_regularizer_type=None,
                                    kernel_regularizer_type=None,
                                    activity_regularizer_step=0.00005,
                                    kernel_regularizer_step=0.00005,
                                    loss='fmssim',
                                    mask=None):
    """
    Create a 2D Unet model
    :param loss:
    :param mask:
    :param kernel_regularizer_step:
    :param activity_regularizer_step:
    :param kernel_regularizer_type:
    :param activity_regularizer_type:
    :param b2:
    :param b1:
    :param alpha_val:
    :param init_lr:
    :param pool_size:
    :param deconvolution_kernel_size:
    :param convolution_kernel_size:
    :param lowest_resolution:
    :param layers:
    :param n_labels:
    :param input_image_size: o
    :return unet_model:
    Example
    -------
    unet_model = create_Unet_model2D( (100,100,1), 1, 4)
    """
    if activity_regularizer_type == 'L1':
        activity_regularizer = l1(activity_regularizer_step)
    elif activity_regularizer_type == 'L2':
        activity_regularizer = l2(activity_regularizer_step)
    else:
        activity_regularizer = None

    if kernel_regularizer_type == 'L1':
        kernel_regularizer = l1(kernel_regularizer_step)
    elif kernel_regularizer_type == 'L2':
        kernel_regularizer = l2(kernel_regularizer_step)
    else:
        kernel_regularizer = None

    layers = np.arange(layers)
    number_of_classification_labels = n_labels

    inputs = Input(shape=input_image_size)

    activation_layer = None
    # ENCODING PATH
    encoding_convolution_layers = []
    pool = None
    for i in range(len(layers)):
        number_of_filters = lowest_resolution * 2 ** (layers[i])

        if i == 0:
            conv = Conv2D(filters=number_of_filters,
                        kernel_size=convolution_kernel_size,
                        activity_regularizer=activity_regularizer,
                        kernel_regularizer=kernel_regularizer,
                        activation=activation_layer,
                        padding='same')(inputs)
            conv = LeakyReLU(alpha=alpha_val)(conv)
        else:
            conv = Conv2D(filters=number_of_filters,
                        kernel_size=convolution_kernel_size,
                        activity_regularizer=activity_regularizer,
                        kernel_regularizer=kernel_regularizer,
                        activation=activation_layer,
                        padding='same')(pool)
            conv = LeakyReLU(alpha=alpha_val)(conv)

        conv = Conv2D(filters=number_of_filters,
                    kernel_size=convolution_kernel_size,
                    activity_regularizer=activity_regularizer,
                    kernel_regularizer=kernel_regularizer,
                    activation=activation_layer,
                    padding='same')(conv)
        conv = LeakyReLU(alpha=alpha_val)(conv)
        encoding_convolution_layers.append(conv)

        if i < len(layers) - 1:
            pool = MaxPooling2D(pool_size=pool_size)(encoding_convolution_layers[i])
            # pool = AveragePooling2D(pool_size=pool_size)(encoding_convolution_layers[i])
    #
    # DECODING PATH
    outputs = encoding_convolution_layers[len(layers) - 1]
    for i in range(1, len(layers)):
        number_of_filters = lowest_resolution * 2 ** (len(layers) - layers[i] - 1)
        tmp_deconv = Conv2DTranspose(filters=number_of_filters,
                                    kernel_size=deconvolution_kernel_size,
                                    activity_regularizer=activity_regularizer,
                                    kernel_regularizer=kernel_regularizer,
                                    padding='same')(outputs)
        tmp_deconv = UpSampling2D(size=pool_size)(tmp_deconv)
        outputs = Concatenate(axis=3)([tmp_deconv, encoding_convolution_layers[len(layers) - i - 1]])

        outputs = Conv2D(filters=number_of_filters,
                        kernel_size=convolution_kernel_size,
                        activity_regularizer=activity_regularizer,
                        kernel_regularizer=kernel_regularizer,
                        activation=activation_layer,
                        padding='same')(outputs)
        outputs = LeakyReLU(alpha=alpha_val)(outputs)
        outputs = Conv2D(filters=number_of_filters,
                        activity_regularizer=activity_regularizer,
                        kernel_regularizer=kernel_regularizer,
                        kernel_size=convolution_kernel_size,
                        activation=activation_layer,
                        padding='same')(outputs)
        outputs = LeakyReLU(alpha=alpha_val)(outputs)

    outputs = Conv2D(filters=number_of_classification_labels,
                    kernel_size=(1, 1),
                    activity_regularizer=activity_regularizer,
                    kernel_regularizer=kernel_regularizer,
                    activation=None)(outputs)

    unet_model = Model(inputs=inputs, outputs=outputs)
    # unet_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=init_lr, beta_1=b1, beta_2=b2),
    #                    metrics=[ssim])

    default_model_compile(model=unet_model, lr=init_lr, loss=loss, mask=mask)
    return unet_model