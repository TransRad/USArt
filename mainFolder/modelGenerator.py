import keras
from datetime import date
from utils.utilsFun import *
import matplotlib.pyplot as plt
import os
import json
from mainFolder.models import create_unet_model_2d_leaky_relu, create_unet_model_2d



def create_model(configs, mask=None):
    """
    main model generator
    :param configs:
    :return model:
    """
    print("ModelGenerator: Create model ... ")
    print(configs.model)
    if configs.model is None:
        configs.model = '2DUnet-leakyReLU'
    if configs.model == '2DUnet-leakyReLU':
        print('****************' + str(get_int('activation_leakyReLU', configs)))
        model = create_unet_model_2d_leaky_relu(input_image_size=(configs.img_size, configs.img_size, 2), n_labels=3,
                                                layers=get_int('layersin', configs), init_lr=get_int('lr', configs),
                                                alpha_val=get_int('activation_leakyReLU', configs),
                                                activity_regularizer_type=configs.activity_regularizer_type,
                                                kernel_regularizer_type=configs.kernel_regularizer_type,
                                                loss=configs.loss, mask=mask)
    elif configs.model == '2DUnet2channels':
        model = create_unet_model_2d(input_image_size=(configs.img_size, configs.img_size, 2), n_labels=2, layers=get_int('layersin', configs), 
                                    init_lr=get_int('lr', configs), output_activation='relu',  mode='regression')
    elif configs.model == '2DUnet':
        model = create_unet_model_2d(input_image_size=(configs.img_size, configs.img_size,1), n_labels=1, layers=get_int('layersin', configs), 
                                    init_lr=get_int('lr', configs), output_activation='relu',  mode='regression')
    else:
        print('Select the right type of Model: 2DUnet-leakyReLU or 2DUnet2channels')
        model = []
    path = f'{configs.output_dir_base}/{str(configs.ID)}/'
    if not os.path.exists(path):
        os.makedirs(path)
    file = f'{path}modelsummary.txt'
    with open(file, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    # model.summary()
    return model


def create_callbacks(configs=None, extra=None):
    """
    main callbacks' generator
    :param configs:
    :return callbacks:
    """
    print("ModelGenerator: Create the callbacks_out ... ")
    path = f'{configs.output_dir_base}/{str(configs.ID)}/'
    if not os.path.exists(path):
        os.makedirs(path)
    path_epoch = f'{configs.output_dir_base}/{str(configs.ID)}/model_per_epoch/'
    if not os.path.exists(path_epoch):
        os.makedirs(path_epoch)
    today = date.today()
    today = today.strftime("%y-%m-%d")
    # select model
    print(path_epoch)
    name = (((f'{path_epoch}/{today}.{configs.name}.' + str(get_int('batch_size', configs))  + '.' + str(configs.num_epochs))  + '.' ) + 
            str(configs.sample) + '.' ) + str(get_int('layersin', configs))
    if extra is not None:
        name = name + extra
    return ([
            keras.callbacks.CSVLogger(f'{name}.txt'),
            keras.callbacks.ModelCheckpoint(
                name + '.weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=0
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_ssim',
                patience=20,
                min_lr=0.000001,
                mode='max',
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(),
            keras.callbacks.History(),]
        if get_int('earlyStopActive', configs) is True
        else [
            keras.callbacks.CSVLogger(f'{name}.txt'),
            keras.callbacks.ModelCheckpoint(
                name + '.weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=0
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_ssim',
                patience=20,
                min_lr=0.000001,
                mode='max',
                verbose=1,
            ),
            keras.callbacks.History(),]
    )


def fit_model(model, callbacks, configs, x, y, x_valid, y_valid):
    """
    fit model
    :param callbacks1:
    :param y_test:
    :param x_test:
    :param y_valid:
    :param x_valid:
    :param y:
    :param x:
    :param configs:
    :param callbacks:
    :param model:
    :return fit:
    :return test_loss:
    """
    print("ModelGenerator: Training ... ")
    path = f'{configs.output_dir_base}{str(configs.ID)}/'
    if not os.path.exists(path):
        os.makedirs(path)
    fit = model.fit(x, y, validation_data=(x_valid, y_valid), epochs=configs.num_epochs,
                    batch_size=get_int('batch_size', configs), verbose=1, shuffle=True, callbacks=callbacks)

    path = path + 'saved_model.hdf5'
    model.save(path)
    return fit #, test_loss


def plot_results_fit(configs, fit):
    """
    plot fitting curves
    :param fit:
    :param configs:
    """
    print("ModelGenerator: Save curves plots: ")
    path = configs.output_dir_base + '/' + str(configs.ID) + '/Fit_plot/'
    if not os.path.exists(path):
        os.makedirs(path)
    today = date.today()
    today = today.strftime("%y-%m-%d")

    fig = plt.figure()
    train_loss = fit.history['loss']
    valid_loss = fit.history['val_loss']
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    fig.savefig('{}/model_loss_{}.{}.{}.{}.{}.jpg'.format(path, today, configs.name, get_int('batch_size', configs), configs.num_epochs, configs.sample))

    plt.close("all")
    print("\t" + path + "model_loss_ ....")

    fig = plt.figure()
    train_ssim = fit.history['ssim']
    valid_ssim = fit.history['val_ssim']
    plt.plot(train_ssim)
    plt.plot(valid_ssim)
    plt.title('ssim')
    plt.ylabel('ssim')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    fig.savefig('{}/model_ssim_{}.{}.{}.{}.{}.jpg'.format(path, today, configs.name, get_int('batch_size', configs), configs.num_epochs, configs.sample))

    plt.close("all")
    print("\t" + path + "model_ssim ....")

    fig = plt.figure()
    plt.plot(fit.history['lr'])
    plt.title('model learning rate')
    plt.ylabel('learning rate')
    plt.xlabel('epoch')
    fig.savefig('{}/model_lr_{}.{}.{}.{}.{}.jpg'.format(path, today,
                                                        configs.name,
                                                        get_int('batch_size', configs),
                                                        configs.num_epochs,
                                                        configs.sample))

    plt.close("all")
    print("\t" + path + "model_lr_ ....")

    return train_loss, valid_loss, train_ssim, valid_ssim

# used
def sub_tuning(range_in, i, configs, x_train, y_train, x_valid, y_valid, tuning_par=True, old_model=None, mask=None):

    """
    sub_tuning
    :param mask:
    :rtype: object
    :param old_model:
    :param i: parameter position: need if tuning_par is True in order to modify the json file
    :param range_in: parameter value: need if tuning_par is True in order to modify the json file
    :param tuning_par: if tuning_par is True, the script is running for tuning a model then modify the json file
                    if tuning_par is False the script is running for training a model
    :param index:
    :param ids:
    :param y_test:
    :param x_test:
    :param y_valid:
    :param x_valid:
    :param y_train:
    :param x_train:
    :param configs:
    """
    if tuning_par is True:
        print(" ---------------------- NEW iteration")
        print("Tuning: " + str(configs.ID) + ", " + configs['par'][i]['name'] + ".init = " + str(range_in))
        configs['par'][i]['int'] = range_in

    # configs.ID = configs.ID + 1
    # Train the model
    model = create_model(configs, mask=mask)
    if old_model is not None:
        model.set_weights(old_model.get_weights())
        print('Transfer weights **** ')
    callbacks = create_callbacks(configs)
    # callbacks1 = create_callbacks(configs, extra='extra')

    fit = fit_model(model, callbacks, configs, x_train, y_train, x_valid, y_valid)

    # Write the object to file.
    path = configs.output_dir_base + '/' + str(configs.ID) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    path = path + 'configs.json'
    with open(path, 'w') as jsonFile:
        json.dump(configs, jsonFile)

    # Plot fit curves
    plot_results_fit(configs, fit)


    return model

def prediction(data, model):
    """
    prediction
    :param data:
    :param model:
    :return unet_image_test:
    """
    print("ModelGenerator: Prediction ... ")
    unet_image_test_2 = model.predict(data)
    return unet_image_test_2