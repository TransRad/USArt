import numpy as np
import matplotlib.pyplot as plt
import MRDLin 
import os
import copy
import random


# create US masks 
def create_under_sampling_mask_uniform(k_space_sz, factor=25, low_fr_pre=16):
    low_fr_pre_half = low_fr_pre / 2
    low_half_lines = k_space_sz * low_fr_pre_half / 100
    half = k_space_sz / 2 - low_half_lines

    mask = np.zeros((k_space_sz, k_space_sz))
    if factor > 0:
        cover = half * factor / 100 * 2
        step = int(half / cover)

        array = range(0, int(half), step)

        mask = np.zeros((k_space_sz, k_space_sz))
        pass_value = 0
        for i in array:
            if (pass_value % 2) == 0:
                i += int(k_space_sz / 2 + low_half_lines - 1)
            mask[i, :] = np.ones((1, k_space_sz))
            pass_value += 1

    if low_fr_pre > 0:
        per4half = int(k_space_sz * 0.01 * low_fr_pre / 2)
        center_index = int(k_space_sz / 2)
        min_index = center_index - per4half
        max_index = center_index + per4half
        sz = max_index - min_index
        mask[min_index:max_index, :] = np.ones((sz, k_space_sz))

    return mask


# create US masks
def create_under_sampling_mask_semi_random(k_space_sz, factor=25, low_fr_pre=16):
    low_fr_pre_half = low_fr_pre / 2
    low_half_lines = k_space_sz * low_fr_pre_half / 100
    half = k_space_sz / 2 - low_half_lines

    mask = np.zeros((k_space_sz, k_space_sz))
    if factor > 0:
        cover = int(half * factor / 100 * 2)
        step = int(half / cover)

        array = random.sample(range(0, int(k_space_sz / 2 - low_half_lines - 1)), cover)

        mask = np.zeros((k_space_sz, k_space_sz))
        pass_value = 0
        for i in array:
            if (pass_value % 2) == 0:
                i += int(k_space_sz / 2 + low_half_lines - 1)
            mask[i, :] = np.ones((1, k_space_sz))
            pass_value += 1

    if low_fr_pre > 0:
        per4half = int(k_space_sz * 0.01 * low_fr_pre / 2)
        center_index = int(k_space_sz / 2)
        min_index = center_index - per4half
        max_index = center_index + per4half
        sz = max_index - min_index
        mask[min_index:max_index, :] = np.ones((sz, k_space_sz))

    return mask


def create_under_sampling_mask_gradient(k_space_sz, factor=25, low_fr_pre=16):
    factor2 = factor * 2
    low_fr_pre_half = low_fr_pre / 2
    low_half_lines = k_space_sz * low_fr_pre_half / 100
    half = k_space_sz / 2 - low_half_lines

    mask = np.zeros((k_space_sz, k_space_sz))
    if factor > 0:
        cover = int(k_space_sz * factor / 100)

        max_index = int(k_space_sz / 2 - low_half_lines)
        x = np.arange(1, cover, 1)
        array = x ** 2 + x + 1

        array = np.array(array / array[-1] * max_index, dtype='int')
        x = np.arange(0, np.shape(array)[0], 1)
        newlist = []
        duplist_i = []
        for i in range(0, len(array)):
            if array[i] not in newlist:
                newlist.append(array[i])
            else:
                duplist_i.append(i)

        replace_elements = list(set(x).symmetric_difference(set(array)))
        for i in range(0, np.shape(duplist_i)[0]):
            array[duplist_i[i]] = replace_elements[i]

        array = array + int(half) + int(low_half_lines * 2) - 1

        pass_value = 0
        for i in array:
            if (pass_value % 2) == 0:
                i = int(k_space_sz - i)
            mask[i, :] = np.ones((1, k_space_sz))
            pass_value += 1

    if low_fr_pre > 0:
        per4half = int(k_space_sz * 0.01 * low_fr_pre / 2)
        center_index = int(k_space_sz / 2)
        min_index = center_index - per4half
        max_index = center_index + per4half
        sz = max_index - min_index
        mask[min_index:max_index, :] = np.ones((sz, k_space_sz))

    return mask


def create_under_sampling_mask(k_space_sz, factor=4, low_fr_pre=4, us_type='uniform'):
    mask = np.zeros((k_space_sz, k_space_sz))
    if us_type == 'uniform':
        mask = create_under_sampling_mask_uniform(k_space_sz, factor=factor, low_fr_pre=low_fr_pre)
    elif us_type == 'semi-random':
        mask = create_under_sampling_mask_semi_random(k_space_sz, factor=factor, low_fr_pre=low_fr_pre)
    elif us_type == 'gradient':
        mask = create_under_sampling_mask_gradient(k_space_sz, factor=factor, low_fr_pre=low_fr_pre)
    else:
        print("Invalid Input of under sampling mask type")
    return mask


# Generator 
def open_data(configs, list_IDs_temp):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, input_channels)
    # Initialization
    
    us_type = ["gradient", "uniform", "semi-random"]      
    if configs.ks_domain == 'True':
        y_out = np.empty((np.shape(list_IDs_temp)[0], configs.img_size, configs.img_size, 3))
        X = np.empty((np.shape(list_IDs_temp)[0], configs.img_size, configs.img_size, configs.input_channels))
    else:
        y_out = np.empty((np.shape(list_IDs_temp)[0], configs.img_size, configs.img_size,1))
        X = np.empty((np.shape(list_IDs_temp)[0], configs.img_size, configs.img_size,1))   
        
    mask_in = np.ones((configs.img_size, configs.img_size))
    if configs.under_sampling == 'True':
        if configs.us_type != 'mix': 
            # underSampling
            mask_in = np.ones((configs.img_size, configs.img_size))
            mask_in = create_under_sampling_mask(configs.img_size, factor=configs.under_sampling_uniform_factor, low_fr_pre=configs.under_sampling_low_fr, 
                                            us_type=configs.us_type)
    # Generate data
    for i, ID in enumerate(list_IDs_temp):
        # # Store sample
        # X[i,] = np.load('data/' + ID + '.npy')
        # # Store class
        # y[i] = self.labels[ID]
        
        # ground true
        name_y = configs.main_data_folder + configs.full_ks + ID + '.npy'
        y_or = np.load(name_y)
        y_i = copy.copy(y_or)
        y_i = np.resize(y_i, (configs.img_size, configs.img_size))
        
        if configs.ks_domain == "True":       
            if configs.under_sampling == 'True':
                if configs.us_type == 'mix':      
                    random_val = int(random.uniform(0, 3))
                    mask_in = create_under_sampling_mask(configs.img_size, factor=configs.under_sampling_uniform_factor, low_fr_pre=configs.under_sampling_low_fr,
                                                            us_type=us_type[random_val])
            # add noise
            if configs.add_noise == 'True':
                [mean, sigma_level] = MRDLin.random_guassian_noise_level()
                artifact = MRDLin.add_guassian_noise_artifacts(ks2d=y_i, mean=mean, sigma_level=sigma_level)
                y_i = artifact[1]
            # add motion artifact
            if configs.add_motion_artifact == 'True':
                motion_artefact_level = MRDLin.random_motion_level()
                artifact = MRDLin.add_motion_artifacts(ks2d=y_i, events=motion_artefact_level[0], ratio=motion_artefact_level[1], phase_direction=0)
                y_i = artifact[1]
                
            X[i, :, :, 0] = y_i.real * mask_in
            X[i, :, :, 1] = y_i.imag * mask_in
            y_out[i, :, :, 0] = y_or.real 
            y_out[i, :, :, 1] = y_or.imag
            y_out[i, :, :, 2] = mask_in
        else:         
            print(configs.output_dir_base)            
            print(configs.ID )
            print(ID)
            print(configs.output_dir_base + "Knet/NextNet/" + ID + '.npy')
            name_x = configs.output_dir_base + "Knet/NextNet/" + ID + '.npy'
            x_i = np.abs(np.load(name_x))
            # print(name_x)
            x_i = np.resize(x_i, (configs.img_size, configs.img_size))
            # normalize
            # min = np.min(x_i)
            # max = np.max(x_i)
            # x_i = (x_i-min)/(max-min)
            X[i, :, :, 0] =  x_i 
            
            # GT
            y_it = abs(MRDLin.recon_corrected_kspace(y_i))
            # normalize
            # min = np.min(y_it)
            # max = np.max(y_it)
            # y_it = (y_it-min)/(max-min)
            y_out[i, :, :, 0]  = y_it # (y_it-min)/(max-min)
            
            
        
        # print("DATA: dddddddddddddddddddddddddddddddd")
        # print(np.shape(X))
        # print(np.shape(y_out))
    
    return X, y_out


def prediction(data, model):
    """
    prediction
    :param data:
    :param model:
    :return unet_image_test:
    """
    # print("ModelGenerator: Prediction ... ")
    # data = np.squeeze(data)
    print('Before pred: ' + str(np.shape(data)))
    unet_image_test_2 = model.predict(data)
    print('After pred: ' + str(np.shape(unet_image_test_2)))
    return unet_image_test_2


def inet_data(configs, partition, old_model, X=None, y=None, fig=False): 
    start = 0
    lines = configs.batch_size
    stop = lines
    
    if not os.path.exists(configs.data_folder_id ):
        os.makedirs(configs.data_folder_id )
    j = 0 
    while j < np.shape(partition)[0]:
        list_IDs_temp = partition[start:stop]
        X, y = open_data(configs, list_IDs_temp)
        
        X1 = np.empty((np.shape(X)[0],np.shape(X)[1],np.shape(X)[2]))
        x_plot = np.empty((np.shape(X)[0],np.shape(X)[1],np.shape(X)[2],2))
        if configs.under_sampling == 'False':
            x_pre = X
        else:
            x_pre = prediction(X, old_model)
        
        mask_new = np.zeros((np.shape(X)[1],np.shape(X)[2], 2))

        for i, ID in enumerate(list_IDs_temp):
            mask_new[:, :, 0] = y[i, :, :, 2]
            mask_new[:, :, 1] = y[i, :, :, 2]
            x = x_pre[i, :, :, 0:2]
            if configs.fKline == "False":
                x_plot[i, :, :, :] = x
            else:
                x_plot[i, :, :, :]= x * (1 - mask_new) + X[i, :, :, :] * mask_new
            
            kspace_or = np.zeros((np.shape(X)[1],np.shape(X)[2]), dtype='complex128')
            kspace_or[:, :].real = np.squeeze(x_plot[i, :, :, 0])
            kspace_or[:, :].imag = np.squeeze(x_plot[i, :, :, 1])
            X1[i, :, :] = abs(MRDLin.recon_corrected_kspace(kspace_or))
            
            name_x = configs.data_folder_id + ID + '.npy'
            np.save(name_x,X1[i, :, :])
            
        if fig:
            plot_data01(y, configs, data_type='test', data_step='GT', list_IDs_temp=list_IDs_temp)
            plot_data01(X, configs, data_type='test', data_step='Input-Knet', list_IDs_temp=list_IDs_temp)
            plot_data01(x_plot, configs, data_type='test', data_step='Output-Knet', list_IDs_temp=list_IDs_temp)
        start = start + lines
        stop = stop + lines
        if stop > np.shape(partition)[0]:
            stop = np.shape(partition)[0]
        j = j + lines


def load_net_partition(path):
    print('Load partition:' + path)
    if path is not None:
        partition = np.load(path + 'partition.npy', allow_pickle=True)
        return partition
    else:
        return 0


def plot_data01(data_or, configs, data_type=None, data_step='total', list_IDs_temp=[]):
    path_pred = configs.output_dir_base + '/' + str(configs.ID) + '/DataPrint/'
        
    if not os.path.exists(path_pred + data_type):
        os.makedirs(path_pred + data_type)

    sz = np.shape(data_or)
    for i_img in range(0, sz[0]):

        kspace_or = np.zeros((sz[1], sz[2]), dtype='complex128')
        kspace_or[:, :].real = np.squeeze(data_or[i_img, :, :, 0])
        kspace_or[:, :].imag = np.squeeze(data_or[i_img, :, :, 1])
        org = abs(MRDLin.recon_corrected_kspace(kspace_or))
        
        fig = plt.figure(figsize=(13, 13))

        # k-space domain
        plt.subplot(1, 2, 1)
        kspace_illu = copy.copy(kspace_or)
        kspace_illu[kspace_illu == 0] = 1
        plt.imshow(np.log(abs(kspace_illu)), cmap='gray')
        plt.title(' ks - ' + data_step, fontweight="bold", fontsize=10)
        plt.xticks([])
        plt.yticks([])

        # image domain
        plt.subplot(1, 2, 2)
        plt.imshow(org, cmap='gray')
        plt.title('img - ' + data_step, fontweight="bold", fontsize=10)
        plt.xticks([])
        plt.yticks([])

        name_fig = path_pred + data_type + '/' + list_IDs_temp[i_img] + '_' + data_step +'.jpg'
        fig.savefig(name_fig)
