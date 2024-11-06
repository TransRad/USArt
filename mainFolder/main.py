import sys
from utils.utilsFun import open_json
from mainFolder.dataGeneratorUtils import load_net_partition, inet_data, open_data
from mainFolder.modelGenerator import sub_tuning
import numpy as np


def run(configs, old_model=None, prepare_inverse_domain_net="False"):
    path = configs.data_folder_id
    print(path)
    print('Load partition')
    partition = load_net_partition(path)
    partition = partition.item()    
    print(partition)
    
    partition['train'] = partition['train'][0:5000]#[0:5000]
    
    x_train, y_train = open_data(configs, partition['train'])
    print('x_train: ')
    print(np.shape(x_train))
    print('y_train: ')
    print(np.shape(y_train))
    x_valid, y_valid = open_data(configs, partition['valid'])

    print('Start run - loss: ' + ' -  Model: ' + str(configs.model) + ' -  ks_domain: ' + str(configs.ks_domain) + ' -  data_folder_id: ' + str(configs.data_folder_id))
    model = []
    if configs.under_sampling == 'True' or configs.ks_domain == "False":
        model = sub_tuning(range_in=[], i=[], configs=configs, x_train=x_train,
                        y_train=y_train, x_valid=x_valid, y_valid=y_valid, tuning_par=False,
                        old_model=old_model)

    print("********* Prepare the input data for Inet")
    path_i = "./results/" + configs.ID + "/NextNet/"
    configs.data_folder_id = path_i
    print(configs.ks_domain)
    if configs.ks_domain == "True":
        inet_data(configs,partition['test'], old_model=model, fig=False)
        inet_data(configs,partition['train'], old_model=model, X=x_train, y=y_train, fig=False)
        inet_data(configs,partition['valid'], old_model=model, X=x_valid, y=y_valid, fig=False)

    return model

if __name__ == "__main__":
    args = sys.argv[1:][0]
    print('input Json path: ' + args)
    configs = open_json(args)
    configs.keys()
    run(configs)
