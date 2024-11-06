# sourse: https://github.com/zaccharieramzi/fastmri-reproducible-benchmark
from functools import partial
import tensorflow as tf
import tensorflow_addons as tfa
from utils.utilsFun import ssim
import numpy as np


def default_model_compile(model, lr, loss="mssim", mask=None):  #
    """
    :param lr:
    :param loss:
    :param model:
    :type mask: object
    """
    
    
    if loss == "compound_mssim":
        loss_in = compound_l1_mssim_loss
        loss_in.__name__ = "compound_mssim"
    elif loss == "mssim":
        # loss_in = compound_l1_mssim_loss
        loss_in = partial(compound_l1_mssim_loss)
        loss_in.__name__ = "mssim"
    elif loss == "fmssim":
        loss_in = partial(compound_l1_fmssim_loss, mask=mask)
        loss_in.__name__ = "fmssim"        
    else:
        loss_in = "mse" #tf.keras.losses.MeanSquaredError()
       
    print('************************************')
    print(loss_in)
    print('************************************')
     
    model.compile(
        run_eagerly=True,
        optimizer=tfa.optimizers.RectifiedAdam(lr=lr),
        loss=loss_in,
        metrics=["mse", ssim]) # metrics=["mse", mssim, ssim]


def compound_l1_fmssim_loss(y_true, y_pred, mask):
    """
    :param y_true:
    :param y_pred:
    :param alpha:
    :type mask: object
    """
    
    alpha = 0.9999
    
    # y_true_np = K.eval(y_true)
    # y_true_np = y_true.eval(session=tf.Session())
    # y_true_np = y_true.numpy()
    # y_true_np = tf.make_ndarray(y_true)
    y_true_np = tf.make_ndarray(tf.make_tensor_proto(y_true))
    
    # sess = tf.Session()
    # y_true_np = sess.run(y_true)
    
    # sess = tf.compat.v1.Session()
    # # get the y_true and y_pred tensors as 1-D numpy array
    # with sess.as_default():
    #     y_true_np = sess.run(y_true)
    
    new_mask = np.zeros((np.shape(y_true)[0], np.shape(y_true)[1], np.shape(y_true)[2], 3))
    new_mask[:, :, :, 0] = 1-y_true_np[:,:,:,2]
    new_mask[:, :, :, 1] = 1-y_true_np[:,:,:,2]
    new_mask[:, :, :, 2] = 1-y_true_np[:,:,:,2]
    
    mask_tensor = tf.convert_to_tensor(new_mask, dtype="float32")
    # new_mask = 1-new_mask
    # mask_tensor_inverse = tf.convert_to_tensor(new_mask, dtype="float32")
    y_true_new = y_true*mask_tensor
    y_pred_new = y_pred*mask_tensor
    # print("------------------------------- LOSS ------------")
    # print(mask_tensor)
    # print(mask_tensor_inverse)
    # y_true_new = y_true
    # y_pred_new = y_pred*mask_tensor + y_true*mask_tensor_inverse
    mssim = tf.image.ssim_multiscale(y_true_new, y_pred_new, max_val=tf.reduce_max(y_true_new))
    l1 = tf.reduce_mean(tf.abs(y_true_new - y_pred_new))
    loss = alpha * (1 - mssim) + (1 - alpha) * l1
    return loss


def compound_l1_mssim_loss(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :param alpha:
    """
    alpha = 0.9999
    mssim = tf.image.ssim_multiscale(y_true, y_pred, max_val=tf.reduce_max(y_true), filter_size=2)
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred))
    loss = alpha * (1 - mssim) + (1 - alpha) * l1
    # loss = (1 - mssim)
    
    return loss

