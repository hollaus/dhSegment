from tensorflow.python import pywrap_tensorflow
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import numpy as np
import tensorflow as tf
import os
import mask_unused_gpus

if __name__ == '__main__':


    input_path = 'pretrained_models/model_try/mod_2_resnet_v1_50.ckpt'
    # input_path = 'pretrained_models/resnet_v1_50.ckpt'

    reader = pywrap_tensorflow.NewCheckpointReader(input_path)

    mask_unused_gpus.mask_unused_gpus(2)
    # os.environ['CUDA_VISIBLE_DEVICES'] = "3"

    # print_tensors_in_checkpoint_file(input_path, all_tensors=False, tensor_name='')
    # print_tensors_in_checkpoint_file(input_path, all_tensors=False, tensor_name='resnet_v1_50/mean_rgb')
    # print_tensors_in_checkpoint_file(input_path, all_tensors=False, tensor_name='resnet_v1_50/conv1/weights/Momentum')
    print_tensors_in_checkpoint_file(input_path, '', all_tensors=False)