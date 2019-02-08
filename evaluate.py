import cv2 as cv
import os
import re
from glob import glob

import tensorflow as tf
import numpy as np
import mask_unused_gpus
import sys

from tqdm import tqdm

from dh_segment.inference import LoadedModel

if __name__ == '__main__':

    
    # Determine the model name:
    if len(sys.argv) < 2:
        print('Please provide a model name as argument')
        sys.exit()
    model_name = str(sys.argv[1])

    # Is the model trained on MS images`:
    # print(sys.argv[2])
    if (len(sys.argv) == 3):
        use_ms = eval(sys.argv[2])
    else:
        use_ms = True

    mask_unused_gpus.mask_unused_gpus(2)

    use_ms=False

    if use_ms:
        # input_path = 'binarization/msbin/gray/test/images'
        input_path = 'binarization/mstex/data/test/images'
        model_dir = model_name + '/export/'
        # input_path = 'binarization/test_ms_tex/msi/'
        # model_dir = 'binarization/model_ms_tex/msi/' + model_name + '/export/'
    else:
        input_path = 'binarization/test_ms_tex/single_channel/'
        model_dir = model_name

    output_dir = 'binarization/ms_tex_binar_test/' + model_name + '/'

    if use_ms:
        input_image_filenames = glob(os.path.join(input_path, '*.jpg'),
                        recursive=False) + \
                    glob(os.path.join(input_path, '*.png'),
                        recursive=False)
        input_image_filenames = [re.sub(r'_\d\d?', '', f) for f in input_image_filenames]
        # input_image_filenames = [re.sub(r'_\d', '', f) for f in input_image_filenames]
        input_files = set(input_image_filenames)

    else:
        input_image_filenames = glob(os.path.join(input_path, '*.jpg'),
                                recursive=False) + \
                            glob(os.path.join(input_path, '*.png'),
                                recursive=False)
        input_files = input_image_filenames
    
    print('Found {} images'.format(len(input_image_filenames)))

    os.makedirs(output_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"

    with tf.Session():  # Start a tensorflow session
        # Load the model
        m = LoadedModel(model_dir, predict_mode='filename')

        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters

        print('number of network parameters: ' + str(total_parameters))

#         Iterate over the images:
        for filename in tqdm(input_files, desc='Processed files'):
            # For each image, predict each pixel's label
            
            prediction_outputs = m.predict(filename)
            probs = prediction_outputs['probs'][0]
            
            original_shape = prediction_outputs['original_shape']
            # probs = probs[:, :, 1]  # Take only class '1' (class 0 is the background, class 1 is the page)
            # probs = probs / np.max(probs)  # Normalize to be in [0, 1]
            # img = cv.imread(filename, 0)
            # cv.imwrite(output_dir + os.path.basename(filename), img)
            
            if use_ms:
                filename = re.sub(r'.png', '_2.png', filename)

            img = cv.imread(filename, cv.IMREAD_COLOR)

            p = probs[:, :, 1] * 255

            # Upscale to have full resolution image (cv2 uses (w,h) and not (h,w) for giving shapes)
            bin_upscaled = cv.resize(p.astype(np.uint8, copy=False), tuple(
                original_shape[::-1]), interpolation=cv.INTER_NEAREST)
            # pred = probs[:, :, 2] * 255

            # Upscale to have full resolution image (cv2 uses (w,h) and not (h,w) for giving shapes)
            # bin_upscaled_red = cv.resize(pred.astype(np.uint8, copy=False), tuple(
            #     original_shape[::-1]), interpolation=cv.INTER_NEAREST)                

            img[:, :, 1] = bin_upscaled
            # img[:, :, 2] = bin_upscaled_red

            pseudo_filename = re.sub(r'.png', 'pseudo.png', filename)
            filename = re.sub(r'_\d.png', '.png', filename)

            b = (bin_upscaled > 128) * 255
            b = np.array(b, dtype=np.uint8)

            # b_red = (bin_upscaled_red > 128) * 255
            # b_red = np.array(b_red, dtype=np.uint8)
            
            b_multiclass = np.zeros(img.shape)
            b_multiclass[:,:,1] = b
            # b_multiclass[:,:,2] = b_red

            cv.imwrite(output_dir + os.path.basename(pseudo_filename), img)
            cv.imwrite(output_dir + os.path.basename(filename), b_multiclass)




# ===============================================================================================================
# \\moe\ResultTray\holl\ms\binarization\ms_tex_binar_test\ms_resnet_large\
# F: 0.56551 R: 0.47877 P: 0.83511
# F: 0.76683     0.53423     0.49183     0.55508     0.74797     0.64981     0.84789     0.63763     0.41311    0.010691

# ===============================================================================================================
# \\moe\ResultTray\holl\ms\binarization\ms_tex_binar_test\rn\ (single channel!)
# F: 0.62753 R: 0.61162 P: 0.72608
# F: 0.76141     0.51172     0.59566     0.60379     0.78367     0.74422      0.8069     0.65014     0.63192     0.18591

