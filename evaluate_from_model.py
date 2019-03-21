import cv2 as cv
import os
import re
from glob import glob

import json
from pprint import pprint
import tensorflow as tf
import numpy as np
import mask_unused_gpus
import sys
from sacred import Experiment

from tqdm import tqdm

from dh_segment.inference import LoadedModel

def evaluate_fnc(model_dir):

    model_name = os.path.basename(os.path.normpath(model_dir))

    model_json_file = os.path.join(model_dir, 'config.json')
    if not os.path.isfile(model_json_file):
        print('Sorry, I could not load the config.json file')
        return

    with open(model_json_file) as f:
        data = json.load(f)
        train_data_dir = data["train_data"]
        print('train data: ' + train_data_dir)
        # simply generate the test data dir, by replacing the last occurence of /train with /test
        test_data_dir = re.sub('(/train)(?!.*/train)', '/train_and_val', train_data_dir)
        # test_data_dir = re.sub('(/train)(?!.*/train)', '/test', train_data_dir)
        print('saving to : ' + test_data_dir)
        # test_data_dir = re.sub('(/train)(?!.*/train)', '/train_and_val', train_data_dir)
        test_data_dir = os.path.join(test_data_dir, 'images')
        print('test_data_dir: ' + test_data_dir)
        if not os.path.isdir(test_data_dir):
            print('Sorry, the test folder is not existing: ' + test_data_dir)
            return

        # simply generate the results dir, by using a 'result' child of the parent folder of the train_data_dir
        result_parent_dir = os.path.join(os.path.dirname(train_data_dir), 'results_train_and_val')
        output_dir = os.path.join(result_parent_dir, model_name)


        # TODO: read this from json
        use_ms = True
        # TODO: is this necessary?
        model_dir = os.path.join(model_dir, 'export')

        if use_ms:
            input_image_filenames = glob(os.path.join(test_data_dir, '*.jpg'),
                                         recursive=False) + \
                glob(os.path.join(test_data_dir, '*.png'),
                     recursive=False)
            input_image_filenames = [
                re.sub(r'_\d\d?', '', f) for f in input_image_filenames]
            # input_image_filenames = [re.sub(r'_\d', '', f) for f in input_image_filenames]
            input_files = set(input_image_filenames)
            print('Found {} MSI images'.format(len(input_files)))

        else:
            input_image_filenames = glob(os.path.join(test_data_dir, '*.jpg'),
                                         recursive=False) + \
                glob(os.path.join(test_data_dir, '*.png'),
                     recursive=False)
            input_files = input_image_filenames
            print('Found {} images'.format(len(input_files)))

        mask_unused_gpus.mask_unused_gpus(2)

        os.makedirs(output_dir, exist_ok=True)

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

            # print('number of network parameters: ' + str(total_parameters))

    #         Iterate over the images:
            for filename in tqdm(input_files, desc='Processed files'):

                prediction_outputs = m.predict(filename)
                probs = prediction_outputs['probs'][0]

                original_shape = prediction_outputs['original_shape']

                if use_ms:
                    filename = re.sub(r'.png', '_2.png', filename)

                img = cv.imread(filename, cv.IMREAD_COLOR)

                # just use the fg class:
                p = probs[:, :, 1] * 255
                bin_upscaled = cv.resize(p.astype(np.uint8, copy=False), tuple(
                    original_shape[::-1]), interpolation=cv.INTER_CUBIC)

                b = (bin_upscaled > 127) * 255
                b = np.array(b, dtype=np.uint8)

                b_rgb = np.zeros((bin_upscaled.shape[0], bin_upscaled.shape[1], 3))
                b_rgb[:, :, 1] = b

                # do we have a second fg class?
                if (probs.shape[2] == 3):
                    p_fg2 = probs[:, :, 2] * 255
                    bin_upscaled_fg2 = cv.resize(p_fg2.astype(np.uint8, copy=False), tuple(
                        original_shape[::-1]), interpolation=cv.INTER_CUBIC)
                    b_fg2 = (bin_upscaled_fg2 > 127) * 255
                    b_fg2 = np.array(b_fg2, dtype=np.uint8)
                    b_rgb[:,:,2] = b_fg2

                filename = re.sub(r'_\d.png', '.png', os.path.basename(filename))
                full_filename = os.path.join(output_dir, filename)

                cv.imwrite(full_filename, b_rgb)



if __name__ == '__main__':

    # Determine the model name:
    if len(sys.argv) < 2:
        print('Please provide a model dir as argument')
        sys.exit()
    model_dir = str(sys.argv[1])
    evaluate_fnc(model_dir)

