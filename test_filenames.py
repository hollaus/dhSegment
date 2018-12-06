import os
from glob import glob
import re

input_data = 'binarization/training_ms_tex/msi/images'
use_ms = True

input_image_filenames = glob(os.path.join(input_data, '*.jpg'),
                            recursive=False) + \
                        glob(os.path.join(input_data, '*.png'),
                            recursive=False)

input_image_filenames = [re.sub(r'_\d\.png', '', f) for f in input_image_filenames]

input_image_filenames = set(input_image_filenames)
print(input_image_filenames)

print('Found {} images'.format(len(input_image_filenames)))