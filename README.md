# msSegment

[![Documentation Status](https://readthedocs.org/projects/dhsegment/badge/?version=latest)](https://dhsegment.readthedocs.io/en/latest/?badge=latest)

**msSegment** is a program for the binarization of multispectral document images. The project is a fork of [dhSegment](https://github.com/dhlab-epfl/dhSegment), which is developed by [Benoit Seguin](https://github.com/SeguinBe) and [Sofia Ares Oliveira](https://github.com/solivr) at DHLAB, EPFL.


## Simple cmd's for Fabian

`watch -n1 nvidia-smi`
- export environment: `pip freeze > requirements.txt`
- read environment: `pip install -r requirements.txt`
- show the GPU workload:
`watch -n 0.5 nvidia-smi`
- show available versions for tensorflow:
`pip install tensorflow==`
- install version 1.12:
`pip install tensorflow==1.12.0`
- check if gpu is supported by tensorflow version:
`python`
`import tensorflow as tf`
`sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))`
