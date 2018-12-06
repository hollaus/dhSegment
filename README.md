# dhSegment

[![Documentation Status](https://readthedocs.org/projects/dhsegment/badge/?version=latest)](https://dhsegment.readthedocs.io/en/latest/?badge=latest)

**dhSegment** is a tool for Historical Document Processing. Its generic approach allows to segment regions and
extract content from different type of documents. See 
[some examples here](https://dhsegment.readthedocs.io/en/latest/intro.html#use-cases).

The complete description of the system can be found in the corresponding [paper](https://arxiv.org/abs/1804.10371).

It was created by [Benoit Seguin](https://twitter.com/Seguin_Be) and Sofia Ares Oliveira at DHLAB, EPFL.

## Installation and usage
The [installation procedure](https://dhsegment.readthedocs.io/en/latest/start/install.html) 
and examples of usage can be found in the documentation (see section below).

## Demo
Have a try at the [demo](https://dhsegment.readthedocs.io/en/latest/start/demo.html) to train (optional) and apply dhSegment in page extraction using the `demo.py` script.

## Documentation

*Under construction*

The documentation is available on [readthedocs](https://dhsegment.readthedocs.io/).

##
If you are using this code for your research, you can cite the corresponding paper as :
```
@inproceedings{oliveiraseguinkaplan2018dhsegment,
  title={dhSegment: A generic deep-learning approach for document segmentation},
  author={Ares Oliveira, Sofia and Seguin, Benoit and Kaplan, Frederic},
  booktitle={Frontiers in Handwriting Recognition (ICFHR), 2018 16th International Conference on},
  pages={7--12},
  year={2018},
  organization={IEEE}
}
```

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
