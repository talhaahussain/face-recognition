import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import json
import cv2
import numpy as np
from matplotlib import pyplot as plt


def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

tf.config.list_physical_devices('GPU')

images = tf.data.Dataset.list_files('data/images/*.jpg', shuffle=False )
images = images.map(load_image)

print(images.as_numpy_iterator().next())
