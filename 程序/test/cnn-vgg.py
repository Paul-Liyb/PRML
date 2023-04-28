import warnings
warnings.filterwarnings('ignore', message='Unused import statement*')

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16model =VGG16(weights='imagenet')

tf.