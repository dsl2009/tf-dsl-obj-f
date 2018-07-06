
import tensorflow as tf
from tensorflow.contrib import slim
x = tf.placeholder(dtype=tf.float32,shape=[1,244,244,3])
x = slim.conv2d(x,60,1,trainable=False)
print(slim.get_variables_to_restore())
print(slim.get_variables())
print(slim.get_trainable_variables())