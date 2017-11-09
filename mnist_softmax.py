# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
# from tensorflow.examples.tutorials.mnist import input_data

import input_data

import tensorflow as tf

import os

from mnist_demo import * 

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', 'data/', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

sess = tf.InteractiveSession()

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Train
tf.initialize_all_variables().run()
# print(tf.argmax(W,1).eval())
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  # print(tf.argmax(W,1).eval())
  train_step.run({x: batch_xs, y_: batch_ys})

  # break

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy=tf.cast(tf.argmax(y, 1),tf.float32)
# accuracy=y
# print((mnist.test.labels))

dir_name="test_num"
files = os.listdir(dir_name)
cnt=len(files)
for i in range(cnt):
  files[i]=dir_name+"/"+files[i]
  # print(files[i])
  test_images1,test_labels1=GetImage([files[i]])
  # print (tf.cast(correct_prediction, tf.float32).eval)
  # print(shape(test_images1))
  mnist.test = input_data.DataSet(test_images1, test_labels1, dtype=tf.float32)
  res=accuracy.eval({x: mnist.test.images, y_: mnist.test.labels})

  # print(shape(mnist.test.images))
  # print (tf.argmax(y, 1))
  # print(y.eval())
  print("output:",int(res[0]))
  print("\n")
  # if(res==1):
  #   print("correct!\n")
  # else:
  #   print("wrong!\n")

  # print("input:",files[i].strip().split('/')[1][0])

