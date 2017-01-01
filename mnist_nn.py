# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  y_ = tf.placeholder(tf.float32, [None, 10])

  W_h = tf.Variable(tf.random_normal([784, FLAGS.n], mean=1.0/784))
  b_h = tf.Variable(tf.random_normal([FLAGS.n]))
  z_h = tf.matmul(x, W_h) + b_h
  a_h = tf.sigmoid(z_h)

  W_o = tf.Variable(tf.random_normal([FLAGS.n, 10], mean=1.0/FLAGS.n))
  b_o = tf.Variable(tf.random_normal([10]))
  z_o = tf.matmul(a_h, W_o) + b_o
  y = tf.sigmoid(z_o)


  if FLAGS.loss == 'softmax':
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(z_o, y_))
  elif FLAGS.loss == 'cross-entropy':
    loss = tf.reduce_mean(
      -tf.reduce_sum(y_ * tf.log(y) + (1-y_) * tf.log(1-y), 1))
  elif FLAGS.loss == 'quadratic':
    loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.squared_difference(y, y_), 1))
  else:
    raise ArgumentError("bad loss function")

  if FLAGS.regularize > 0:
    weights = tf.concat(0, [tf.reshape(W_h, [-1]), tf.reshape(W_o, [-1])])
    loss = loss + FLAGS.regularize * tf.reduce_mean(weights*weights)

  train_step = tf.train.GradientDescentOptimizer(FLAGS.eta).minimize(loss)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for i in range(FLAGS.epochs * int(mnist.train.num_examples/FLAGS.batch)):
    batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i % 100 == 0:
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      print("i={0} cost={1:0.4f} acc={2:0.4f}".format(
        i,
        sess.run(loss, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}),
        sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels})))

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='data/MNIST',
                      help='Directory for storing input data')
  parser.add_argument('-n', type=int, default=30,
                      help='number of hidden neurons')
  parser.add_argument('--eta', type=float, default=0.5,
                      help='learning rate')
  parser.add_argument('--batch', type=int, default=100,
                      help='batch size')
  parser.add_argument('--epochs', type=int, default=30,
                      help='epochs')
  parser.add_argument('--loss', type=str, default='cross-entropy')
  parser.add_argument('--regularize', type=float, default=0.0)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
