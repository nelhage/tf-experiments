from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import gym
import math

import collections

import tensorflow as tf
import numpy as np

WIDTH  = 210
HEIGHT = 160
PLANES = 3
ACTIONS = 6

FLAGS = None

class PingPongModel(object):
  def __init__(self):
    with tf.name_scope('Frames'):
      self.prev_frame = tf.placeholder(tf.float32, [None, WIDTH * HEIGHT * PLANES], name="ThisFrame")
      self.this_frame = tf.placeholder(tf.float32, [None, WIDTH * HEIGHT * PLANES], name="PrevFrame")

    deltas = self.this_frame - self.prev_frame

    self.w_h = tf.Variable(tf.random_normal([WIDTH * HEIGHT * PLANES, FLAGS.hidden],
                                           mean=1/math.sqrt(WIDTH * HEIGHT * PLANES)))
    self.b_h = tf.Variable(tf.random_normal([FLAGS.hidden]))

    z_h = tf.matmul(deltas, self.w_h) + self.b_h
    a_h = tf.sigmoid(z_h)

    self.w_o = tf.Variable(tf.random_normal([FLAGS.hidden, ACTIONS],
                                            mean=1.0/math.sqrt(float(FLAGS.hidden))))
    self.b_o = tf.Variable(tf.random_normal([ACTIONS]))

    z_o = tf.matmul(a_h, self.w_o) + self.b_o

    self.act_probs = tf.nn.softmax(z_o)

    self.reward  = tf.placeholder(tf.float32, [], name="Reward")
    self.actions = tf.placeholder(tf.float32, [None, ACTIONS], name="SampledActions")

    self.loss = tf.reduce_mean(self.reward * tf.nn.softmax_cross_entropy_with_logits
                               (labels=self.actions, logits=z_o))
    self.train_step = tf.train.GradientDescentOptimizer(FLAGS.eta).minimize(self.loss)


def main(_):
  env = gym.make('Pong-v0')
  model = PingPongModel()

  session = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  prev_frame = env.reset().reshape(-1)
  this_frame = prev_frame

  frames  = [prev_frame]
  actions = []

  while True:
    if FLAGS.render:
      env.render()

    act_probs = session.run(model.act_probs, feed_dict={
      model.this_frame: np.expand_dims(this_frame, 0),
      model.prev_frame: np.expand_dims(prev_frame, 0)})
    r = np.random.uniform()
    for i, a in enumerate(act_probs[0]):
      if r <= a:
        action = i
        break
      r -= a
    actions.append(action)

    prev_frame = this_frame
    this_frame, reward, done, info = env.step(action)
    this_frame = this_frame.reshape(-1)

    if reward != 0:
      print("reward={0} frames={1} actions={2}".format
            (reward, len(actions), collections.Counter(actions)))

      actions = actions[1:]
      actionv = np.zeros((len(actions), 6))
      actionv[np.arange(len(actions)), actions] = 1

      loss, _ = session.run(
        [model.loss, model.train_step],
        feed_dict = {
          model.this_frame: frames[1:],
          model.prev_frame: frames[:-1],
          model.actions:    actionv,
          model.reward:     reward,
        })
      print("reward={0} loss={1}".format(reward, loss))

      frames = []
      actions = []

    frames.append(this_frame)

    if done:
      print("done total={0}".format(reward))
      env.reset()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--render', default=False, action='store_true',
                      help='render simulation')
  parser.add_argument('--hidden', type=int, default=100,
                      help='hidden neurons')
  parser.add_argument('--eta', type=float, default=0.5,
                      help='learning rate')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
