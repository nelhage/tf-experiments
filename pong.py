from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import math
import time
import collections

import functools
import operator

import attr

import gym

import tensorflow as tf
import numpy as np

WIDTH  = 210
HEIGHT = 160
PLANES = 1
ACTIONS = 2

DISCOUNT = 0.99

FLAGS = None

class PingPongModel(object):
  @staticmethod
  def weight_variable(shape):
    initial = tf.truncated_normal(
      shape, stddev=0.05)
    return tf.Variable(initial)

  @staticmethod
  def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  @staticmethod
  def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

  def __init__(self):
    with tf.name_scope('Frames'):
      self.prev_frame = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, PLANES], name="ThisFrame")
      self.this_frame = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, PLANES], name="PrevFrame")

    deltas = self.this_frame - self.prev_frame
    deltas = deltas[:,::2,::2]

    with tf.name_scope('Conv'):
      frame = tf.reshape(deltas, (-1, WIDTH//2, HEIGHT//2, PLANES))

      self.W_conv1 = self.weight_variable((4, 4, PLANES, 16))
      self.B_conv1 = self.bias_variable((16,))

      self.h_conv1 = tf.nn.relu(
        tf.nn.conv2d(frame, self.W_conv1, strides=[1, 2, 2, 1], padding='SAME')
        + self.B_conv1)
      self.h_pool1 = self.max_pool_2x2(self.h_conv1)
      tf.summary.histogram('conv1', self.h_conv1)

      self.W_conv2 = self.weight_variable((4, 4, 16, 16))
      self.B_conv2 = self.bias_variable((16,))

      self.h_conv2 = tf.nn.relu(
        tf.nn.conv2d(self.h_conv1, self.W_conv2, strides=[1, 2, 2, 1], padding='SAME')
        + self.B_conv2)
      self.h_pool2 = self.max_pool_2x2(self.h_conv2)
      tf.summary.histogram('conv2', self.h_conv2)

    with tf.name_scope('Hidden'):
      channels = int(functools.reduce(operator.mul, self.h_pool2.get_shape()[1:]))
      self.W_o = self.weight_variable((channels, FLAGS.hidden))
      self.B_o = self.bias_variable((FLAGS.hidden, ))
      inp = tf.reshape(self.h_pool2, (-1, channels))

      z_h = tf.matmul(inp, self.W_o) + self.B_o
      tf.summary.histogram('z_h', z_h)
      a_h = tf.nn.relu(z_h)

    with tf.name_scope('Output'):
      self.W_h = self.weight_variable((FLAGS.hidden, ACTIONS))
      self.B_h = self.bias_variable((ACTIONS, ))

      self.z_o = tf.matmul(a_h, self.W_h) + self.B_h

    self.act_probs = tf.nn.softmax(self.z_o)

    with tf.name_scope('Train'):
      self.reward  = tf.placeholder(tf.float32, [None], name="Reward")
      self.actions = tf.placeholder(tf.float32, [None, ACTIONS], name="SampledActions")

      self.loss = tf.reduce_mean(
        -self.reward *
        tf.nn.softmax_cross_entropy_with_logits(labels=self.actions, logits=self.z_o))
      self.train_step = tf.train.GradientDescentOptimizer(FLAGS.eta).minimize(self.loss)

@attr.s
class Step(object):
  this_frame = attr.ib()
  prev_frame = attr.ib()
  action     = attr.ib()
  reward     = attr.ib()

def build_rewards(steps):
  rewards = np.zeros((len(steps),))
  r = 0
  for i in reversed(range(len(steps))):
    if steps[i].reward != 0:
      r = steps[i].reward
    rewards[i] = r
    r *= DISCOUNT
  rewards -= rewards.mean()
  rewards /= rewards.std()
  return rewards

def build_actions(steps):
  actions = np.zeros((len(steps), ACTIONS))
  actions[np.arange(len(actions)), [s.action for s in steps]] = 1
  return actions

def process_frame(frame):
  return np.expand_dims(np.mean(frame, 2), -1)

def main(_):
  env = gym.make('Pong-v0')
  model = PingPongModel()

  this_frame = process_frame(env.reset())
  prev_frame = np.zeros_like(this_frame)

  steps = []
  reset_time = time.time()
  saver = tf.train.Saver(
    max_to_keep=5, keep_checkpoint_every_n_hours=1)

  session = tf.InteractiveSession()
  if FLAGS.load_model:
    saver.restore(session, FLAGS.load_model)
  else:
    tf.global_variables_initializer().run()

  summary_op = tf.summary.merge_all()
  summary_writer = tf.summary.FileWriter('train/summary', session.graph)

  rounds = 0

  while True:
    if FLAGS.render:
      env.render()

    z, act_probs = session.run([model.z_o, model.act_probs], feed_dict={
      model.this_frame: np.expand_dims(this_frame, 0),
      model.prev_frame: np.expand_dims(prev_frame, 0)})
    if FLAGS.debug:
      print("up={0:.3f} down={1:.3f} z={2}".
            format(act_probs[0][0], act_probs[0][1], z[0]))
    r = np.random.uniform()
    for i, a in enumerate(act_probs[0]):
      if r <= a:
        action = i
        break
      r -= a
    # action = 2 if np.random.uniform() < 0.5 else 3

    next_frame, reward, done, info = env.step(2 + action)

    steps.append(Step(prev_frame=prev_frame,
                      this_frame=this_frame,
                      action=action,
                      reward=reward))

    prev_frame = this_frame
    this_frame = process_frame(next_frame)

    if reward != 0:
      print("reward={0}".format(reward))

    if done:
      if FLAGS.train:
        train_start = time.time()

        rewards = build_rewards(steps)
        actions = build_actions(steps)

        loss, summary, _ = session.run(
          [model.loss, summary_op, model.train_step],
          feed_dict = {
            model.this_frame: [s.this_frame for s in steps],
            model.prev_frame: [s.prev_frame for s in steps],
            model.actions:    actions,
            model.reward:     rewards,
          })
        train_end = time.time()

        print("done round={round} frames={frames} reward={reward} loss={loss} actions={actions}".format(
          frames = len(steps),
          reward = sum([s.reward for s in steps]),
          actions = collections.Counter([s.action for s in steps]),
          loss = loss,
          round = rounds,
        ))
        print("play_time={0:.3f}s train_time={1:.3f}s fps={2:.3f}s".format(
          train_start-reset_time, train_end-train_start, len(steps)/(train_start-reset_time)))

      del steps[:]
      prev_frame = np.zeros_like(this_frame)

      rounds += 1
      if FLAGS.checkpoint > 0 and rounds % FLAGS.checkpoint == 0:
        saver.save(session, FLAGS.checkpoint_path, global_step=rounds)
        summary_writer.add_summary(summary, rounds)

      env.reset()
      reset_time = time.time()

def arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--render', default=False, action='store_true',
                      help='render simulation')
  parser.add_argument('--train', default=True, type=bool,
                      help='Train model')
  parser.add_argument('--hidden', type=int, default=20,
                      help='hidden neurons')
  parser.add_argument('--eta', type=float, default=0.5,
                      help='learning rate')
  parser.add_argument('--checkpoint', type=int, default=0,
                      help='checkpoint every N rounds')
  parser.add_argument('--checkpoint_path', type=str, default='models/pong',
                      help='checkpoint path')
  parser.add_argument('--load_model', type=str, default=None,
                      help='restore model')

  parser.add_argument('--debug', action='store_true',
                      help='debug spew')
  return parser

if __name__ == '__main__':
  parser = arg_parser()
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
else:
  FLAGS = arg_parser().parse_args([])
