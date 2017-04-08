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
import scipy.signal

WIDTH  = 210
HEIGHT = 160
PLANES = 1
ACTIONS = 2

DISCOUNT = 0.99

FLAGS = None

class PingPongModel(object):
  @staticmethod
  def weight_variable(shape, name=None):
    initial = tf.truncated_normal(
      shape, stddev=0.05)
    return tf.Variable(initial, name)

  @staticmethod
  def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name)

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

      self.W_conv1 = self.weight_variable((4, 4, PLANES, 16), 'w_1')
      self.B_conv1 = self.bias_variable((16,), 'b_1')

      self.h_conv1 = tf.nn.relu(
        tf.nn.conv2d(frame, self.W_conv1, strides=[1, 2, 2, 1], padding='SAME')
        + self.B_conv1)
      self.h_pool1 = self.max_pool_2x2(self.h_conv1)
      tf.summary.histogram('conv1', self.h_conv1)

      self.W_conv2 = self.weight_variable((4, 4, 16, 16), 'w_1')
      self.B_conv2 = self.bias_variable((16,), 'b_2')

      self.h_conv2 = tf.nn.relu(
        tf.nn.conv2d(self.h_conv1, self.W_conv2, strides=[1, 2, 2, 1], padding='SAME')
        + self.B_conv2)
      self.h_pool2 = self.max_pool_2x2(self.h_conv2)
      tf.summary.histogram('conv2', self.h_conv2)

    with tf.name_scope('Hidden'):
      channels = int(functools.reduce(operator.mul, self.h_pool2.get_shape()[1:]))
      self.W_h = self.weight_variable((channels, FLAGS.hidden), 'w')
      self.B_h = self.bias_variable((FLAGS.hidden, ), 'b')
      inp = tf.reshape(self.h_pool2, (-1, channels))

      z_h = tf.matmul(inp, self.W_h) + self.B_h
      tf.summary.histogram('z_h', z_h)
      a_h = tf.nn.relu(z_h)

    with tf.name_scope('Output'):
      self.W_o = self.weight_variable((FLAGS.hidden, ACTIONS), 'w_l')
      self.B_o = self.bias_variable((ACTIONS, ), 'b_l')

      self.z_o = tf.matmul(a_h, self.W_o) + self.B_o

      self.W_v = self.weight_variable((FLAGS.hidden, 1), 'w_v')
      self.B_v = self.bias_variable((1,), 'b_v')

    self.logits = self.z_o
    tf.summary.histogram('logits', self.logits)
    self.act_probs = tf.nn.softmax(self.logits)
    self.vp = tf.reshape(tf.tanh(tf.matmul(a_h, self.W_v) + self.B_v), (-1,))

    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
      tf.summary.scalar('norm/' + var.name, tf.norm(var))

  def add_train_ops(self):
    with tf.name_scope('Train'):
      self.adv  = tf.placeholder(tf.float32, [None], name="Advantage")
      tf.summary.histogram('advantage', self.adv)
      self.rewards = tf.placeholder(tf.float32, [None], name="Reward")
      tf.summary.histogram('weighted_reward', self.rewards)
      self.actions = tf.placeholder(tf.float32, [None, ACTIONS], name="SampledActions")

      self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.actions, logits=self.z_o)
      self.pg_loss = tf.reduce_mean(self.adv * self.cross_entropy)
      tf.summary.scalar('pg_loss', self.pg_loss)
      self.v_loss = 0.5 * tf.reduce_mean(tf.square(self.vp - self.rewards))
      tf.summary.scalar('value_loss', self.v_loss)
      tf.summary.histogram('value_err', (self.vp - self.rewards))
      self.entropy = -tf.reduce_mean(
        tf.reduce_sum(self.act_probs * tf.nn.log_softmax(self.logits), axis=1))
      tf.summary.scalar('entropy', self.entropy)

      self.loss = (
        FLAGS.pg_weight * self.pg_loss +
        FLAGS.v_weight * self.v_loss -
        FLAGS.entropy_weight * self.entropy)

      self.train_step = tf.train.AdamOptimizer(FLAGS.eta).minimize(self.loss)

@attr.s
class Rollout(object):
  frames    = attr.ib(default=attr.Factory(list))
  actions   = attr.ib(default=attr.Factory(list))
  rewards   = attr.ib(default=attr.Factory(list))
  vp        = attr.ib(default=attr.Factory(list))

  discounted = attr.ib(default=None)

  def clear(self):
    del self.frames[:]
    del self.actions[:]
    del self.rewards[:]
    del self.vp[:]

def build_rewards(rollout):
  discounted = np.zeros((len(rollout.actions),))
  r = 0
  for i,rw in reversed(list(enumerate(rollout.rewards))):
    if rw != 0:
      r = rw
    discounted[i] = r
    r *= DISCOUNT

  rollout.discounted = discounted
  return rollout.discounted

def build_advantage(rollout):
  return rollout.discounted - rollout.vp

def build_actions(rollout):
  actions = np.zeros((len(rollout.actions), ACTIONS))
  actions[np.arange(len(actions)), rollout.actions] = 1
  return actions

def process_frame(frame):
  return np.expand_dims(np.mean(frame, 2), -1)

def main(_):
  env = gym.make('Pong-v0')
  model = PingPongModel()
  if FLAGS.train:
    model.add_train_ops()

  this_frame = process_frame(env.reset())
  prev_frame = np.zeros_like(this_frame)

  rollout = Rollout(
    frames=[prev_frame],
  )
  reset_time = time.time()
  saver = tf.train.Saver(
    max_to_keep=5, keep_checkpoint_every_n_hours=1)

  session = tf.InteractiveSession()
  if FLAGS.load_model:
    saver.restore(session, FLAGS.load_model)
  else:
    tf.global_variables_initializer().run()

  if FLAGS.logdir:
    try:
      os.makedirs(os.path.dirname(FLAGS.logdir))
    except FileExistsError:
      pass
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.logdir, session.graph)
  else:
    summary_op = summary_writer = None
  write_summaries = summary_writer and FLAGS.summary_interval
  write_checkpoints = FLAGS.logdir and FLAGS.checkpoint

  rounds = 0

  avgreward = 0

  while True:
    if FLAGS.render:
      env.render()

    act_probs, vp = session.run(
      [model.act_probs, model.vp],
      feed_dict=
      {
        model.this_frame: np.expand_dims(this_frame, 0),
        model.prev_frame: np.expand_dims(prev_frame, 0)
      })
    r = np.random.uniform()
    for i, a in enumerate(act_probs[0]):
      if r <= a:
        action = i
        break
      r -= a
    # action = 2 if np.random.uniform() < 0.5 else 3

    next_frame, reward, done, info = env.step(2 + action)

    prev_frame = this_frame
    this_frame = process_frame(next_frame)

    rollout.frames.append(this_frame)
    rollout.actions.append(action)
    rollout.rewards.append(reward)
    rollout.vp.append(vp[0])

    if done:
      if FLAGS.train:
        train_start = time.time()

        rewards = build_rewards(rollout)
        adv     = build_advantage(rollout)
        actions = build_actions(rollout)

        ops = {
          'pg_loss': model.pg_loss,
          'v_loss': model.v_loss,
          'train': model.train_step,
        }
        if summary_op is not None:
          ops['summary'] = summary_op

        out = session.run(
          ops,
          feed_dict = {
            model.this_frame: rollout.frames[1:],
            model.prev_frame: rollout.frames[:-1],
            model.actions:    actions,
            model.rewards:    rewards,
            model.adv:        adv,
          })
        train_end = time.time()

        avgreward = 0.9 * avgreward + 0.1 * sum(rollout.rewards)
        print("done round={round} frames={frames} reward={reward} expreward={avgreward:.1f} pg_loss={pg_loss} v_loss={v_loss} actions={actions}".format(
          frames = len(rollout.actions),
          reward = sum(rollout.rewards),
          avgreward = avgreward,
          actions = collections.Counter(rollout.actions),
          pg_loss = out['pg_loss'],
          v_loss = out['v_loss'],
          round = rounds,
        ))
        fps = len(rollout.actions)/(train_start-reset_time)
        print("play_time={0:.3f}s train_time={1:.3f}s fps={2:.3f}s".format(
          train_start-reset_time, train_end-train_start, fps))

      if write_summaries and rounds % FLAGS.summary_interval == 0:
        summary = tf.Summary()
        summary.value.add(tag='env/frames', simple_value=float(len(rollout.actions)))
        summary.value.add(tag='env/fps', simple_value=fps)
        summary.value.add(tag='env/reward', simple_value=sum(rollout.rewards))
        summary_writer.add_summary(summary, rounds)
        summary_writer.add_summary(out['summary'], rounds)

      prev_frame = np.zeros_like(this_frame)
      rollout.clear()
      rollout.frames.append(prev_frame)

      rounds += 1
      if write_checkpoints and rounds % FLAGS.checkpoint == 0:
        saver.save(session, os.path.join(FLAGS.logdir, 'pong'), global_step=rounds)


      env.reset()
      reset_time = time.time()

def arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--render', default=False, action='store_true',
                      help='render simulation')
  parser.add_argument('--train', default=True, action='store_true',
                      help='Train model')
  parser.add_argument('--no-train', action='store_false', dest='train',
                      help="Don't train")
  parser.add_argument('--hidden', type=int, default=256,
                      help='hidden neurons')
  parser.add_argument('--eta', type=float, default=1e-4,
                      help='learning rate')
  parser.add_argument('--checkpoint', type=int, default=0,
                      help='checkpoint every N rounds')
  parser.add_argument('--summary_interval', type=int, default=5,
                      help='write summaries every N rounds')
  parser.add_argument('--logdir', type=str, default=None,
                      help='log path')
  parser.add_argument('--load_model', type=str, default=None,
                      help='restore model')

  parser.add_argument('--debug', action='store_true',
                      help='debug spew')

  parser.add_argument('--pg_weight', type=float, default=1.0)
  parser.add_argument('--v_weight', type=float, default=0.5)
  parser.add_argument('--entropy_weight', type=float, default=0.01)
  return parser

if __name__ == '__main__':
  parser = arg_parser()
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
else:
  FLAGS = arg_parser().parse_args([])
