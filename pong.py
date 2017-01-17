from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import math
import time
import collections

import attr

import gym

import tensorflow as tf
import numpy as np

WIDTH  = 210
HEIGHT = 160
PLANES = 1
ACTIONS = 2

FLAGS = None

class PingPongModel(object):
  def __init__(self):
    with tf.name_scope('Frames'):
      self.prev_frame = tf.placeholder(tf.float32, [None, WIDTH * HEIGHT * PLANES], name="ThisFrame")
      self.this_frame = tf.placeholder(tf.float32, [None, WIDTH * HEIGHT * PLANES], name="PrevFrame")

    deltas = self.this_frame - self.prev_frame

    with tf.name_scope('Hidden'):
      self.w_h = tf.Variable(tf.random_normal([WIDTH * HEIGHT * PLANES, FLAGS.hidden],
                                              mean=1/math.sqrt(WIDTH * HEIGHT * PLANES)),
                             name='Weights')
      self.b_h = tf.Variable(tf.random_normal([FLAGS.hidden]),
                             name='Biases')

      z_h = tf.matmul(deltas, self.w_h) + self.b_h
      a_h = tf.sigmoid(z_h)

    with tf.name_scope('Output'):
      self.w_o = tf.Variable(tf.random_normal([FLAGS.hidden, ACTIONS],
                                              mean=1.0/math.sqrt(float(FLAGS.hidden))),
                             name='Weights')
      self.b_o = tf.Variable(tf.random_normal([ACTIONS]),
                             name='Biases')

      z_o = tf.matmul(a_h, self.w_o) + self.b_o

    self.act_probs = tf.nn.softmax(z_o)

    with tf.name_scope('Train'):
      self.reward  = tf.placeholder(tf.float32, [None], name="Reward")
      self.actions = tf.placeholder(tf.float32, [None, ACTIONS], name="SampledActions")

      self.loss = tf.reduce_mean(
        -self.reward *
        tf.nn.softmax_cross_entropy_with_logits(labels=self.actions, logits=z_o))
      self.train_step = tf.train.GradientDescentOptimizer(FLAGS.eta).minimize(self.loss)

  def save_variables(self):
    [self.w_h, self.b_h, self.w_o, self.b_o]

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
  return rewards

def build_actions(steps):
  actions = np.zeros((len(steps), ACTIONS))
  actions[np.arange(len(actions)), [s.action for s in steps]] = 1
  return actions

def process_frame(frame):
  return np.mean(frame, 2).reshape(-1)

def main(_):
  env = gym.make('Pong-v0')
  model = PingPongModel()

  session = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  this_frame = process_frame(env.reset())
  prev_frame = np.zeros_like(this_frame)

  steps = []
  reset_time = time.time()
  saver = tf.train.Saver(
    model.save_variables(),
    max_to_keep=5, keep_checkpoint_every_n_hours=1)

  rounds = 0

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
      train_start = time.time()

      rewards = build_rewards(steps)
      actions = build_actions(steps)

      loss, _ = session.run(
        [model.loss, model.train_step],
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
        saver.save(session, "pong", global_step=rounds)

      env.reset()
      reset_time = time.time()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--render', default=False, action='store_true',
                      help='render simulation')
  parser.add_argument('--hidden', type=int, default=200,
                      help='hidden neurons')
  parser.add_argument('--eta', type=float, default=0.5,
                      help='learning rate')
  parser.add_argument('--checkpoint', type=int, default=0,
                      help='checkpoint every N rounds')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
