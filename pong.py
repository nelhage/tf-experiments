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
HISTORY = 2
ACTIONS = 2

FLAGS = None

class PingPongModel(object):
  VARIABLES_COLLECTIONS = {
    'weights': [tf.GraphKeys.WEIGHTS],
  }

  def __init__(self):
    self.global_step = tf.Variable(1, name='global_step', trainable=False)
    with tf.name_scope('Frames'):
      self.frames = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, PLANES], name="Frames")

    downsampled = self.frames[:,::2,::2]
    stacks = [downsampled[i:-(HISTORY-1-i) if i < HISTORY-1 else None] for i in range(HISTORY)]
    frames = tf.stack(stacks, axis=4)
    frames = tf.reshape(frames, (-1, WIDTH//2, HEIGHT//2, HISTORY*PLANES))

    self.h_conv1 = tf.contrib.layers.conv2d(
      frames, 16,
      scope='Conv1',
      stride=[2, 2],
      kernel_size=[4, 4],
      padding='SAME',
      biases_initializer = tf.constant_initializer(0.1),
      variables_collections = self.VARIABLES_COLLECTIONS,
    )

    tf.summary.histogram('conv1', self.h_conv1)

    out = self.h_conv1
    if FLAGS.pool:
      out = tf.contrib.layers.max_pool2d(
        out, kernel_size=[2, 2], stride=[2, 2], padding='SAME')

    self.h_conv2 = tf.contrib.layers.conv2d(
      out, 16,
      scope='Conv2',
      stride=[2, 2],
      kernel_size=[4, 4],
      padding='SAME',
      biases_initializer = tf.constant_initializer(0.1),
      variables_collections = self.VARIABLES_COLLECTIONS,
    )
    tf.summary.histogram('conv2', self.h_conv2)

    out = self.h_conv2
    if FLAGS.pool:
      out = tf.contrib.layers.max_pool2d(
        out, kernel_size=[2, 2], stride=[2, 2], padding='SAME')

    a_h = tf.contrib.layers.fully_connected(
      tf.contrib.layers.flatten(out),
      scope = 'Hidden',
      num_outputs = FLAGS.hidden,
      activation_fn = tf.nn.relu,
      biases_initializer = tf.constant_initializer(0.1),
      variables_collections = self.VARIABLES_COLLECTIONS,
    )
    tf.summary.histogram('a_h', a_h)

    self.z_o = tf.contrib.layers.fully_connected(
      a_h,
      scope = 'Logits',
      num_outputs = ACTIONS,
      activation_fn = None,
      biases_initializer = tf.constant_initializer(0.1),
      variables_collections = self.VARIABLES_COLLECTIONS,
    )
    self.vp = tf.reshape(
      tf.contrib.layers.fully_connected(
        a_h,
        scope = 'Value',
        num_outputs = 1,
        activation_fn = tf.tanh,
        biases_initializer = tf.constant_initializer(0.1),
        variables_collections = self.VARIABLES_COLLECTIONS,
      ), (-1,))

    self.logits = self.z_o
    tf.summary.histogram('logits', self.logits)
    self.act_probs = tf.nn.softmax(self.logits)

  def add_loss(self):
    with tf.name_scope('Train'):
      self.adv  = tf.placeholder(tf.float32, [None], name="Advantage")
      tf.summary.histogram('advantage', self.adv)
      self.rewards = tf.placeholder(tf.float32, [None], name="Reward")
      tf.summary.histogram('weighted_reward', self.rewards)
      tf.summary.histogram('predicted_value', self.vp)
      self.actions = tf.placeholder(tf.float32, [None, ACTIONS], name="SampledActions")

      self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.actions, logits=self.z_o)
      self.pg_loss = tf.reduce_mean(self.adv * self.cross_entropy)
      tf.summary.scalar('pg_loss', self.pg_loss)
      self.v_loss = 0.5 * tf.reduce_mean(tf.square(self.vp - self.rewards))
      tf.summary.scalar('value_loss', self.v_loss)
      self.entropy = -tf.reduce_mean(
        tf.reduce_sum(self.act_probs * tf.nn.log_softmax(self.logits), axis=1))
      tf.summary.scalar('entropy', self.entropy)

      self.l2_loss = tf.contrib.layers.apply_regularization(
        tf.contrib.layers.l2_regularizer(FLAGS.l2_weight))

      if FLAGS.l2_weight != 0:
        tf.summary.scalar('l2_loss', self.l2_loss / FLAGS.l2_weight)

      self.loss = (
        FLAGS.pg_weight * self.pg_loss +
        FLAGS.v_weight * self.v_loss -
        FLAGS.entropy_weight * self.entropy +
        self.l2_loss
      )

def train_model(model):
  with tf.control_dependencies([model.global_step.assign_add(tf.shape(model.frames)[0])]):
    optimizer = tf.train.AdamOptimizer(FLAGS.eta)
    grads = optimizer.compute_gradients(model.loss)
    clipped, norm = tf.clip_by_global_norm(
      [g for (g, v) in grads], FLAGS.clip_gradient)
    tf.summary.scalar('grad_norm', norm)
    return optimizer.apply_gradients(
      (c, v) for (c, (_,v)) in zip(clipped, grads))

@attr.s(init=False)
class Rollout(object):
  frames    = attr.ib()
  actions   = attr.ib()
  rewards   = attr.ib()
  vp        = attr.ib()

  next_frame = attr.ib()

  def __init__(self):
    self.frames = np.zeros((FLAGS.train_frames, WIDTH, HEIGHT, PLANES))
    self.next_frame = 0
    self.actions = []
    self.rewards = []
    self.vp = []

  def advance_frame(self):
    out = self.frames[self.next_frame]
    self.next_frame += 1
    return out

  def get_frames(self):
    return self.frames[:self.next_frame]

  def clear(self):
    self.frames[:HISTORY] = self.frames[self.next_frame-HISTORY:self.next_frame]
    self.next_frame = 1
    del self.actions[:]
    del self.rewards[:]
    del self.vp[:]

class PongEnvironment(object):
  def __init__(self, model):
    self.model = model

  @staticmethod
  def process_frame(frame, out):
    frame = np.mean(frame, 2, keepdims=True, out=out)
    out -= np.mean(out, (0, 1), keepdims=True)
    out /= np.std(out, axis=(0, 1), keepdims=True)
    return out

  def rollouts(self, session):
    env = gym.make('Pong-v0')

    rollout = Rollout()
    for i in range(HISTORY-1):
      rollout.advance_frame().fill(0)
    self.process_frame(env.reset(), rollout.advance_frame())

    while True:
      if FLAGS.render:
        env.render()

      act_probs, vp, global_step = session.run(
        [self.model.act_probs, self.model.vp, self.model.global_step],
        feed_dict={
          self.model.frames: rollout.frames[
            rollout.next_frame-HISTORY:rollout.next_frame]
        })
      r = np.random.uniform()

      for i, a in enumerate(act_probs[0]):
        if r <= a:
          action = i
          break
        r -= a

      next_frame, reward, done, info = env.step(2 + action)

      rollout.actions.append(action)
      rollout.rewards.append(reward)
      rollout.vp.append(vp[0])

      if done or rollout.next_frame == FLAGS.train_frames:
        rollout.rewards[-1] = rollout.vp[-1]

        yield rollout

        rollout.clear()

        if done:
          self.process_frame(env.reset(), rollout.frames[0])

      self.process_frame(next_frame, rollout.advance_frame())

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def build_rewards(rollout, gamma):
  return discount(rollout.rewards, gamma)

def build_advantage(rollout, gamma, lambda_=1.0):
  rewards = np.array(rollout.rewards)
  vp_t = np.array(rollout.vp + [0])

  delta_t = rewards + gamma * vp_t[1:] - vp_t[:-1]
  return discount(delta_t, gamma*lambda_)

def build_actions(rollout):
  actions = np.zeros((len(rollout.actions), ACTIONS))
  actions[np.arange(len(actions)), rollout.actions] = 1
  return actions

def main(_):
  model = PingPongModel()
  env = PongEnvironment(model)

  if FLAGS.train:
    model.add_loss()
    train_step = train_model(model)

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

  write_summaries = summary_writer and FLAGS.summary_frames
  next_summary = FLAGS.summary_frames
  write_checkpoints = FLAGS.logdir and FLAGS.checkpoint
  next_checkpoint = time.time() + FLAGS.checkpoint

  avgreward = None

  for rollout in env.rollouts(session):
    if FLAGS.train:
      train_start = time.time()

      rewards = build_rewards(rollout, FLAGS.discount)
      adv     = build_advantage(rollout, FLAGS.discount)
      actions = build_actions(rollout)

      ops = {
        'pg_loss': model.pg_loss,
        'v_loss': model.v_loss,
        'train': train_step,
        'global_step': model.global_step,
        'vp': model.vp,
      }
      if summary_op is not None:
        ops['summary'] = summary_op

      out = session.run(
        ops,
        feed_dict = {
          model.frames:  rollout.get_frames(),
          model.actions: actions,
          model.rewards: rewards,
          model.adv:     adv,
        })
      train_end = time.time()
#      print("run_vp={}".format(rollout.vp))
#      print("batch_vp={}".format(out['vp']))

      if avgreward is None:
        avgreward = np.mean(rewards)
      avgreward = 0.9 * avgreward + 0.1 * np.mean(rewards)
      print("done round={global_step} frames={frames} reward={reward:.3f} expreward={avgreward:.3f} pg_loss={pg_loss} v_loss={v_loss} actions={actions}".format(
        frames = len(rollout.actions),
        reward = np.mean(rewards),
        avgreward = avgreward,
        actions = collections.Counter(rollout.actions),
        pg_loss = out['pg_loss'],
        v_loss = out['v_loss'],
        global_step = out['global_step'],
      ))
      fps = len(rollout.actions)/(train_start-reset_time)
      print("play_time={0:.3f}s train_time={1:.3f}s fps={2:.3f}s".format(
        train_start-reset_time, train_end-train_start, fps))
      reset_time = time.time()

    if write_summaries and out['global_step'] >= next_summary:
      next_summary = out['global_step'] + FLAGS.summary_frames
      summary = tf.Summary()
      summary.value.add(tag='env/frames', simple_value=float(len(rollout.actions)))
      summary.value.add(tag='env/fps', simple_value=fps)
      summary.value.add(tag='env/reward', simple_value=np.mean(rewards))
      summary_writer.add_summary(summary, out['global_step'])
      summary_writer.add_summary(out['summary'], out['global_step'])

    if write_checkpoints and time.time() > next_checkpoint:
      next_checkpoint = time.time() + FLAGS.checkpoint
      saver.save(session, os.path.join(FLAGS.logdir, 'pong'), global_step=out['global_step'])

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
  parser.add_argument('--discount', type=float, default=0.99,
                      help='discount rate')
  parser.add_argument('--checkpoint', type=int, default=0,
                      help='checkpoint every N seconds')
  parser.add_argument('--summary_frames', type=int, default=1000,
                      help='write summaries every N frames')
  parser.add_argument('--logdir', type=str, default=None,
                      help='log path')
  parser.add_argument('--load_model', type=str, default=None,
                      help='restore model')

  parser.add_argument('--train_frames', default=1000, type=int,
                      help='Train model every N frames')

  parser.add_argument('--pool', default=False, action='store_true',
                      help='max pool after convolving')

  parser.add_argument('--pg_weight', type=float, default=1.0)
  parser.add_argument('--v_weight', type=float, default=0.5)
  parser.add_argument('--entropy_weight', type=float, default=0.01)
  parser.add_argument('--l2_weight', type=float, default=1e-5)
  parser.add_argument('--clip_gradient', type=float, default=40.0)
  return parser

if __name__ == '__main__':
  parser = arg_parser()
  FLAGS = parser.parse_args()
  tf.app.run(main=main, argv=sys.argv[:1])
else:
  FLAGS = arg_parser().parse_args([])
