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

import cluster

WIDTH  = 210
HEIGHT = 160
PLANES = 1

FLAGS = None

class PingPongModel(object):
  VARIABLES_COLLECTIONS = {
    'weights': [tf.GraphKeys.WEIGHTS],
  }

  def __init__(self, num_actions):
    self.num_actions = num_actions
    with tf.name_scope('Frames'):
      self.frames = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, PLANES], name="Frames")

    downsampled = self.frames[:,::2,::2]
    if FLAGS.history == 1:
      frames = downsampled[1:] - downsampled[:-1]
    else:
      stacks = [downsampled[i:-(FLAGS.history-1-i) if i < FLAGS.history-1 else None] for i in range(FLAGS.history)]
      frames = tf.stack(stacks, axis=4)
      frames = tf.reshape(frames, (-1, WIDTH//2, HEIGHT//2, FLAGS.history*PLANES))

    self.h_conv1 = tf.contrib.layers.conv2d(
      frames, 16,
      scope='Conv1',
      stride=[2, 2],
      kernel_size=[4, 4],
      padding='SAME',
      biases_initializer = tf.constant_initializer(0.1),
      variables_collections = self.VARIABLES_COLLECTIONS,
    )


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

    self.z_o = tf.contrib.layers.fully_connected(
      a_h,
      scope = 'Logits',
      num_outputs = num_actions,
      activation_fn = None,
      biases_initializer = tf.constant_initializer(0.1),
      variables_collections = self.VARIABLES_COLLECTIONS,
    )
    self.vp = tf.reshape(
      tf.contrib.layers.fully_connected(
        a_h,
        scope = 'Value',
        num_outputs = 1,
        activation_fn = None,
        biases_initializer = tf.constant_initializer(0.1),
        variables_collections = self.VARIABLES_COLLECTIONS,
      ), (-1,))

    self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      tf.get_variable_scope().name)
    self.logits = self.z_o
    self.act_probs = tf.nn.softmax(self.logits)

  def add_loss(self):
    with tf.name_scope('Train'):
      self.adv  = tf.placeholder(tf.float32, [None], name="Advantage")
      self.rewards = tf.placeholder(tf.float32, [None], name="Reward")
      self.actions = tf.placeholder(tf.float32, [None, self.num_actions], name="SampledActions")

      self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.actions, logits=self.z_o)
      self.pg_loss = tf.reduce_mean(self.adv * self.cross_entropy)
      tf.summary.scalar('pg_loss', self.pg_loss)
      self.v_loss = 0.5 * tf.reduce_mean(tf.square(self.vp - self.rewards))
      tf.summary.scalar('value_loss', self.v_loss)
      self.entropy = -tf.reduce_mean(
        tf.reduce_sum(self.act_probs * tf.nn.log_softmax(self.logits), axis=1))
      tf.summary.scalar('entropy', self.entropy)

      if FLAGS.l2_weight != 0:
        self.l2_loss = tf.contrib.layers.apply_regularization(
          tf.contrib.layers.l2_regularizer(FLAGS.l2_weight))
        tf.summary.scalar('l2_loss', self.l2_loss / FLAGS.l2_weight)
      else:
        self.l2_loss = 0

      self.loss = (
        FLAGS.pg_weight * self.pg_loss +
        FLAGS.v_weight * self.v_loss -
        FLAGS.entropy_weight * self.entropy +
        self.l2_loss
      )

def train_model(model, apply_to = None):
  optimizer = tf.train.AdamOptimizer(FLAGS.eta)
  grads = optimizer.compute_gradients(model.loss, model.var_list)
  clipped, norm = tf.clip_by_global_norm(
    [g for (g, v) in grads], FLAGS.clip_gradient)
  tf.summary.scalar('grad_norm', norm)
  return optimizer.apply_gradients(zip(clipped, apply_to or model.var_list))

@attr.s(init=False)
class Rollout(object):
  frames    = attr.ib()
  actions   = attr.ib()
  rewards   = attr.ib()
  vp        = attr.ib()

  next_frame = attr.ib()
  last = attr.ib()
  first = attr.ib()

  def __init__(self):
    self.frames = np.zeros((FLAGS.train_frames, WIDTH, HEIGHT, PLANES))
    self.next_frame = 0
    self.actions = []
    self.rewards = []
    self.vp = []
    self.last = False
    self.first = True

  def advance_frame(self):
    out = self.frames[self.next_frame]
    self.next_frame += 1
    return out

  def get_frames(self):
    return self.frames[:self.next_frame]

  def clear(self):
    self.frames[:FLAGS.history] = self.frames[self.next_frame-FLAGS.history:self.next_frame]
    self.next_frame = 1
    del self.actions[:]
    del self.rewards[:]
    del self.vp[:]
    self.last = False
    self.first = False

class PongEnvironment(object):
  def __init__(self, env, model):
    self.env = env
    self.model = model
    self.train_step = None

  @staticmethod
  def process_frame(frame, out):
    frame = np.mean(frame, 2, keepdims=True, out=out)
    out -= np.mean(out, (0, 1), keepdims=True)
    out /= np.std(out, axis=(0, 1), keepdims=True)
    return out

  def rollouts(self, session):
    rollout = Rollout()
    for i in range(max(1, FLAGS.history-1)):
      rollout.advance_frame().fill(0)
    self.process_frame(self.env.reset(), rollout.advance_frame())

    while True:
      if FLAGS.render:
        self.env.render()

      act_probs, vp, global_step = session.run(
        [self.model.act_probs, self.model.vp, self.global_step],
        feed_dict={
          self.model.frames: rollout.frames[
            rollout.next_frame-max(2, FLAGS.history):rollout.next_frame]
        })
      r = np.random.uniform()

      for i, a in enumerate(act_probs[0]):
        if r <= a:
          action = i
          break
        r -= a

      next_frame, reward, done, info = self.env.step(action)

      rollout.actions.append(action)
      rollout.rewards.append(reward)
      rollout.vp.append(vp[0])

      if done or rollout.next_frame == FLAGS.train_frames:
        rollout.last = done

        yield rollout

        rollout.clear()

        if done:
          rollout.first = True
          self.process_frame(self.env.reset(), rollout.frames[0])

      self.process_frame(next_frame, rollout.advance_frame())

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def build_rewards(rollout, gamma):
  return discount(rollout.rewards[:-1]+rollout.vp[-1:], gamma)

def build_advantage(rollout, gamma, lambda_=1.0):
  rewards = np.array(rollout.rewards[:-1]+rollout.vp[-1:])
  vp_t = np.array(rollout.vp + [0])

  delta_t = rewards + gamma * vp_t[1:] - vp_t[:-1]
  return discount(delta_t, gamma*lambda_)

def build_actions(n_action, rollout):
  actions = np.zeros((len(rollout.actions), n_action))
  actions[np.arange(len(actions)), rollout.actions] = 1
  return actions

def run_training(session, sv, env, summary_op=None):
  summary_writer = sv.summary_writer
  write_summaries = FLAGS.train and FLAGS.logdir

  avgreward = None
  reset_time = time.time()
  rollout_frames = 0
  rollout_reward = 0

  for rollout in env.rollouts(session):
    if FLAGS.train:
      train_start = time.time()

      rewards = build_rewards(rollout, FLAGS.discount)
      adv     = build_advantage(rollout, FLAGS.discount)
      actions = build_actions(env.model.num_actions, rollout)

      ops = {
        'pg_loss': env.model.pg_loss,
        'v_loss': env.model.v_loss,
        'train': env.train_step,
        'global_step': env.global_step,
        'vp': env.model.vp,
      }
      if summary_op is not None:
        ops['summary'] = summary_op

      session.run(env.sync_step)
      out = session.run(
        ops,
        feed_dict = {
          env.model.frames:  rollout.get_frames(),
          env.model.actions: actions,
          env.model.rewards: rewards,
          env.model.adv:     adv,
        })
      train_end = time.time()

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

    rollout_frames += len(rollout.actions)
    rollout_reward += sum(rollout.rewards)

    if write_summaries and rollout.first:
      print("rollout done frames={frames} reward={reward} step={global_step} vp0={vp0}".format(
        frames = rollout_frames,
        reward = rollout_reward,
        global_step = out['global_step'],
        vp0 = rollout.vp[0],
      ))
      summary = tf.Summary()
      summary.value.add(tag='env/frames', simple_value=rollout_frames)
      summary.value.add(tag='env/fps', simple_value=fps)
      summary.value.add(tag='env/reward', simple_value=rollout_reward)
      summary.value.add(tag='Train/vp0', simple_value=rollout.vp[0])
      summary_writer.add_summary(summary, out['global_step'])
      summary_writer.add_summary(out['summary'], out['global_step'])
      rollout_frames = 0
      rollout_reward = 0

def run_ps():
  cluster_def = cluster.cluster_def(FLAGS.workers)
  server = tf.train.Server(cluster_def, job_name="ps", task_index=0,
                           config=tf.ConfigProto(device_filters=[cluster.ps_device()]))
  server.join()

def main(_):
  if FLAGS.ps:
    return run_ps()

  cluster_def = cluster.cluster_def(FLAGS.workers)
  server = tf.train.Server(cluster_def, job_name="worker", task_index=FLAGS.task)
  config = tf.ConfigProto(device_filters=[
    cluster.ps_device(),
    cluster.worker_device(FLAGS.task)])
  device = cluster.worker_device(FLAGS.task)

  gymenv = gym.make(FLAGS.environment)

  with tf.device(tf.train.replica_device_setter(1, worker_device=device)):
    with tf.variable_scope('global'):
      global_model = PingPongModel(gymenv.action_space.n)
      global_step = tf.get_variable("global_step", [], tf.int32,
                                    initializer=tf.constant_initializer(0, dtype=tf.int32),
                                    trainable=False)

  with tf.device(device):
    with tf.variable_scope('local'):
      local_model = PingPongModel(gymenv.action_space.n)
      local_model.add_loss()
      env = PongEnvironment(gymenv, local_model)
      env.global_step = global_step

      summary_op = tf.summary.merge_all()
      inc_step = global_step.assign_add(tf.shape(local_model.frames)[0])
      with tf.control_dependencies([inc_step]):
        env.train_step = train_model(local_model, apply_to=global_model.var_list)

      env.sync_step = tf.group(*[v1.assign(v2) for v1, v2 in
                                 zip(local_model.var_list,
                                     global_model.var_list)])

  variables_to_save = [v for v in tf.global_variables() if not v.name.startswith("local")]
  saver = tf.train.Saver(variables_to_save)
  summary_writer = tf.summary.FileWriter(
    os.path.join(FLAGS.logdir, "worker-{}".format(FLAGS.task)))

  sv = tf.train.Supervisor(logdir = FLAGS.logdir,
                           global_step = global_step,
                           saver = saver,
                           summary_writer = summary_writer,
                           summary_op = None,
                           save_model_secs = FLAGS.checkpoint,
                           is_chief=(FLAGS.task==0))

  with sv.managed_session(server.target, config) as session:
    session.run(env.sync_step)
    run_training(session, sv, env, summary_op)

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
  parser.add_argument('--history', type=int, default=2,
                      help='history frames')
  parser.add_argument('--eta', type=float, default=1e-4,
                      help='learning rate')
  parser.add_argument('--discount', type=float, default=0.99,
                      help='discount rate')
  parser.add_argument('--checkpoint', type=int, default=0,
                      help='checkpoint every N seconds')
  parser.add_argument('--logdir', type=str, default=None,
                      help='log path')

  parser.add_argument('--train_frames', default=1000, type=int,
                      help='Train model every N frames')

  parser.add_argument('--no-pool', dest='pool', default=True, action='store_false',
                      help='disable max pool')

  parser.add_argument('--pg_weight', type=float, default=1.0)
  parser.add_argument('--v_weight', type=float, default=0.5)
  parser.add_argument('--entropy_weight', type=float, default=0.01)
  parser.add_argument('--l2_weight', type=float, default=0.0)
  parser.add_argument('--clip_gradient', type=float, default=40.0)

  parser.add_argument('--environment', type=str, default='Pong-v0',
                      help="gym environment to run")

  parser.add_argument('--ps', action='store_true', default=False,
                      help='Run the parameter server')
  parser.add_argument('--workers',
                      type=int,
                      default=1,
                      help='Total worker count')
  parser.add_argument('--task',
                      type=int,
                      default=0,
                      help='Task ID')
  return parser

if __name__ == '__main__':
  parser = arg_parser()
  FLAGS = parser.parse_args()
  tf.app.run(main=main, argv=sys.argv[:1])
else:
  FLAGS = arg_parser().parse_args([])
