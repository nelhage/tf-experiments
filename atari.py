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
import model

FLAGS = None

def train_model(model, apply_to = None):
  optimizer = tf.train.GradientDescentOptimizer(FLAGS.eta)
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

  def __init__(self):
    self.frames = np.zeros((FLAGS.train_frames, model.WIDTH, model.HEIGHT, model.PLANES))
    self.next_frame = 0
    self.actions = []
    self.rewards = []
    self.vp = []
    self.last = False

  def advance_frame(self):
    out = self.frames[self.next_frame]
    self.next_frame += 1
    return out

  def get_frames(self):
    return self.frames[:self.next_frame]

  def clear(self, history):
    self.frames[:history-1] = self.frames[self.next_frame-history+1:self.next_frame]
    self.next_frame = history - 1
    del self.actions[:]
    del self.rewards[:]
    del self.vp[:]
    self.last = False

class RunEnvironment(object):
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
    done = True

    while True:
      if done:
        rollout.clear(1)
        for i in range(self.cfg.history-1):
          rollout.advance_frame().fill(0)
        self.process_frame(self.env.reset(), rollout.advance_frame())

      act_probs, vp, global_step = session.run(
        [self.model.act_probs, self.model.vp, self.global_step],
        feed_dict={
          self.model.frames: rollout.frames[
            rollout.next_frame-self.cfg.history:rollout.next_frame]
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

        rollout.clear(self.cfg.history)

      if not done:
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
  write_summaries = summary_writer is not None

  avgreward = None
  reset_time = time.time()
  rollout_frames = 0
  rollout_reward = 0

  for rollout in env.rollouts(session):
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

    if env.sync_step:
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

    if rollout.last:
      print("rollout done frames={frames} reward={reward} step={global_step} vp0={vp0}".format(
        frames = rollout_frames,
        reward = rollout_reward,
        global_step = out['global_step'],
        vp0 = rollout.vp[0],
      ))
      if write_summaries:
        summary = tf.Summary()
        summary.value.add(tag='env/frames', simple_value=rollout_frames)
        summary.value.add(tag='env/fps', simple_value=fps)
        summary.value.add(tag='env/reward', simple_value=rollout_reward)
        summary.value.add(tag='Train/vp0', simple_value=rollout.vp[0])
        summary_writer.add_summary(summary, out['global_step'])
        summary_writer.add_summary(out['summary'], out['global_step'])
      rollout_frames = 0
      rollout_reward = 0

def build_env():
  gymenv = gym.make(FLAGS.environment)

  cfg = model.Config(
    num_actions = gymenv.action_space.n,
    history = max(2, FLAGS.history),
    difference = FLAGS.history == 1,
    pool = FLAGS.pool,
    hidden = FLAGS.hidden,
  )

  singleton = FLAGS.workers == 1
  device = cluster.worker_device(FLAGS.task)

  if singleton:
    device = None
    global_step = tf.get_variable("global_step", [], tf.int32,
                                  initializer=tf.constant_initializer(0, dtype=tf.int32),
                                  trainable=False)
  else:
    with tf.device(tf.train.replica_device_setter(1, worker_device=device)):
      global_step = tf.get_variable("global_step", [], tf.int32,
                                    initializer=tf.constant_initializer(0, dtype=tf.int32),
                                    trainable=False)
      with tf.variable_scope('global'):
        global_model = model.AtariModel(cfg)

  with tf.device(device):
    with tf.variable_scope('global' if singleton else 'local'):
      local_model = model.AtariModel(cfg)
      local_model.add_loss(
        v_weight = FLAGS.v_weight,
        pg_weight = FLAGS.pg_weight,
        l2_weight = FLAGS.l2_weight,
        entropy_weight = FLAGS.entropy_weight,
      )
      env = RunEnvironment(gymenv, local_model)
      env.cfg = cfg
      env.global_step = global_step

      inc_step = global_step.assign_add(tf.shape(local_model.frames)[0])
      with tf.control_dependencies([inc_step]):
        env.train_step = train_model(local_model,
                                     apply_to=(not singleton and global_model.var_list))

      if singleton:
        env.local_init = None
        env.sync_step = None
        env.global_variables = []
        env.local_variables = tf.global_variables()
      else:
        env.sync_step = tf.group(*[v1.assign(v2) for v1, v2 in
                                   zip(local_model.var_list,
                                   global_model.var_list)])
        env.global_variables = [v for v in tf.global_variables() if not v.name.startswith("local")]
        env.local_variables = [v for v in tf.global_variables() if v.name.startswith("local")]
  return env

def run_ps():
  cluster_def = cluster.cluster_def(FLAGS.workers)
  server = tf.train.Server(cluster_def, job_name="ps", task_index=0,
                           config=tf.ConfigProto(device_filters=[cluster.ps_device()]))
  server.join()

def main(_):
  if FLAGS.ps:
    return run_ps()

  env = build_env()

  variables_to_save = [v for v in tf.global_variables() if not v.name.startswith("local")]
  saver = tf.train.Saver(
    keep_checkpoint_every_n_hours = 1,
    var_list = variables_to_save,
  )
  if FLAGS.logdir and FLAGS.task == 0:
    summary_writer = tf.summary.FileWriter(
      os.path.join(FLAGS.logdir, "worker-{}".format(FLAGS.task)))
  else:
    summary_writer = None
  summary_op = tf.summary.merge_all()

  if FLAGS.load_model:
    init_fn = lambda s: saver.restore(s, FLAGS.load_model)
  else:
    init_fn = None

  sv = tf.train.Supervisor(
    logdir = FLAGS.logdir,
    global_step = env.global_step,
    saver = saver,
    summary_writer = summary_writer,
    summary_op = None,
    save_model_secs = FLAGS.checkpoint,
    is_chief = (FLAGS.task==0),
    init_fn = init_fn,
    ready_for_local_init_op = tf.report_uninitialized_variables(env.global_variables),
    local_init_op = tf.variables_initializer(env.local_variables)
  )
  cluster_def = cluster.cluster_def(FLAGS.workers)
  devices = [cluster.worker_device(FLAGS.task)]


  if FLAGS.workers == 1:
    master = ''
  else:
    devices.append(cluster.ps_device())
    server = tf.train.Server(cluster_def, job_name="worker", task_index=FLAGS.task)
    master = server.target

  config = tf.ConfigProto(device_filters=devices)

  with sv.managed_session(master=master, config=config) as session:
    run_training(session, sv, env, summary_op)

def arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--hidden', type=int, default=256,
                      help='hidden neurons')
  parser.add_argument('--history', type=int, default=2,
                      help='history frames')
  parser.add_argument('--eta', type=float, default=0.2,
                      help='learning rate')
  parser.add_argument('--discount', type=float, default=0.99,
                      help='discount rate')
  parser.add_argument('--checkpoint', type=int, default=0,
                      help='checkpoint every N seconds')
  parser.add_argument('--logdir', type=str, default=None,
                      help='log path')
  parser.add_argument('--load_model', type=str, default=None,
                      help='load model')

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
