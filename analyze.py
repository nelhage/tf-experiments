from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import gym

import tensorflow as tf
import numpy as np

import model

def process_frame(frame):
  frame = np.mean(frame, 2, keepdims=True)
  frame -= np.mean(frame, (0, 1), keepdims=True)
  frame /= np.std(frame, (0, 1), keepdims=True)
  return frame

def main(argv):
  gymenv = gym.make(FLAGS.environment)
  cfg = model.Config(
    num_actions = gymenv.action_space.n,
    history = FLAGS.history,
    pool = FLAGS.pool,
    hidden = FLAGS.hidden,
  )
  with tf.variable_scope('global'):
      predict = model.AtariModel(cfg)

  session = tf.InteractiveSession()

  var_list = None
  if FLAGS.strip_global:
    var_list = dict(
      (v.name.replace('global/', '').replace(":0", ""), v)
      for v in tf.global_variables()
    )

  saver = tf.train.Saver(var_list=var_list)
  saver.restore(session, FLAGS.model)

  depth = FLAGS.history
  if depth == 1:
    depth = 2

  done = True
  reward = None
  while True:
    if done:
      if reward is not None:
        print("rollout done frames={frames} reward={reward}".format(
          frames = nframes,
          reward = reward,
        ))
        if FLAGS.one:
          break
      reward = 0
      nframes = 0
      gymenv.reset()
      frames = np.zeros((depth, model.WIDTH, model.HEIGHT, model.PLANES))
      frames[0] = process_frame(gymenv.reset())
      i = 1

    if FLAGS.render:
      gymenv.render()

    feed_frames = np.concatenate([frames[i:], frames[:i]])

    act_probs = session.run(predict.act_probs, feed_dict={
      predict.frames: feed_frames
    })
    r = np.random.uniform()

    for j, a in enumerate(act_probs[0]):
      if r <= a:
        action = j
        break
      r -= a

    next_frame, r, done, info = gymenv.step(action)
    reward += r
    nframes += 1
    frames[i] = process_frame(next_frame)
    i = (i + 1) % depth

def arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--render', default=False, action='store_true',
                      help='render')

  parser.add_argument('--hidden', type=int, default=256,
                      help='hidden neurons')
  parser.add_argument('--history', type=int, default=2,
                      help='history frames')

  parser.add_argument('--no-pool', dest='pool', default=True, action='store_false',
                      help='disable max pool')

  parser.add_argument('--environment', type=str, default='Pong-v0',
                      help="gym environment to run")

  parser.add_argument('--model', type=str, default=None,
                      help="model to load")
  parser.add_argument('--strip-global',
                      default=False,
                      dest='strip_global',
                      action='store_true',
                      help="Strip global/ from names")

  parser.add_argument('--one',
                      default=False,
                      dest='one',
                      action='store_true')

  parser.add_argument('--no-strip-global',
                      dest='strip_global',
                      action='store_false')

  return parser

if __name__ == '__main__':
  parser = arg_parser()
  FLAGS, args = parser.parse_known_args()
  tf.app.run(main=main, argv=args)
else:
  FLAGS = arg_parser().parse_args([])
