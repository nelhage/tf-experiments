from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import attr

WIDTH  = 210
HEIGHT = 160
PLANES = 1

@attr.s(frozen=True, slots=True)
class Config(object):
    num_actions = attr.ib(validator=attr.validators.instance_of(int))
    history = attr.ib(validator=attr.validators.instance_of(int))
    difference = attr.ib(validator=attr.validators.instance_of(bool))
    pool = attr.ib(validator=attr.validators.instance_of(bool))
    hidden = attr.ib(validator=attr.validators.instance_of(int))

class AtariModel(object):
  VARIABLES_COLLECTIONS = {
    'weights': [tf.GraphKeys.WEIGHTS],
  }

  def __init__(self, cfg):
    self.cfg = cfg
    self.num_actions = cfg.num_actions
    with tf.name_scope('Frames'):
      self.frames = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, PLANES], name="Frames")

    frames = self.frames[:,::2,::2]
    if cfg.difference:
      frames = downsampled[1:] - downsampled[:-1]

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
    if cfg.pool:
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
    if cfg.pool:
      out = tf.contrib.layers.max_pool2d(
        out, kernel_size=[2, 2], stride=[2, 2], padding='SAME')

    if not cfg.difference:
      stacks = [out[i:-(cfg.history-1-i) if i < cfg.history-1 else None] for i in range(cfg.history)]
      out = tf.concat(stacks, axis=3)

    a_h = tf.contrib.layers.fully_connected(
      tf.contrib.layers.flatten(out),
      scope = 'Hidden',
      num_outputs = cfg.hidden,
      activation_fn = tf.nn.relu,
      biases_initializer = tf.constant_initializer(0.1),
      variables_collections = self.VARIABLES_COLLECTIONS,
    )

    self.z_o = tf.contrib.layers.fully_connected(
      a_h,
      scope = 'Logits',
      num_outputs = cfg.num_actions,
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

  def add_loss(self, pg_weight, l2_weight, v_weight, entropy_weight):
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

      if l2_weight != 0:
        self.l2_loss = tf.contrib.layers.apply_regularization(
          tf.contrib.layers.l2_regularizer(l2_weight))
        tf.summary.scalar('l2_loss', self.l2_loss / l2_weight)
      else:
        self.l2_loss = 0

      self.loss = (
        pg_weight * self.pg_loss +
        v_weight * self.v_loss -
        entropy_weight * self.entropy +
        self.l2_loss
      )
