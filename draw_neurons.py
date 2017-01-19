import tensorflow as tf
import numpy as np

import PIL.Image

import pong

import os

pong.FLAGS, args = pong.arg_parser().parse_known_args()

model = pong.PingPongModel()
saver = tf.train.Saver(
  model.save_variables(),
  max_to_keep=5, keep_checkpoint_every_n_hours=1)

session = tf.InteractiveSession()

saver.restore(session, args[0])
w_h = session.run(model.w_h)

w_h -= w_h.min(0)
w_h /= w_h.max(0)

try:
  os.makedirs("neurons")
except FileExistsError:
  pass

for i in range(w_h.shape[1]):
  px = (w_h[:,i]*255).reshape(pong.WIDTH, pong.HEIGHT).astype(np.uint8)
  PIL.Image.fromarray(px).save("neurons/n{0}.png".format(i))
