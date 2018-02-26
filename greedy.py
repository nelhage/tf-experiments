import gym
import numpy as np

def main():
  env = gym.make('Breakout-v0')
  done = True

  while True:
    if done:
      frame = env.reset()
      frame, r, done, _ = env.step(0)

    prow = (frame[190, -9:9, :] == (200, 72, 72))[:,0]
    if np.any(prow):
      paddlex = 9 + 8 + np.argmax(prow)
    else:
      paddlex = frame.shape[1]/2

    area = (frame[93:188, 9:-9, :] == (200, 72, 72))[:,:,0]
    if np.any(area):
      ballx = 9 + np.argmax(np.argmax(area, axis=0))
    else:
      ballx = frame.shape[1]/2

    if paddlex == ballx:
      action = np.random.randint(4)
    elif paddlex < ballx:
      action = 2
    else:
      action = 3

    # print("ball={} paddle={} action={}".format(ballx, paddlex, action))

    frame, r, done, _ = env.step(action)
    env.render()


if __name__ == '__main__':
  main()
