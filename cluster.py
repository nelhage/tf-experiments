import tensorflow as tf

BASE_PORT = 11000

def cluster_spec(workers=1, ps=1):
  cluster = {}
  workers = ['ps']*ps + ['worker']*workers
  host = '127.0.0.1'

  for off, what in enumerate(workers):
    cluster.setdefault(what, []).append("{}:{}".format(host, BASE_PORT+off))

  return cluster

def cluster_def(workers=1, ps=1):
  return tf.train.ClusterSpec(cluster_spec(workers, ps)).as_cluster_def()

def ps_device():
  return "/job:ps"

def worker_device(idx):
  return "/job:worker/task:{0}/cpu:0".format(idx)
