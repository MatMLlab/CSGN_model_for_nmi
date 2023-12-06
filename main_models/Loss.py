"""
Loss for material capsule
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf
import numpy as np
from scipy import optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sonnet as snt
import math
from tensorflow.python.keras.layers import Input, Dense

class Pre_loss(object):
    def __init__(self, k=2):
        self.k = k
    def mse(self, y_pred):
        mse = y_pred.best_pre_loss

        return mse

# calculate the mse loss between prediction and real bandgap
def pred_loss(prediction, labels, n_classes = 1):
  """Classification probe with stopped gradient on features."""

  def _pre_probe(features):
      logits = snt.Linear(1)(features)
      logits_1 = tf.reduce_mean(features, axis =-1)
      logits = tf.nn.sigmoid(logits_1)
      xe = tf.reduce_mean(tf.square(logits - labels))
      return xe, logits, logits_1

  return snt.Module(_pre_probe)(prediction)
def pred_loss_1(prediction, labels):

    #prediction = tf.squeeze(prediction, -1)
    #labels = tf.squeeze(labels, 0)

    xe = tf.reduce_mean(tf.abs(prediction - labels))

    kl_losses = kl_loss(prediction, labels)



    return 1*xe + 0*kl_losses

def kl_loss(prediction, labels):

    #prediction = tf.squeeze(prediction, -1)
    #labels = tf.squeeze(labels, 0)
    kl = tf.keras.losses.KLDivergence()(prediction, labels)

    xe = tf.reduce_mean(kl)

    return xe


def mse_metric(target, pre):
    likelihood_loss, mse = tf.unstack(pre)

    alpha = 1
    phi = 0.1
    loss = phi * likelihood_loss + alpha * mse

    return mse

def pre_mse(target, pre):

    best_pre_loss = pre

    return best_pre_loss


def find_peak(input, length_data):

    l = []
    for i in range(1, length_data - 1):
        if input[i-1] < input[i] and input[i] >input[i+1]:
            l.append(i)
        elif input[i] == input[i-1]:
            l.append(i)
    num = len(l)
    return l, num


def gen(l, window_size):
    index = 0
    ans = 0
    times = 0
    while True:
        while index < window_size:
            ans += l[times + index]
            index += 1
        yield float(ans)/float(window_size)
        ###Reset
        index = 0
        ans = 0
        times += 1



def mean_filter(input, window_size):
    window_size = 4
    temp = gen(input, window_size)
    filtered = []
    for i in range(len(input)-window_size):
        filtered.append(next(temp))

    return filtered


oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi) # normalisation factor for gaussian, not needed.
def tf_normal(y, mu, sigma):
    result = tf.subtract(y, mu)
    result = tf.multiply(result,tf.reciprocal(sigma))
    result = -tf.square(result)/2
    return tf.multiply(tf.exp(result),tf.reciprocal(sigma))*oneDivSqrtTwoPI

def get_lossfunc(out_pi, out_sigma, out_mu, y):
    result = tf_normal(y, out_mu, out_sigma)
    result = tf.multiply(result, out_pi)
    result = tf.reduce_sum(result, 1, keep_dims=True)
    result = -tf.log(result)
    return tf.reduce_mean(result)













