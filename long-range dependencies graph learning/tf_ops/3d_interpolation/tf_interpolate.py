import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)

def three_nn(xyz1, xyz2):

    return interpolate_module.three_nn(xyz1, xyz2)
ops.NoGradient('ThreeNN')
def three_interpolate(points, idx, weight):

    return interpolate_module.three_interpolate(points, idx, weight)
@tf.RegisterGradient('ThreeInterpolate')
def _three_interpolate_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    weight = op.inputs[2]
    return [interpolate_module.three_interpolate_grad(points, idx, weight, grad_out), None, None]