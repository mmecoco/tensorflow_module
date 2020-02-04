import numpy as np
import tensorflow as tf

def weight_variable(shape, stddev=0.01):
    """
    새로운 가중치 변수를 주어진 shape에 맞게 선언하고,
    Normal(0.0, stddev^2)의 정규분포로부터의 샘플링을 통해 초기화함.
    :param shape: list(int).
    :param stddev: float, 샘플링 대상이 되는 정규분포의 표준편차 값.
    :return weights: tf.Variable.
    """
    weights = tf.get_variable('weights', shape, tf.float32,
                              tf.random_normal_initializer(mean=0.0, stddev=stddev))
    return weights