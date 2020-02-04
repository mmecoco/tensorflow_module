import numpy as np
import tensorflow as tf

def bias_variable(shape, value=1.0):
    """
    새로운 바이어스 변수를 주어진 shape에 맞게 선언하고, 
    주어진 상수값으로 추기화함.
    :param shape: list(int).
    :param value: float, 바이어스의 초기화 값.
    :return biases: tf.Variable.
    """
    biases = tf.get_variable('biases', shape, tf.float32,
                             tf.constant_initializer(value=value))
    return biases