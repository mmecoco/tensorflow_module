from bias import bias_variable
from weight import weight_variable
import numpy as np
import tensorflow as tf

def conv2d(x, W, stride, padding='SAME'):
    """
    주어진 입력값과 필터 가중치 간의 2D 컨볼루션을 수행함.
    :param x: tf.Tensor, shape: (N, H, W, C).
    :param W: tf.Tensor, shape: (fh, fw, ic, oc).
    :param stride: int, 필터의 각 방향으로의 이동 간격.
    :param padding: str, 'SAME' 또는 'VALID',
                         컨볼루션 연산 시 입력값에 대해 적용할 패딩 알고리즘.
    :return: tf.Tensor.
    """
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)


def max_pool(x, side_l, stride, padding='SAME'):
    """
    주어진 입력값에 대해 최댓값 풀링(max pooling)을 수행함.
    :param x: tf.Tensor, shape: (N, H, W, C).
    :param side_l: int, 풀링 윈도우의 한 변의 길이.
    :param stride: int, 풀링 윈도우의 각 방향으로의 이동 간격. 
    :param padding: str, 'SAME' 또는 'VALID',
                         풀링 연산 시 입력값에 대해 적용할 패딩 알고리즘.
    :return: tf.Tensor.
    """
    return tf.nn.max_pool(x, ksize=[1, side_l, side_l, 1],
                          strides=[1, stride, stride, 1], padding=padding)


def conv_layer(x, side_l, stride, out_depth, padding='SAME', **kwargs):
    """
    새로운 컨볼루션 층을 추가함.
    :param x: tf.Tensor, shape: (N, H, W, C).
    :param side_l: int, 필터의 한 변의 길이.
    :param stride: int, 필터의 각 방향으로의 이동 간격.
    :param out_depth: int, 입력값에 적용할 필터의 총 개수.
    :param padding: str, 'SAME' 또는 'VALID',
                         컨볼루션 연산 시 입력값에 대해 적용할 패딩 알고리즘.
    :param kwargs: dict, 추가 인자, 가중치/바이어스 초기화를 위한 하이퍼파라미터들을 포함함.
        - weight_stddev: float, 샘플링 대상이 되는 정규분포의 표준편차 값.
        - biases_value: float, 바이어스의 초기화 값.
    :return: tf.Tensor.
    """
    weights_stddev = kwargs.pop('weights_stddev', 0.01)
    biases_value = kwargs.pop('biases_value', 0.1)
    in_depth = int(x.get_shape()[-1])

    filters = weight_variable([side_l, side_l, in_depth, out_depth], stddev=weights_stddev)
    biases = bias_variable([out_depth], value=biases_value)
    return conv2d(x, filters, stride, padding=padding) + biases


def fc_layer(x, out_dim, **kwargs):
    """
    새로운 완전 연결 층을 추가함.
    :param x: tf.Tensor, shape: (N, D).
    :param out_dim: int, 출력 벡터의 차원수.
    :param kwargs: dict, 추가 인자, 가중치/바이어스 초기화를 위한 하이퍼파라미터들을 포함함. 
        - weight_stddev: float, 샘플링 대상이 되는 정규분포의 표준편차 값.
        - biases_value: float, 바이어스의 초기화 값.
    :return: tf.Tensor.
    """
    weights_stddev = kwargs.pop('weights_stddev', 0.01)
    biases_value = kwargs.pop('biases_value', 0.1)
    in_dim = int(x.get_shape()[-1])

    weights = weight_variable([in_dim, out_dim], stddev=weights_stddev)
    biases = bias_variable([out_dim], value=biases_value)
    return tf.matmul(x, weights) + biases