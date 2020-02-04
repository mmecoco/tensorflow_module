from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf
import sys

sys.path.insert(0, "layers")

from convolutions import conv2d, conv_layer, fc_layer, max_pool
from ConvNet import ConvNet

class AlexNet(ConvNet):
    """AlexNet 클래스."""

    def _build_model(self, **kwargs):
        """
        모델 생성.
        :param kwargs: dict, AlexNet 생성을 위한 추가 인자.
            - image_mean: np.ndarray, 평균 이미지: 이미지들의 각 입력 채널별 평균값, shape: (C,).
            - dropout_prob: float, 완전 연결 층에서 각 유닛별 드롭아웃 수행 확률.
        :return d: dict, 각 층에서의 출력값들을 포함함.
        """
        d = dict()    # 각 중간층에서의 출력값을 포함하는 dict.
        X_mean = kwargs.pop('image_mean', 0.0)
        dropout_prob = kwargs.pop('dropout_prob', 0.0)
        num_classes = int(self.y.get_shape()[-1])

        # Dropout을 적용할 층들에서의 각 유닛별 '유지' 확률
        keep_prob = tf.cond(self.is_train,
                            lambda: 1. - dropout_prob,
                            lambda: 1.)

        # input
        X_input = self.X - X_mean    # 기존 입력값으로부터 평균 이미지를 뺌

        # conv1 - relu1 - pool1
        with tf.variable_scope('conv1'):
            d['conv1'] = conv_layer(X_input, 11, 4, 96, padding='VALID',
                                    weights_stddev=0.01, biases_value=0.0)
            print('conv1.shape', d['conv1'].get_shape().as_list())
        d['relu1'] = tf.nn.relu(d['conv1'])
        # (227, 227, 3) --> (55, 55, 96)
        d['pool1'] = max_pool(d['relu1'], 3, 2, padding='VALID')
        # (55, 55, 96) --> (27, 27, 96)
        print('pool1.shape', d['pool1'].get_shape().as_list())

        # conv2 - relu2 - pool2
        with tf.variable_scope('conv2'):
            d['conv2'] = conv_layer(d['pool1'], 5, 1, 256, padding='SAME',
                                    weights_stddev=0.01, biases_value=0.1)
            print('conv2.shape', d['conv2'].get_shape().as_list())
        d['relu2'] = tf.nn.relu(d['conv2'])
        # (27, 27, 96) --> (27, 27, 256)
        d['pool2'] = max_pool(d['relu2'], 3, 2, padding='VALID')
        # (27, 27, 256) --> (13, 13, 256)
        print('pool2.shape', d['pool2'].get_shape().as_list())

        # conv3 - relu3
        with tf.variable_scope('conv3'):
            d['conv3'] = conv_layer(d['pool2'], 3, 1, 384, padding='SAME',
                                    weights_stddev=0.01, biases_value=0.0)
            print('conv3.shape', d['conv3'].get_shape().as_list())
        d['relu3'] = tf.nn.relu(d['conv3'])
        # (13, 13, 256) --> (13, 13, 384)

        # conv4 - relu4
        with tf.variable_scope('conv4'):
            d['conv4'] = conv_layer(d['relu3'], 3, 1, 384, padding='SAME',
                                    weights_stddev=0.01, biases_value=0.1)
            print('conv4.shape', d['conv4'].get_shape().as_list())
        d['relu4'] = tf.nn.relu(d['conv4'])
        # (13, 13, 384) --> (13, 13, 384)

        # conv5 - relu5 - pool5
        with tf.variable_scope('conv5'):
            d['conv5'] = conv_layer(d['relu4'], 3, 1, 256, padding='SAME',
                                    weights_stddev=0.01, biases_value=0.1)
            print('conv5.shape', d['conv5'].get_shape().as_list())
        d['relu5'] = tf.nn.relu(d['conv5'])
        # (13, 13, 384) --> (13, 13, 256)
        d['pool5'] = max_pool(d['relu5'], 3, 2, padding='VALID')
        # (13, 13, 256) --> (6, 6, 256)
        print('pool5.shape', d['pool5'].get_shape().as_list())

        # 전체 feature maps를 flatten하여 벡터화
        f_dim = int(np.prod(d['pool5'].get_shape()[1:]))
        f_emb = tf.reshape(d['pool5'], [-1, f_dim])
        # (6, 6, 256) --> (9216)

        # fc6
        with tf.variable_scope('fc6'):
            d['fc6'] = fc_layer(f_emb, 4096,
                                weights_stddev=0.005, biases_value=0.1)
        d['relu6'] = tf.nn.relu(d['fc6'])
        d['drop6'] = tf.nn.dropout(d['relu6'], keep_prob)
        # (9216) --> (4096)
        print('drop6.shape', d['drop6'].get_shape().as_list())

        # fc7
        with tf.variable_scope('fc7'):
            d['fc7'] = fc_layer(d['drop6'], 4096,
                                weights_stddev=0.005, biases_value=0.1)
        d['relu7'] = tf.nn.relu(d['fc7'])
        d['drop7'] = tf.nn.dropout(d['relu7'], keep_prob)
        # (4096) --> (4096)
        print('drop7.shape', d['drop7'].get_shape().as_list())

        # fc8
        with tf.variable_scope('fc8'):
            d['logits'] = fc_layer(d['relu7'], num_classes,
                                weights_stddev=0.01, biases_value=0.0)
        # (4096) --> (num_classes)

        # softmax
        d['pred'] = tf.nn.softmax(d['logits'])

        return d

    def _build_loss(self, **kwargs):
        """
        모델 학습을 위한 손실 함수 생성.
        :param kwargs: dict, 정규화 항을 위한 추가 인자.
            - weight_decay: float, L2 정규화 계수.
        :return tf.Tensor.
        """
        weight_decay = kwargs.pop('weight_decay', 0.0005)
        variables = tf.trainable_variables()
        l2_reg_loss = tf.add_n([tf.nn.l2_loss(var) for var in variables])

        # 소프트맥스 교차 엔트로피 손실 함수
        softmax_losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
        softmax_loss = tf.reduce_mean(softmax_losses)

        return softmax_loss + weight_decay*l2_reg_loss