from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf
import time

class ConvNet(object):
    """컨볼루션 신경망 모델의 베이스 클래스."""

    def __init__(self, input_shape, num_classes, **kwargs):
        """
        모델 생성자.
        :param input_shape: tuple, shape (H, W, C) 및 값 범위 [0.0, 1.0]의 입력값.
        :param num_classes: int, 총 클래스 개수.
        """
        self.X = tf.placeholder(tf.float32, [None] + input_shape)
        self.y = tf.placeholder(tf.float32, [None] + [num_classes])
        self.is_train = tf.placeholder(tf.bool)

        # 모델과 손실 함수 정의
        self.d = self._build_model(**kwargs)
        self.logits = self.d['logits']
        self.pred = self.d['pred']
        self.loss = self._build_loss(**kwargs)

    @abstractmethod
    def _build_model(self, **kwargs):
        """
        모델 생성.
        해당 함수를 추후 구현해야 함. 
        """
        pass

    @abstractmethod
    def _build_loss(self, **kwargs):
        """
        모델 학습을 위한 손실 함수 생성.
        해당 함수를 추후 구현해야 함. 
        """
        pass

    def predict(self, sess, dataset, verbose=False, **kwargs):
        """
        주어진 데이터셋에 대한 예측을 수행함.
        :param sess: tf.Session.
        :param dataset: DataSet.
        :param verbose: bool, 예측 과정에서 구체적인 정보를 출력할지 여부.
        :param kwargs: dict, 예측을 위한 추가 인자.
            - batch_size: int, 각 반복 회차에서의 미니배치 크기.
            - augment_pred: bool, 예측 과정에서 데이터 증강을 수행할지 여부.
        :return _y_pred: np.ndarray, shape: (N, num_classes).
        """
        batch_size = kwargs.pop('batch_size', 256)
        augment_pred = kwargs.pop('augment_pred', True)

        if dataset.labels is not None:
            assert len(dataset.labels.shape) > 1, 'Labels must be one-hot encoded.'
        num_classes = int(self.y.get_shape()[-1])
        pred_size = dataset.num_examples
        num_steps = pred_size // batch_size

        if verbose:
            print('Running prediction loop...')

        # 예측 루프를 시작함
        _y_pred = []
        start_time = time.time()
        for i in range(num_steps+1):
            if i == num_steps:
                _batch_size = pred_size - num_steps*batch_size
            else:
                _batch_size = batch_size
            X, _ = dataset.next_batch(_batch_size, shuffle=False,
                                      augment=augment_pred, is_train=False)
            # if augment_pred == True:  X.shape: (N, 10, h, w, C)
            # else:                     X.shape: (N, h, w, C)

            # 예측 과정에서 데이터 증강을 수행할 경우,
            if augment_pred:
                y_pred_patches = np.empty((_batch_size, 10, num_classes),
                                          dtype=np.float32)    # (N, 10, num_classes)
                # 10종류의 patch 각각에 대하여 예측 결과를 산출하고,
                for idx in range(10):
                    y_pred_patch = sess.run(self.pred,
                                            feed_dict={self.X: X[:, idx],    # (N, h, w, C)
                                                       self.is_train: False})
                    y_pred_patches[:, idx] = y_pred_patch
                # 이들 10개 예측 결과의 평균을 산출함
                y_pred = y_pred_patches.mean(axis=1)    # (N, num_classes)
            else:
                # 예측 결과를 단순 산출함
                y_pred = sess.run(self.pred,
                                  feed_dict={self.X: X,
                                             self.is_train: False})    # (N, num_classes)

            _y_pred.append(y_pred)
        if verbose:
            print('Total evaluation time(sec): {}'.format(time.time() - start_time))

        _y_pred = np.concatenate(_y_pred, axis=0)    # (N, num_classes)

        return _y_pred