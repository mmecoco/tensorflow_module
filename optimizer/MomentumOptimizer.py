from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf

from Optimizer import Optimizer

class MomentumOptimizer(Optimizer):
    """모멘텀 알고리즘을 포함한 경사 하강 optimizer 클래스."""

    def _optimize_op(self, **kwargs):
        """
        경사 하강 업데이트를 위한 tf.train.MomentumOptimizer.minimize Op.
        :param kwargs: dict, optimizer의 추가 인자.
            - momentum: float, 모멘텀 계수.
        :return tf.Operation.
        """
        momentum = kwargs.pop('momentum', 0.9)

        update_vars = tf.trainable_variables()
        return tf.train.MomentumOptimizer(self.learning_rate_placeholder, momentum, use_nesterov=False)\
                .minimize(self.model.loss, var_list=update_vars)

    def _update_learning_rate(self, **kwargs):
        """
        성능 평가 점수 상에 개선이 없을 때, 현 학습률 값을 업데이트함.
        :param kwargs: dict, 학습률 스케줄링을 위한 추가 인자.
            - learning_rate_patience: int, 성능 향상이 연속적으로 이루어지지 않은 epochs 수가 
                                      해당 값을 초과할 경우, 학습률 값을 감소시킴.
            - learning_rate_decay: float, 학습률 업데이트 비율.
            - eps: float, 업데이트된 학습률 값과 기존 학습률 값 간의 차이가 해당 값보다 작을 경우,
                          학습률 업데이트를 취소함.
        """
        learning_rate_patience = kwargs.pop('learning_rate_patience', 10)
        learning_rate_decay = kwargs.pop('learning_rate_decay', 0.1)
        eps = kwargs.pop('eps', 1e-8)

        if self.num_bad_epochs > learning_rate_patience:
            new_learning_rate = self.curr_learning_rate * learning_rate_decay
            # 새 학습률 값과 기존 학습률 값 간의 차이가 eps보다 큰 경우에 한해서만 업데이트를 수행함
            if self.curr_learning_rate - new_learning_rate > eps:
                self.curr_learning_rate = new_learning_rate
            self.num_bad_epochs = 0