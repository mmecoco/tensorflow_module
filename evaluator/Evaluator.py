from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import tensorflow as tf

class Evaluator(object):
    """성능 평가를 위한 evaluator의 베이스 클래스."""

    @abstractproperty
    def worst_score(self):
        """
        최저 성능 점수.
        :return float.
        """
        pass

    @abstractproperty
    def mode(self):
        """
        점수가 높아야 성능이 우수한지, 낮아야 성능이 우수한지 여부. 'max'와 'min' 중 하나.
        e.g. 정확도, AUC, 정밀도, 재현율 등의 경우 'max',
             오류율, 미검률, 오검률 등의 경우 'min'.
        :return: str.
        """
        pass

    @abstractmethod
    def score(self, y_true, y_pred):
        """
        실제로 사용할 성능 평가 지표.
        해당 함수를 추후 구현해야 함.
        :param y_true: np.ndarray, shape: (N, num_classes).
        :param y_pred: np.ndarray, shape: (N, num_classes).
        :return float.
        """
        pass

    @abstractmethod
    def is_better(self, curr, best, **kwargs):
        """
        현재 주어진 성능 점수가 현재까지의 최고 성능 점수보다 우수한지 여부를 반환하는 함수.
        해당 함수를 추후 구현해야 함.
        :param curr: float, 평가 대상이 되는 현재 성능 점수.
        :param best: float, 현재까지의 최고 성능 점수.
        :return bool.
        """
        pass