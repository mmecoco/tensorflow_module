from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

from Evaluator import Evaluator

class AccuracyEvaluator(Evaluator):
    """정확도를 평가 척도로 사용하는 evaluator 클래스."""

    @property
    def worst_score(self):
        """최저 성능 점수."""
        return 0.0

    @property
    def mode(self):
        """점수가 높아야 성능이 우수한지, 낮아야 성능이 우수한지 여부."""
        return 'max'

    def score(self, y_true, y_pred):
        """정확도에 기반한 성능 평가 점수."""
        return accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))

    def is_better(self, curr, best, **kwargs):
        """
        상대적 문턱값을 고려하여, 현재 주어진 성능 점수가 현재까지의 최고 성능 점수보다
        우수한지 여부를 반환하는 함수.
        :param kwargs: dict, 추가 인자.
            - score_threshold: float, 새로운 최적값 결정을 위한 상대적 문턱값으로,
                               유의미한 차이가 발생했을 경우만을 반영하기 위함.
        """
        score_threshold = kwargs.pop('score_threshold', 1e-4)
        relative_eps = 1.0 + score_threshold
        return curr > best * relative_eps