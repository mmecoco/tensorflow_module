import os
import sys
import numpy as np
import tensorflow as tf
from imread import imread, imsave

sys.path.insert(0, "datasets")
sys.path.insert(0, "models")
sys.path.insert(0, "evaluator")
sys.path.insert(0, "optimizer")

from read_asirra_subset import read_asirra_subset
from DataSet import DataSet
from AlexNet import AlexNet as ConvNet
from AccuracyEvaluator import AccuracyEvaluator as Evaluator
from MomentumOptimizer import MomentumOptimizer as Optimizer

""" 1. 원본 데이터셋을 메모리에 로드함 """
root_dir = os.path.join('../', 'data', 'asirra')    # FIXME
test_dir = os.path.join(root_dir, 'test')

# 테스트 데이터셋을 로드함
X_test, y_test = read_asirra_subset(test_dir, one_hot=True)
test_set = DataSet(X_test, y_test)

# 중간 점검
print('Test set stats:')
print(test_set.images.shape)
print(test_set.images.min(), test_set.images.max())
print((test_set.labels[:, 1] == 0).sum(), (test_set.labels[:, 1] == 1).sum())


""" 2. 테스트를 위한 하이퍼파라미터 설정 """
hp_d = dict()
image_mean = np.load('/tmp/asirra_mean.npy')    # 평균 이미지를 로드
hp_d['image_mean'] = image_mean

# FIXME: 테스트 관련 하이퍼파라미터
hp_d['batch_size'] = 256
hp_d['augment_pred'] = True


""" 3. Graph 생성, 파라미터 로드, session 초기화 및 테스트 시작 """
# 초기화 
graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model = ConvNet([227, 227, 3], 2, **hp_d)
evaluator = Evaluator()
saver = tf.train.Saver()

sess = tf.Session(graph=graph, config=config)
saver.restore(sess, 'save/model.ckpt')    # 학습된 파라미터 로드 및 복원
test_y_pred = model.predict(sess, test_set, **hp_d)
test_score = evaluator.score(test_set.labels, test_y_pred)

print('Test accuracy: {}'.format(test_score))
