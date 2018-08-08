"""
Neeural Network API.
Author: Yuliang Tao
Email : nerotao@foxmail.com
"""

import numpy as np
from neuralnetwork import ShapeError, NeuralNetwork

class NnApi(object):
    """
    Neural Network API class
    """
    def __init__(self, tr_data):
        self.training_x, self.training_y = self._parse_training_data(tr_data)
        net = [(self.training_x.shape[0], ''), (4, 'sigmoid'), (4, 'sigmoid'), (self.training_y.shape[0], 'feedthrough')]
        try:
            self.network = NeuralNetwork(self.training_x, self.training_y, sizes=net)
        except ShapeError as e:
            print('初始化神经网络结构失败!')

    def _parse_training_data(self, tr_data):
        """
        Training data is an array of dict:
        E.g. [ {'input' : [d1, d2, d3...], 'output' : out}, ...]
        returned data is x & y whose
        shape is N x M, where M is the size of training data
        """
        training_x = [d['input'] for d in tr_data]
        training_y = [d['output'] for d in tr_data]
        training_x = np.array(training_x).transpose()
        training_y = np.array(training_y).reshape(1, len(tr_data))
        return training_x, training_y

    def _parse_predict_data(self, pred_data):
        pred_x = [d['input'] for d in pred_data]
        return np.array(pred_x).transpose()

    def train(self, tol0=0.1, retry_num=5):
        """ network training"""
        epoch0, mu0, beta = (20000, 0.1, 10)
        retry = 0
        print("使用LM算法开始训练神经网络...")
        while retry < retry_num:
            residual, mu, citer, msg = \
                self.network.train(retry=retry, epoch=epoch0, mu0=mu0, beta=beta, tol=tol0)
            if residual is None:
                if retry == (retry_num - 1):
                    print("训练失败!".format(msg))
                    return False, -1
                else:
                    print("第{}轮训练失败:{}, 重试中...".format(retry, msg))
                    self.network.randomize_wb()
                    # continue
            elif residual > tol0:
                if retry == (retry_num -1):
                    print("训练失败!".format(msg))
                    return False, -1
                else:
                    print("第{}轮训练失败: 运算未能收敛, 重试中...".format(retry))
                    self.network.randomize_wb()
                    # continue
            else:
                print("神经网络训练完成, 迭代次数={1}, 最终残差={0}"
                               .format(residual, citer + retry * epoch0))
                break
            retry += 1
        return True, residual

    def predict(self, pred_data):
        """Predict data: returned data is list"""
        pred_x = self._parse_predict_data(pred_data)
        return self.network.predict(pred_x)[0]

    def save_network(self, fname):
        self.network.dump(fname)

    def load_network(self, fname):
        try:
            self.network = NeuralNetwork(self.training_x, self.training_y)
            self.network.load(fname)
        except ShapeError as e:
            print('加载神经网络文件{}失败, 输入或输出维度不匹配!'.format(fname))
            return False
        return True
