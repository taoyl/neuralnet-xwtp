"""
Neeural Network using LM algorithm.
Author: Yuliang Tao
Email : nerotao@foxmail.com
"""

import copy
import json
import codecs
import numpy as np
from scipy.linalg import norm, pinv

#############################################################################
# Global definitions

def relu(z):
    return np.maximum(z, 0)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def tanh(z):
    return (np.exp(z) + np.exp(-z)) / (np.exp(z) + np.exp(-z))


def feedthrough(z):
    return z


def relu_prime(z):
    return 1.0 * (z > 0)


def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))


def tanh_prime(z):
    return 1.0 - tanh(z) * tanh(z)


def feedthrough_prime(z):
    return np.ones((z.shape[0], z.shape[1]))


def activate(acf='relu'):
    if acf == "relu":
        return relu
    elif acf == "sigmoid":
        return sigmoid
    elif acf == "tanh":
        return tanh
    else:
        return feedthrough


def activate_derivative(acf='relu'):
    if acf == 'relu':
        return relu_prime
    elif acf == "sigmoid":
        return sigmoid_prime
    elif acf == "tanh":
        return tanh_prime
    else:
        return feedthrough_prime


#############################################################################
# Class
class ShapeError(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)


class NeuralLayer:
    """
    :param shape : is list of tuples (x, y), where x is the size of neurons,
                   y is the size of inputs.
    :param acf   : activate function, supports sigmod, tanh and feedthrough.
                   Input layer and output layer must be feedthrough
    """

    def __init__(self, shape, acf):
        self.osize, self.isize = shape
        self.acf = acf
        self.f = activate(acf)
        self.f_prime = activate_derivative(acf)
        # PEP8: define attributes in __init__
        self.inputs = None
        self.outputs = None
        self.z = None
        self.biases = None
        self.weights = None
        self.errors = None
        self.w_pard = None
        self.b_pard = None
        self.init_wb()

    # def __setattr__(self, key, value):
    #     print("key={}, value={}".format(key, value))
    #     self.__dict__[key] = value

    def init_wb(self):
        """Initialize weights and biases"""
        self.biases = np.random.randn(self.osize, 1)
        self.weights = np.random.randn(self.osize, self.isize)

    def dump(self):
        """ Debug purpose"""
        dumps = {
            'size': self.osize,
            'acf': self.acf,
            'w': self.weights.tolist(),
            'b': self.biases.tolist()
        }
        return dumps

    def load(self, attrs):
        self.osize = attrs['size']
        self.acf = attrs['acf']
        self.f = activate(self.acf)
        self.f_prime = activate_derivative(self.acf)
        self.weights = np.array(attrs['w'])
        self.biases = np.array(attrs['b'])


class NeuralNetwork:
    """
    :param sizes: a tuple of neural layers (size, 'sigmoid'), excluding the input layer
    :param inputs: training input data, ISIZE x M (M is total number of data)
    :param ref_outputs: reference output data, OSIZE x M
    """

    def __init__(self, inputs, ref_outputs, sizes=None):
        self.inputs = inputs
        self.ref_outputs = ref_outputs
        if sizes:
            self.layers = self.__init_layers(sizes, inputs.shape[0])
            self.size = len(sizes)

    def __init_layers(self, sizes, input_size):
        if sizes[0][0] != input_size:
            raise ShapeError("Error: input size(={}) is not matched with input layer size(={})"
                             .format(input_size, sizes[0][0]))
        self.size = len(sizes)
        layers = [NeuralLayer((sizes[x][0], sizes[x-1][0]), sizes[x][1])
                  for x in range(1, len(sizes))]
        return layers

    def __fp(self):
        prev_layer = None
        # print("prev_layer outputs={}".format(prev_layer.outputs))
        for layer in self.layers:
            layer.inputs = self.inputs if prev_layer is None else prev_layer.outputs
            # print("layer input={}".format(layer.inputs))
            layer.z = np.dot(layer.weights, layer.inputs) + layer.biases
            layer.outputs = layer.f(layer.z)
            # print('layer z={}'.format(layer.z))
            # print('layer outputs={}'.format(layer.outputs))
            prev_layer = layer

    def __bp(self):
        # special case for the output layer
        self.layers[-1].errors = 0.0 - self.layers[-1].f_prime(self.layers[-1].z)
        # print("last layer error={}".format(self.layers[-1].errors))
        prev_layer = self.layers[-1]
        # start from the second-last layer
        for layer in self.layers[-2::-1]:
            weighted_errors = np.dot(prev_layer.weights.transpose(), prev_layer.errors)
            layer.errors = weighted_errors * layer.f_prime(layer.z)
            # print("layer error={}".format(layer.errors))
            prev_layer = layer

    def __update_wb(self, delta_wb, revert=False):
        idx = 0
        for layer in self.layers:
            delta_w = delta_wb[idx: idx + layer.weights.size]
            idx += layer.weights.size
            delta_b = delta_wb[idx: idx + layer.biases.size]
            idx += layer.biases.size
            if revert:
                layer.weights -= np.reshape(delta_w, layer.weights.shape)
                layer.biases -= np.reshape(delta_b, layer.biases.shape)
            else:
                layer.weights += np.reshape(delta_w, layer.weights.shape)
                layer.biases += np.reshape(delta_b, layer.biases.shape)

    def __calc_partial_derivative(self):
        for layer in self.layers:
            inputs = layer.inputs.transpose()
            errors = layer.errors.reshape(-1, 1, order='F')
            matrix = np.zeros((errors.shape[0], inputs.shape[1]))
            # for each neuron
            for i in range(inputs.shape[0]):
                matrix[layer.osize*i: layer.osize*i + layer.osize, :] = \
                    np.repeat([inputs[i, :]], layer.osize, axis=0)
            # print("errors shape={0}, matrix shape={1}".format(errors.shape, matrix.shape))
            w_pard = errors * matrix
            layer.w_pard = np.zeros((inputs.shape[0], layer.weights.size))
            for i in range(inputs.shape[0]):
                row = w_pard[layer.osize*i: layer.osize*i + layer.osize, :]
                # print("row shape={}".format(row.shape))
                layer.w_pard[i, :] = row.reshape(row.size)
            layer.b_pard = layer.errors.transpose()

    def __calc_jacobian(self):
        self.__calc_partial_derivative()
        wb_num = sum([x.weights.size+x.biases.size for x in self.layers])
        data_num = self.inputs.shape[1]
        jacobian = np.zeros((data_num, wb_num))
        idx = 0
        for layer in self.layers:
            col_w_pard = layer.w_pard.shape[1]
            col_b_pard = layer.b_pard.shape[1]
            jacobian[:, idx:idx+col_w_pard] = layer.w_pard
            idx += col_w_pard
            jacobian[:, idx:idx+col_b_pard] = layer.b_pard
            idx += col_b_pard
        # print(jacobian)
        return jacobian

    def __cost_derivative(self, output, ref):
        """Return the vector of partial derivatives \partial C_x /
                \partial a for the output activations."""
        return ref - output

    def __normalize(self, x):
        """Normalize a vector"""
        return norm(x) / len(x)

    def __lm_optimize(self, retry, epoch, mu0, beta, tol, cb=None):
        """Optimizae the mini value of cost function with L-M algorithm"""
        np.seterr(all='print')

        mu = mu0
        pred = self.predict()
        res_err = self.__cost_derivative(pred, self.ref_outputs)
        normed_res_err = self.__normalize(res_err)
        final_epoch = 0
        for i in range(epoch):
            # residual error is n x M, where n is the size of output layer.
            # We need to transpose it to get the error matrix of output layer
            et = res_err.transpose()
            if cb:
                cb('Epoch[{0}]{1: >5}: error={2}, mu={3}'
                   .format(retry, i, normed_res_err, mu), 'blue', 'I')
            else:
                print('Epoch[{0}]{1: >5}: error={2}, mu={3}'.format(retry, i, normed_res_err, mu))
            if normed_res_err < tol:
                final_epoch = i
                break
            self.__bp()
            jacobian = self.__calc_jacobian()
            jacobian_square = jacobian.transpose().dot(jacobian)
            grad = -jacobian.transpose().dot(et)
            # print("je shape{}, je={}".format(grad.shape, grad))
            while True:
                hessian = jacobian_square + mu * np.diag(np.ones(jacobian_square.shape[0]))
                # print("shape{}, A={}".format(hessian.shape, hessian))
                delta_wb = np.dot(pinv(hessian), grad)
                # print("delta_wb={}".format(delta_wb))
                layers = copy.deepcopy(self.layers)
                self.__update_wb(delta_wb)
                pred = self.predict()
                res_err = self.__cost_derivative(pred, self.ref_outputs)
                new_normed_res_error = self.__normalize(res_err)
                # print("pred1={}, pref1={}".format(pred1, new_normed_res_error))
                if new_normed_res_error < normed_res_err:
                    mu = mu / beta
                    normed_res_err = new_normed_res_error
                    break
                else:
                    mu = mu * beta
                    # self.__update_wb(delta_wb, revert=True)
                    # reduce computations using copy
                    self.layers = copy.deepcopy(layers)
        return normed_res_err, mu, final_epoch

    def predict(self, inputs=None):
        """Predict the output according to the inputs"""
        if inputs is not None:
            self.inputs = inputs
        self.__fp()
        return self.layers[-1].outputs

    def train(self, retry, epoch, mu0, beta, tol=0.1, cb=None):
        """Train the network with LM algorithm"""
        residual, mu, final_epoch, err_msg = [None] * 4
        try:
            residual, mu, final_epoch = self.__lm_optimize(retry, epoch, mu0, beta, tol, cb)
        except ValueError as e:
            err_msg = e
        return residual, mu, final_epoch, err_msg

    def randomize_wb(self):
        for layer in self.layers:
            layer.init_wb()


    def dump(self, file=None):
        """Save trained neural network to file"""
        layers = [x.dump() for x in self.layers]
        dumps = {
            'isize': self.inputs.shape[0],
            'layer': layers,
        }
        if file:
            json.dump(dumps, codecs.open(file, 'w', encoding='utf-8'), separators=(',', ':'), indent=4)
        else:
            print(dumps)

    def load(self, file):
        """Load trained neural network from a file"""
        loads = json.loads(codecs.open(file, 'r', encoding='utf-8').read())
        isize = loads['isize']
        attrs = loads['layer']
        sizes = [(x['size'], x['acf']) for x in attrs]
        # Don't forget the input layer size
        sizes.insert(0, (isize, ''))
        self.layers = self.__init_layers(sizes, isize)
        for layer, attr in zip(self.layers, attrs):
            layer.load(attr)
        return True
