import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from scipy.special import expit
import torch

np.set_printoptions(threshold=1000000)

"""
    这是问题ex4的深层神经网络版本
"""

# 读取数据
data = loadmat('ex4data1.mat')

# 搭建神经网络


# 获取网格信息
def initial_layer(_y):
    # 返回一个列表，列表中第i个元素代表第i个隐藏层中的激活单元个数
    _hidden_layer = int(input("请输入隐藏层的个数："))  # 隐藏层的个数
    _layer_dims = []  # 空列表，用于存放隐藏层中激活单元的个数

    for i in range(_hidden_layer):
        _dims = input("请输入第" + str(i + 1) + "个隐藏层的激活单元个数：")
        _layer_dims.append(int(_dims))

    _layer_dims.append(_y.shape[0])

    for i in range(_hidden_layer):
        print("第"+str(i)+"层的激活单元数量为："+str(_layer_dims[i]))

    return _layer_dims


# 初始化神经网络
def initial_parameters(_X, _layer_dims):
    """
        _X：输入端
        _layer_dims：列表，列表中第i个元素代表第i个隐藏层中的激活单元个数
        返回值：
        _parameters：字典，其中存储了传递矩阵W_i和常数项b_i的信息，以及神经网络层数
        W_i：计算第i个隐藏层所要用到的传递矩阵，形状为(_layer_dims[i], _layer_dims[i-1])
        b_i：计算第i个隐藏层所要用到的常数向量，形状为(_layer_dims[i], 1)
    """
    _layer = len(_layer_dims)  # 层数
    _parameters = {"L": _layer, "m": _X.shape[0]}  # 将神经网络层数作为参数加入其中
    np.random.seed(1)

    # 计算第1层的传递矩阵和常数向量
    # W_i = np.random.random((_layer_dims[0], _X.shape[0])) * np.sqrt(2/_X.shape[0])
    # b_i = np.random.random((_layer_dims[0], 1)) * np.sqrt(2/_X.shape[0])
    W_i = np.random.random((_layer_dims[0], _X.shape[0])) * np.sqrt(1 / _X.shape[0])
    b_i = np.zeros((_layer_dims[0], 1))

    _parameters['W' + str(1)] = W_i
    _parameters['b' + str(1)] = b_i

    # 利用循环计算剩余的传递矩阵和常数向量
    for _l in range(1, _layer):
        # W_i = np.random.random((_layer_dims[_l], _layer_dims[_l-1])) * np.sqrt(2/_layer_dims[_l-1])
        # b_i = np.random.random((_layer_dims[_l], 1)) * np.sqrt(2/_layer_dims[_l-1])
        W_i = np.random.random((_layer_dims[_l], _layer_dims[_l - 1])) * np.sqrt(1 / _X.shape[0])
        b_i = np.zeros((_layer_dims[_l], 1))

        _parameters['W' + str(_l + 1)] = W_i
        _parameters['b' + str(_l + 1)] = b_i

    return _parameters


# 归一化输入
def Normalization(_X):
    _m = _X.shape[0]
    _mu = np.sum(_X, axis=0) / _m  # 求出每行的平均值
    _X = _X - _mu
    _sigma = np.sqrt(np.sum(np.power(_X, 2), axis=0) / _m)
    _X = _X / _sigma

    return _X


# 各类激活函数及其导数


# sigmoid函数
def sigmoid(z):
    return expit(z)


def sigmoid_derivative(z):
    return np.multiply(sigmoid(z), 1 - sigmoid(z))


# ReLU函数
def Leaky_ReLU(z, alpha=0.1):
    w = np.where(z > 0, z, alpha * z)
    return w


def Leaky_ReLU_derivative(z, alpha=0.1):
    w = np.where(z > 0, 1, alpha)
    return w


# SoftMax函数，仅用在最后一层
def SoftMax(z):
    t = np.exp(z)
    a = t / np.sum(t, axis=0)
    return a


# 代价函数


# 对sigmoid函数的代价函数
def cost_sigmoid(_AL, _y, _parameters, _lambda=0.0):
    """
    _AL: 预测值；
    _y: 真值；
    _parameters: 参数列表，包括网格的基本信息
    normalization: 是否正则化，默认关闭
    return: 代价函数值cost
    """
    _m = _parameters['m']
    _L = _parameters['L']
    _first = np.multiply(_y, np.log(_AL))
    _second = np.multiply(1 - _y, np.log(1 - _AL))
    _cost = (-1 / _m) * np.sum(_first + _second)
    for _l in range(1, _L + 1):
        _cost += _lambda * np.sum(np.power(_parameters['W' + str(_l)], 2)) / (2 * _m)

    return _cost


# 对SoftMax函数的代价函数
def cost_SoftMax(_AL, _y, _parameters, _lambda=0.0):
    """
    _AL: 预测值；
    _y: 真值；
    _parameters: 参数列表，包括网格的基本信息
    normalization: 是否正则化，默认关闭
    return: 代价函数值cost
    """
    _m = _parameters['m']
    _cost = (-1 / _m) * np.sum(np.multiply(_y, np.log(_AL)))
    _L = _parameters['L']
    for _l in range(1, _L + 1):
        _cost += _lambda * np.sum(np.power(_parameters['W' + str(_l)], 2)) / (2 * _m)

    return _cost


# 正向传播和反向传播
def forward_propagation(_X, _parameters, activation='sigmoid', predict='sigmoid'):
    """
    _X: 输入端
    _parameters: 参数字典，包括神经网络的各项基本信息
    activation: 除最后一层之外的激活函数，默认为sigmoid
    predict: 最后一层的激活函数，默认为sigmoid
    return:
    a_L，预测值，正向传播的结果
    caches，字典，包括各层的预测值a_l以及中间值z_l
    """

    _L = _parameters['L']  # 神经网络的层数
    _caches = {'a0': _X}  # 将_X作为a_0添加进caches中

    # 对除最后一层以外的其他层进行前向传播
    for _l in range(1, _L):
        z_l = np.dot(_parameters['W' + str(_l)], _caches['a' + str(_l - 1)]) + _parameters['b' + str(_l)]
        a_l = np.zeros(z_l.shape)
        if activation == 'sigmoid':
            a_l = sigmoid(z_l)
        elif activation == 'ReLU':
            a_l = Leaky_ReLU(z_l)

        _caches['a' + str(_l)] = a_l
        _caches['z' + str(_l)] = z_l

    # 对最后一层进行前向传播
    z_L = np.dot(_parameters['W' + str(_L)], _caches['a' + str(_L - 1)]) + _parameters['b' + str(_L)]
    a_L = np.zeros(z_L.shape)
    if predict == 'sigmoid':
        a_L = sigmoid(z_L)
    elif predict == 'SoftMax':
        a_L = SoftMax(z_L)

    _caches['a' + str(_L)] = a_L
    _caches['z' + str(_L)] = z_L

    return a_L, _caches


def backward_propagation(_AL, _y, _parameters, _caches, _lambda=0.0, activation='sigmoid'):
    """
    _AL: 预测值
    _y: 真值
    _parameters: 参数字典，包括神经网络的基本信息
    _caches: 参数字典，包括正向传播的各个中间值
    activation: 除最后一层外的激活函数
    return: grads,包括各参数梯度的字典
    """
    _m = _parameters['m']
    _L = _parameters['L']
    _grads = {}  # 将dz_L添加进grads中

    for _l in reversed(range(1, _L + 1)):
        if _l == _L:
            da_l = _AL - _y
            _grads['da' + str(_l)] = da_l

        if activation == 'sigmoid':
            dz_l = np.multiply(_grads['da' + str(_l)], sigmoid_derivative(_caches['z' + str(_l)]))
        else:
            dz_l = np.multiply(_grads['da' + str(_l)], Leaky_ReLU_derivative(_caches['z' + str(_l)]))

        dw_l = (1 / _m) * np.dot(dz_l, _caches['a' + str(_l - 1)].T) + (_lambda / _m) * _parameters['W' + str(_l)]
        db_l = (1 / _m) * np.sum(dz_l, axis=1, keepdims=True)
        da_l = np.dot(_parameters['W' + str(_l)].T, dz_l)

        _grads['da' + str(_l - 1)] = da_l
        _grads['dW' + str(_l)] = dw_l
        _grads['db' + str(_l)] = db_l

    return _grads


# 参数更新

# 梯度下降法
def update_parameters(_parameters, _grads, _learning_rate):
    """
    _parameters: 参数列表，包括神经网络的基本信息
    _grads: 包含了各个变量梯度的字典
    update_method: 优化方法，默认为None
    _i: 调用函数时进行的迭代次数(仅针对Adam算法)
    return: 更新后的_parameters
    """

    _L = _parameters['L']

    for _l in range(1, _L + 1):
        _parameters['W' + str(_l)] = _parameters['W' + str(_l)] - _learning_rate * _grads['dW' + str(_l)]
        _parameters['b' + str(_l)] = _parameters['b' + str(_l)] - _learning_rate * _grads['db' + str(_l)]

    return _parameters


# 动量梯度下降
def update_parameters_Momentum(_parameters, _grads, _v_grads, _learning_rate, _beta=0.05):
    """
    _parameters: 参数列表，包括神经网络的基本信息
    _grads: 包含了各个变量梯度的字典
    _v_grads: 。。。
    update_method: 优化方法，默认为None
    _i: 调用函数时进行的迭代次数(仅针对Adam算法)
    return: 更新后的_parameters
    """

    _L = _parameters['L']

    for _l in range(1, _L + 1):

        _v_grads['dW' + str(_l)] = _beta * _v_grads['dW' + str(_l)] + (1 - _beta) * _grads['dW' + str(_l)]
        _v_grads['db' + str(_l)] = _beta * _v_grads['db' + str(_l)] + (1 - _beta) * _grads['db' + str(_l)]

        _parameters['W' + str(_l)] = _parameters['W' + str(_l)] - _learning_rate * _v_grads['dW' + str(_l)]
        _parameters['b' + str(_l)] = _parameters['b' + str(_l)] - _learning_rate * _v_grads['db' + str(_l)]

    return _parameters, _v_grads


# RMS prop
def update_parameters_RMS(_parameters, _grads, _s_grads, _learning_rate, _beta=0.999, _epsilon=1e-8):
    """
    _parameters: 参数列表，包括神经网络的基本信息
    _grads: 包含了各个变量梯度的字典
    _s_grads: 。。。
    update_method: 优化方法，默认为None
    _i: 调用函数时进行的迭代次数(仅针对Adam算法)
    return: 更新后的_parameters
    """

    _L = _parameters['L']

    for _l in range(1, _L + 1):

        _s_grads['dW' + str(_l)] = _beta * _s_grads['dW' + str(_l)] + (1 - _beta) * np.power(_grads['dW' + str(_l)], 2)
        _s_grads['db' + str(_l)] = _beta * _s_grads['db' + str(_l)] + (1 - _beta) * np.power(_grads['db' + str(_l)], 2)

        _parameters['W' + str(_l)] = (_parameters['W' + str(_l)] -
                                      _learning_rate * _grads['dW' + str(_l)] /
                                      (_epsilon + np.sqrt(_s_grads['dW' + str(_l)])))

        _parameters['b' + str(_l)] = (_parameters['b' + str(_l)] -
                                      _learning_rate * _grads['db' + str(_l)] /
                                      (_epsilon + np.sqrt(_s_grads['db' + str(_l)])))

    return _parameters, _s_grads


# Adam算法
def update_parameters_Adam(_parameters, _grads, _v_grads, _s_grads, _learning_rate,
                           _t, _beta1=0.9, _beta2=0.999, _epsilon=1e-8):
    """
    _parameters: 参数列表，包括神经网络的基本信息
    _grads: 包括了各个变量梯度的字典
    _v_grads:
    _s_grads:
    _t: 表示当前处于第t次迭代中
    _beta1: 动量梯度下降中的参数
    _beta2: RMS prop中的参数
    _epsilon: 极小量，用于避免除以0的情形
    return: 更新后的_parameters
    """

    _L = _parameters['L']

    for _l in range(1, _L + 1):

        _v_grads['dW' + str(_l)] = _beta1 * _v_grads['dW' + str(_l)] + (1 - _beta1) * _grads['dW' + str(_l)]
        _v_grads['db' + str(_l)] = _beta1 * _v_grads['db' + str(_l)] + (1 - _beta1) * _grads['db' + str(_l)]

        _s_grads['dW' + str(_l)] = (_beta2 * _s_grads['dW' + str(_l)] +
                                    (1 - _beta2) * np.power(_grads['dW' + str(_l)], 2))
        _s_grads['db' + str(_l)] = (_beta2 * _s_grads['db' + str(_l)] +
                                    (1 - _beta2) * np.power(_grads['db' + str(_l)], 2))

        _v_dwc = _v_grads['dW' + str(_l)] / (1 - np.power(_beta1, _t+1))
        _v_dbc = _v_grads['db' + str(_l)] / (1 - np.power(_beta1, _t+1))

        _s_dwc = _s_grads['dW' + str(_l)] / (1 - np.power(_beta2, _t+1))
        _s_dbc = _s_grads['db' + str(_l)] / (1 - np.power(_beta2, _t+1))

        _parameters['W' + str(_l)] = (_parameters['W' + str(_l)] - _learning_rate *
                                      (_v_dwc / (np.sqrt(_s_dwc) + _epsilon)))

        _parameters['b' + str(_l)] = (_parameters['b' + str(_l)] - _learning_rate *
                                      (_v_dbc / (np.sqrt(_s_dbc) + _epsilon)))

    return _parameters, _v_grads, _s_grads


def initialize_velocity(_parameters):

    _L = _parameters['L']
    v = {}
    for _l in range(_L):
        v['dW' + str(_l + 1)] = np.zeros(_parameters['W' + str(_l + 1)].shape)
        v['db' + str(_l + 1)] = np.zeros(_parameters['b' + str(_l + 1)].shape)

    return v


# 整合上述函数搭建神经网络
def deep_nn_model(_X, _y, _learning_rate, _iteration, _activation='sigmoid', predict='sigmoid', update='None'):

    # 1、构建神经网络
    _layer_dims = initial_layer(_y)

    # 2、初始化参数
    # _X = Normalization(_X)  # 归一化处理
    _parameters = initial_parameters(_X, _layer_dims)
    _v_grads = initialize_velocity(_parameters)
    _s_grads = initialize_velocity(_parameters)
    _alpha = _learning_rate

    # 3、循环训练
    for t in range(_iteration):
        # 3.1、正向传播
        (_AL, _caches) = forward_propagation(_X, _parameters, _activation, predict)
        _learning_rate = _alpha / (1 + t/5000)

        # 3.2、计算代价函数
        if predict == 'sigmoid':
            _cost = cost_sigmoid(_AL, _y, _parameters, _lambda=0.0)
        elif predict == 'SoftMax':
            _cost = cost_SoftMax(_AL, _y, _parameters, _lambda=0.0)

        if t % 10 == 0:
            print("第"+str(t)+"次迭代后的cost："+str(_cost))

        # 3.3、逆向传播
        _grads = backward_propagation(_AL, _y, _parameters, _caches, _lambda=0.0, activation=_activation)

        # 3.4、更新参数
        if update == 'None':
            _parameters = update_parameters(_parameters, _grads, _learning_rate)

        elif update == 'Momentum':
            _parameters, _v_grads = update_parameters_Momentum(_parameters, _grads, _v_grads, _learning_rate)

        elif update == 'RMS':
            _parameters, _s_grads = update_parameters_RMS(_parameters, _grads, _s_grads, _learning_rate)

        elif update == 'Adam':
            _parameters, _v_grads, _s_grads = update_parameters_Adam(_parameters, _grads,
                                                                     _v_grads, _s_grads, _learning_rate, t)

    return _parameters


# 精度检验
def accuracy(_AL, _y):
    _predict = np.argmax(_AL, axis=0, keepdims=True) + 1
    _predict = np.reshape(_predict, (np.size(_predict), -1))
    _y = np.reshape(_y, (np.size(_y), -1))

    _error = _predict - _y
    count = 0

    for i in _error:
        if i == 0:
            count += 1

    accurate = 100 * count / np.size(_predict)

    print("模型的精度为：" + str(accurate) + "%")


# learning_rate = 0.092
# iteration = 5000
# parameters = deep_nn_model(X_train, y_train_OneHot, learning_rate, iteration, update='Adam')
# X_train_after, _ = forward_propagation(X_train, parameters, activation='sigmoid')
# accuracy(X_train_after, y_train)
# X_check_after, _ = forward_propagation(X_check, parameters, activation='sigmoid')
# accuracy(X_check_after, y_check)
# print(X.type)
