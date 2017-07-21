# %load network.py

"""
network.py
~~~~~~~~~~
IT WORKS

A module to implement（实现） the stochastic（随机） gradient descent learning
algorithm（算法） for a feedforward（前馈） neural network.  Gradients（梯度） are calculated
using backpropagation（反向传播）.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized（最佳化的）,
and omits（忽略、删除、遗漏） many desirable features（不错的特性，更好的方案）.

"""


#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np


class Network(object):
    def __init__(self, sizes):
        #  注意这里的self指的是类的实例对象的self
        """__init__初始类函数，传入第一个参数含有self
        The list ``sizes`` contains the number of neurons in the
        respective（各个） layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.
        Note that the first layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers.
        注意第一层被假定为输入层并且按照惯例我们不对第一层的神经元设置偏置，因为偏置仅在后面的层中用于计算输出
        """
        self.num_layers = len(sizes)  # 假设[2, 3, 1]则len(sizes)=3
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        """
        np.random.randn(size)所谓标准正太分布（μ=0, σ=1），
        对应于np.random.normal(loc=0, scale=1, size)
        sizes[1:]表示取第二个值到最后一个值的所有值sizes[:-1]则是表示从第一个值到倒数第二个值的所有值
        当sizes取值为[2,3,1]时
        self.biases是一个一维两行的列表，并且第一行是一个3行1列的二维数组，第二行是一个1行1列的二维数组
       [array([[ 0.06148192],
       [ 0.28541901],
       [ 0.0796364 ]]), array([[ 0.19690658]])]
       当sizes取值为[3,4,5]时
       self.biases是一个一维两行的列表，并且第一行是一个4行1列的二维数组，第二行是一个5行1列的二维数组
       [array([[ 0.85292893],
       [ 1.60514696],
       [ 2.4742093 ],
       [-0.71263092]]), array([[ 0.77120836],
       [-1.15710001],
       [ 1.12458714],
       [-0.36316267],
       [ 0.730775  ]])]
       对于sizes=[2, 3, 1],weights而言x取[2,3],y取[3, 1]所以数组的形式是[3, 2],[1, 3]
       [array([[ 0.11734692, -0.60634616],
       [ 0.31558075,  0.29302227],
       [-0.31596948, -0.77749944]]), array([[ 1.08608315,  0.09010017,  0.47932846]])]
        """
    #对每个Network类添加一个feedforward方法，对于网络中给定的一个输入a,返回对应的输出
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    # stochastic（随机） gradient descent learning algorithm（算法）
    #  随机梯度下降算法
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples（元组）
        ``(x, y)`` representing the training inputs（训练输入） and the desired
        outputs（期望输出）.  The other non-optional（不可选参数）parameters are
        self-explanatory（不需要说明的）.  If ``test_data`` is provided then the
        network will be evaluated（评估） against the test data after each
        epoch（迭代器周期）, and partial progress（部分过程） printed out.  This is useful for
        tracking progress（追踪进度）, but slows things down substantially（相当）."""
        """training_data是一个（x，y）元祖的列表，表示训练输入和其对应的期望输出。
        变量epochs表示迭代期数量，mini_batch_size表示采样时小批量数据大小，eta是学习速率"""
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        """在每个迭代器，首先随机的将训练数据打乱，然后将他分成多个适当大小的批量数据，是一个简单的从训练数据的随机采样方法
        对每一个mini_batch我们使用update_mini_batch方法应用一次梯度下降"""
        for j in range(epochs):
            random.shuffle(training_data)  # 将所有数据重新排序
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]  # 将其以mini_batch_size为单位分成小块
            for mini_batch in mini_batches:  # 对于mini_batches中的数据我们用mini_batch进行遍历
                self.update_mini_batch(mini_batch, eta)  # 使用单词梯度下降的迭代更新网络的权重和偏置
            if test_data:
                print("Epoch {} : {} / {}".format(j, self.evaluate(test_data), n_test))
                #  如果有test_data,输出训练批次，预测正确的个数，训练数据长度
            else:
                print("Epoch {} complete".format(j))
                #  如果没有test_data，输出训练批次

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases（不断更新biases和weights） by applying
        gradient descent（梯度下降） using backpropagation（反向传播、一种快速计算代价函数的梯度的方法） to a single mini batch.
        The ``mini_batch`` is a list of tuples（元组） ``(x, y)``, and ``eta``
        is the learning rate（学习率）.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # 累加器
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # 累加器
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)  # 反向传播算法，快速计算代价函数的梯度的方法
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        """weights=self.weight(eta/len(mini_batch))*nabla_w"""
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x（描述代价函数的梯度）.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward 前馈
        activation = x
        #  note:mini_batch实际上是和training_data中数组数据格式一致的小批量数据，x：表示图片信息，y:表示正确答案
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])  # 计算输出误差
        nabla_b[-1] = delta  # BP3 ，得到输出层的b误差等于输出误差
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())  # BP4，得到输出层的w误差等于输出误差和上一层的输出值
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.
        # （注意：下面循环中的变量l和第二章中对变量l的用法有些不同，这里的变量l=1指的是神经网络的最后一层，l=2指的是网络的倒数第二层）
        #  Here, l = 1 means the last layer of neurons, l = 2 is the second-last layer, and so on.
        #  It's a renumbering of the scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.这是书中对该计划的一种重新考虑，它利用了Python可以在列表中使用负索引的事实。
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result（返回test输入中通过神经网络预测正确的个数）.
        Note that the neural network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.
        （注意神经网络的输出假定是最后一层有最高的激活值的索引）"""
        test_results = [(np.argmax(self.feedforward(x)), y)  # np.argmax返回的是最值的索引，这里的y值输入图像的标准值
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations.
        返回cost函数对输出值的偏导数"""
        return (output_activations - y)


#### Miscellaneous functions
#  指定特定的激活函数，此处为逻辑函数即S型函数
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function.
    sigmoid函数的导数"""
    return sigmoid(z) * (1 - sigmoid(z))
