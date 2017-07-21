"""network2.py
~~~~~~~~~~~~~~

An improved version of network.py, implementing the stochastic
gradient descent learning algorithm（实现随机梯度下降学习算法） for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function(改进包括增加了交叉熵cost方法),
regularization(规则化), and better initialization of network weights（更好的网络权值初始化）. Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.(没有经过优化，并且省略了很多激动人心的功能)

"""

#### Libraries
# Standard library
import json
import random
import sys

# Third-party libraries
import numpy as np


#### Define the quadratic and cross-entropy cost functions
#  定义代价函数

class QuadraticCost(object):
    """二次代价函数方法"""
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.
        返回关于网络输出a和目标输出y的二次代价函数的直接计算结果.
        """
        return 0.5*np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.返回基于二次代价函数的误差表达式"""

        return (a - y)*sigmoid_prime(z)


class CrossEntropyCost(object):
    """交叉熵方法这里的fn()和delta()是静态方法，这些方法不依赖于对象，所以不需要传入self参数"""
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output（期望输出）
        ``y``.  Note that np.nan_to_num is used to ensure numerical stability.
        （注意np.nan_to_num用于将极大值或者极小值NaN即一个不是num的值转化为num）
        In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        (特别是如果y和a在同一个位置上都取1的话得出的结果是nan,如果使用nan_to_num函数的话则可以正常显示是(0.0))
        """
        return np.sum(np.nan_to_num(-y*np.log(a) - (1 - y)*np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        """
        输出层误差方程BP1，链式法则
        Return the error delta（误差率） from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.
        为了使接口与其他cost类的delta方法一致，它被包含在方法的参数中。

        """
        return (a - y)


#### Main Network class
class Network(object):
    def __init__(self, sizes, cost=CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.（size列表包含了网络每层中神经元的数量）  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).

        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        """
        改进化的权值和偏置的初始化方法，weights的标准差为1/sqrt(x)
        Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation（标准差） 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.
        （注意第一层是神经网络的输入层，按照惯例我们不对输入层设置偏置）
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """普通方法初始化weights和biases
        Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.(通过平均值为0，标准差为1的高斯正太分布初始化biases和weights)

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.(第一层被认为是输入层，我们不对其设置biases)

        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison(为了方便比较，我们和第一章使用同样的初始化器).
        It will usually be better to use the default weight initializer instead.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            early_stopping_n=0):  # 提前结束
        """Train the neural network using mini-batch stochastic gradient
        descent. (用小批量数据随机梯度下降法训练数据) The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters（不可选的参数）are self-explanatory（不需说明的）, as is the
        regularization parameter ``lmbda``（这其中包括有正规化参数lmbda）.  The method also accepts
        ``evaluation_data``(这个模式也可以接受评估数据), usually either the validation or test（一般是确认数据或者是测试数据）
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        （通过设置合适的标记我们可以监视不管是训练数据还是评估数据的cost值和准确度）
        appropriate flags.

        The method returns a tuple containing fourlists（这个模式返回一个包含四个列表的元组）:
        the (per-epoch) costs on the evaluation data（平均每一批次数据在估计数据的costs）,
        the accuracies on the evaluation data(在估计数据上的精确值),
        the costs on the training data,(平均每一批次数据在估计数据的costs)
        and the accuracies on the training data.(在训练数据上的精确值)
        All values are evaluated at the end of each training epoch.所有的值都会在每一批次的训练数据的最后被计算）
        So, for example,
        if we train for 30 epochs（如果我们训练30批次）, then the first element of the tuple（那么元祖的第一个元素）
        will be a 30-element list（30个元素的列表） containing the cost on the evaluation data at the end of each epoch.
        （包含了每一组预测数据的末尾的cost值）
        Note that the lists are empty if the corresponding flag is not set.
        (如果相应的标记没有被设置的话则列表会被设置成为空)
        """

        # early stopping functionality:
        best_accuracy = 1  # 最好的精度

        training_data = list(training_data)
        n = len(training_data)  # 训练数据量，注意这个n值在整个类中都可以使用，在其他的函数汇总也可以使用

        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)  # 评价数据数据量

        # early stopping functionality:
        best_accuracy = 0
        no_accuracy_change = 0

        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))

            print("Epoch %s training complete"%j)

            if monitor_training_cost:  # 是否显示training_data的cost,这里使用的是正规化方法，lmbda指的是正规化参数
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data))

            # Early stopping:
            if early_stopping_n > 0:
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    no_accuracy_change = 0
                    # print("Early-stopping: Best so far {}".format(best_accuracy))
                else:
                    no_accuracy_change += 1

                if (no_accuracy_change == early_stopping_n):
                    # print("Early-stopping: No accuracy change in last epochs: {}".format(early_stopping_n))
                    return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy
            """默认没有启动early stopping
            early_stopping_n表示的是数据未发生变化的次数，设置best_accuracy,如果当前的学习准确率大于best_accuracy
            no_accuracy_change开始计数，如果当前的准确率开始下降并且no_accuracy_change+1，
            当no_accuracy_change到达一定的次数时提前终止函数
            """
        return evaluation_cost, evaluation_accuracy, \
               training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter（规则化参数）, and
        ``n`` is the total size of the training data set.

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1 - eta*(lmbda/n))*w - (eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass 反向传播调用cost类的delta方法，BP1计算其输出值的梯度
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        """反向传播"""
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):  # num_layers指的是网络的层数值
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta)*sp
            """从后往前依次计算每层的b和w"""
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result.（输出data中正确预测的数量） The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.（神经网络的输出被指定为输出层神经元中有最大激励值的index序号）

        The flag ``convert`` should be set to False if the data set is validation or test data (the usual case)
        （通常情况下convert被指定为false,并且此时的数据是验证数据或预测数据）, and to True if the data set is the training data.
        The need for this flag arises due to differences in the way the results ``y`` are represented
        in the different data sets. （这个flag的设置是根据y值的表示不同进行表示的） In particular, it
        flags whether we need to convert between the different representations.
        It may seem strange to use different representations for the different data sets.
        Why not use the same representation for all three data sets?
        It's done for efficiency reasons（效率的原因）
        -- the program usually evaluates the cost on the training data and the accuracy on other data sets.
        （程序常常评价在训练数据上的cost,以及在其他数据集上的精确度）
        These are different types of computations, and using different representations speeds things up.
        （这是不同的计算并且用了不同的方式使速度加快）
        More details on the representations can be found in mnist_loader.load_data_wrapper.
        （更多的表示的细节可以看mnist_loader,load_data_wrapper）
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]  # np.argmax方法返回元组中最大值的索引
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                       for (x, y) in data]
        """对于training_data来说y中存的是一维元组的列表，对于test_data来说list中存储的是0~9的数据"""

        result_accuracy = sum(int(x == y) for (x, y) in results)
        return result_accuracy

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case通常情况),
        and to True if the data set is the validation or test data.（如果是验证数据或者是测试数据时，convert设置为True）
        See comment(注解，解释) on the similar (but
        reversed相反的) convention（约定） for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)  # 如果是测试数据或者是验证数据时，对y要进行增加维度的操作
            cost += self.cost.fn(a, y)/len(data)
            #普通方法
            # 例如这里传入的是mini_batch_data数据则len(data)是mini_batch中的长度。因为要计算的是平均值
            cost += 0.5*(lmbda/len(data))*sum(
                np.linalg.norm(w) ** 2 for w in self.weights)  # '**' - to the power of.
            #对cost函数进行L2规范化操作
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}  # 通过load函数我们得知其实这里的cost中保存的是交叉熵cost这个类型的名字
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    # 通过列表解开后，我们会发现weights和biases是np.array属性的数据
    return net


#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1 - sigmoid(z))
