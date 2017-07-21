# %load mnist_loader.py
"""
mnist_loader
~~~~~~~~~~~~
A library（库） to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    """Return the MNIST data as a tuple(元组) containing the training data（训练数据）,
    the validation data（确认数据）, and the test data（测试数据）.
    The ``training_data`` is returned as a tuple（元组） with two entries（词目）.
    The first entry contains the actual（真实的） training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit（数字值）
    values (0...9) for the corresponding（相应的，一致的） images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    This is a nice data format（数据格式）, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
   （虽然这是一种好的数据格式，但是为了能在神经网络中使用我们还需要对其进行一定的修改，具体的执行方法请参见load_data_wrapper函数）
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('mnist.pkl.gz', 'rb')  # 由于文件使用.gz的压缩形式保存，所以先要将pkl类型的数据从gz文件中解压缩出来
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()  # 读取plk文件中的数据，并将其返回。
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    （基于load_data方法，但是相比于load_data方法load_data_wrapper方法的数据形式更加适用于神经网络的操作）
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional（10维数组）
    numpy.ndarray representing the unit vector（单位向量） corresponding to the
    correct digit（正确的数字） for ``x``.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional（784维数组）
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e.（拉丁）, the digit values (integers整数)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code.
    （很明显，这表明我们在处理training data和validation/test 数据时数据格式有轻微的不同，这个数据格式的结果对于处理神经网络的问题更加方便）"""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    """Return a 10-dimensional unit vector(单位矢量) with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
