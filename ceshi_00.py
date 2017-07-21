import numpy as np
import pickle
import gzip

# sizes = [2, 3, 1]
# # sizes = [3, 4, 5]
# biases = [np.random.randn(y, 1) for y in sizes[1:]]
# # print(biases)
# a = sizes[1:]
# b = sizes[:-1]
# print(a)
# print(b)
# print(a == b)
# # result
# # [4, 5]
# # [3, 4]
# # False
# # print(np.random.randn(3, 1))
# weights = [np.random.randn(y, x)
#                 for x, y in zip(sizes[:-1], sizes[1:])]
# print(weights)
# a = np.arange(6272).reshape((8, 784))
# b = np.reshape(a, (781, 1))  # total size of new array must be unchanged
# print(a)
"""测试MNIST数据库中数据的保存格式"""


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
f = gzip.open('mnist.pkl.gz', 'rb')  # 由于文件使用.gz的压缩形式保存，所以先要将pkl类型的数据从gz文件中解压缩出来
training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
print(training_data[0].shape)
training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
print(training_inputs[0].shape)
print(training_data[1].shape)
print(training_data[1])
training_results = [vectorized_result(y) for y in training_data[1]]
training_data = zip(training_inputs, training_results)
# print(training_data)
"""现在想要知道training_data zip中的数据格式"""
sum = 0
for each in training_data:
    sum = sum+1
    if sum % 50000 == 0:
        print(each)
validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
print(validation_data[0].shape)
test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
print(test_data[0].shape)
f.close()  # 读取plk文件中的数据，并将其返回。

