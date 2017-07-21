import numpy as np
import pickle
import gzip
from collections import defaultdict
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data()
"""对于利用图像灰度的例子想要知道image和digit的数据格式"""
digit_counts = defaultdict(int)
darknesses = defaultdict(float)
a = 0
# print(list(zip(training_data[0], training_data[1])))  本来想要打印出整个zip中的结构但是由于数据量太大的缘故~
# print(training_data[0].count)  AttributeError: 'numpy.ndarray' object has no attribute 'count'
# 证明training_data[0]和training_data[1]是属于ndarray数据格式并且分别是（50000，,784）的二维数组，（50000,0）的一维数组
# print(training_data[0])
# print(training_data[1])
"""想要研究zip函数对于np二维矩阵到底做了什么以及image和digit的数据格式
我们假设一个实验"""
# a1 = np.arange(18).reshape(6, 3)
# a2 = np.arange(6)
# print('a1:  ', a1)
# print('a2:  ', a2)
# print('list(zip()):  ', list(zip(a1, a2)))
"""
a1:   
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]
 [12 13 14]
 [15 16 17]]
a2:   [0 1 2 3 4 5]
list(zip()):   [(array([0, 1, 2]), 0), (array([3, 4, 5]), 1), (array([6, 7, 8]), 2), (array([ 9, 10, 11]), 3), 
(array([12, 13, 14]), 4), (array([15, 16, 17]), 5)]
"""
for image, digit in zip(training_data[0], training_data[1]):  # for语句的迭代器仅仅是拆开第一个括号部分
    #  a = a+1
    # if a % 50000 == 0:
    #     print('image:  ', image, '\n', 'the dimension of the image', np.shape(image))
    #     print('digit:  ', digit, '\n', 'the dimension of the image', np.shape(digit))
    """ 省略表示一张图片784个像素点灰度的数组
    image,digit对象都是np.darray类型的数组
    the dimension of the image (784,)
    digit:   8 
    the dimension of the image ()"""
    digit_counts[digit] += 1  # 表示图片库中一个数字值得图片出现的张数
    darknesses[digit] += sum(image)  # 图片库中一个数字表示的图片的总的灰度值
    """其中digit_counts和darknesses都是defaultdict类型的数据"""
print(digit_counts)
print(darknesses)
"""defaultdict(<class 'int'>, {0: 4932, 1: 5678, 2: 4968, 3: 5101, 4: 4859, 5: 4506, 6: 4951, 7: 5175, 8: 4842, 9: 4988})
   defaultdict(<class 'float'>, {0: 670106.5, 1: 338751.453125, 2: 577750.1015625, 3: 565552.5859375, 4: 462677.57421875, 
   5: 452161.15625, 6: 531653.828125, 7: 465122.96875, 8: 570777.69140625, 9: 479075.24609375})"""
print('digit_counts.value:  ', digit_counts.values())  # 注意如果实例中出现的参数可以使用.（self）的函数则是调用自身不用传入任何参数的空函数
print('digit_counts.items:  ', digit_counts.items())
print('digit_counts.keys:   ', digit_counts.keys())
print('digit_counts.__sizeof__:  ', digit_counts.__sizeof__())
print('digit_counts.__str__:  ', digit_counts.__str__())
print('digit_counts.__len__:  ', digit_counts.__len__())
"""
digit_counts.value:   dict_values([4932, 5678, 4968, 5101, 4859, 4506, 4951, 5175, 4842, 4988])
digit_counts.tems:   dict_items([(0, 4932), (1, 5678), (2, 4968), (3, 5101), (4, 4859), (5, 4506), (6, 4951), (7, 5175), (8, 4842), (9, 4988)])
digit_counts.keys:    dict_keys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
digit_counts.__sizeof__:   456
digit_counts.__str__:   defaultdict(<class 'int'>, {0: 4932, 1: 5678, 2: 4968, 3: 5101, 4: 4859, 5: 4506, 6: 4951, 7: 5175, 8: 4842, 9: 4988})
digit_counts.__len__:   10
"""
avgs = defaultdict(float)
for digit, n in digit_counts.items():
    avgs[digit] = darknesses[digit] / n  # 用一个数字灰度的总和处以数字在图片集合中出现的总的次数
print('avgs : ', avgs)
"""
avgs :  defaultdict(<class 'float'>, {0: 135.8691200324412, 1: 59.660347503522367, 2: 116.29430385718599, 
3: 110.87092451235051, 4: 95.220739703385476, 5: 100.34646166222814, 6: 107.38312020298929, 7: 89.878834541062801, 
8: 117.88056410703221, 9: 96.045558559292303})
"""
print(min(avgs, key=avgs.get))  # 通过dict字典类型的get方法可以获得dict.get(key)该key的value的值
