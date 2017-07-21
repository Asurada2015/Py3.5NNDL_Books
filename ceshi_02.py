import numpy as np
"""对于np.dot的验证"""
# #  点积 对于一维数组相当于内积，对于二维数组相当于对应矩阵的乘积
# x = np.array([[1, 2], [3, 4]])
# y = np.array([[5, 6], [7, 8]])
# z = (np.dot(x, y))
# print(x.dot(y))
# print(z)
# """
# [[19 22]
#  [43 50]]
# [[19 22]
#  [43 50]]
# """
# xa = np.array([2, 3])
# ya = np.array([4, 5])
# print(xa.dot(ya))  # 2*4+3*5=23
"""关于查验一个py文档所提供的接口"""
import network2
help(network2.Network.SGD)