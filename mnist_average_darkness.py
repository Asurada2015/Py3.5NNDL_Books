"""
mnist_average_darkness
~~~~~~~~~~~~~~~~~~~~~~

A naive classifier for recognizing handwritten digits(识别手写数字) from the MNIST
data set.  The program classifies digits based on how dark they are
--- the idea is that digits like "1" tend to be less dark than digits
like "8", simply because the latter has a more complex shape.  When
shown an image the classifier returns whichever digit in the training
data had the closest average darkness.

The program works in two steps: first it trains the classifier（分类器）, and
then it applies the classifier to the MNIST test data to see how many
digits are correctly classified.

Needless to say, this isn't a very good way of recognizing handwritten
digits!  Still, it's useful to show what sort of performance we get
from naive ideas.虽然想法很简单，但是其实得到的效果还不错"""

#### Libraries
# Standard library（标准库）
from collections import defaultdict

# My libraries
import mnist_loader


def main():
    training_data, validation_data, test_data = mnist_loader.load_data()
    # training phase: compute the average darknesses for each digit,
    # 培训阶段：基于训练数据计算每个数字的平均暗度值
    # based on the training data
    avgs = avg_darknesses(training_data)
    # testing phase: see how many of the test images are classified
    # correctly
    num_correct = sum(int(guess_digit(image, avgs) == digit)
                      for image, digit in zip(test_data[0], test_data[1]))
    """image表示测试图片数据，digit表示测试图片数据的真实值值，avgs是通过训练数据得到的每个数字的平均灰度值
    通过guess_digit方法传图image和avgs返回一个猜测的int类型的数字，如果和真实的数字相等给的话就说明猜测正确。
    然后通过sum函数将正确的图片数统计出来
    """
    print("Baseline classifier using average darkness of image.")
    print("{0} of {1} values correct.".format(num_correct, len(test_data[1])))
    # test_data[1]是一个一维np.ndarray对象，通过求其数组的长度可以知道一共有多少个测试数据


def avg_darknesses(training_data):
    """ Return a defaultdict（默认字典） whose keys are the digits 0 through 9.
    For each digit we compute a value which is the average darkness of
    training images containing that digit（对于每一个数字，我们对于训练数据的该数字计算一个平均灰度值）.
    The darkness for any particular image is just the sum of the darknesses for each pixel.
    （每一个图片的灰度值就是图片中每个像素的的灰度值之和）"""
    digit_counts = defaultdict(int)
    darknesses = defaultdict(float)
    for image, digit in zip(training_data[0], training_data[1]):
        digit_counts[digit] += 1
        darknesses[digit] += sum(image)
    avgs = defaultdict(float)
    for digit, n in digit_counts.items():
        avgs[digit] = darknesses[digit] / n
    return avgs

def guess_digit(image, avgs):
    """Return the digit whose average darkness in the training data is
    closest to the darkness of ``image``.(返回training_data中的灰度值和图片的灰度值最相近的数字值)
     Note that ``avgs`` is assumed to be a defaultdict whose keys are 0...9, and whose values
    are the corresponding average darknesses across the training data.
    （注意avgs是一个defaultdict的数据格式并且关键字是0~9的数字，values是根据training data算出的平均灰度）"""
    darkness = sum(image)  # 传入图片并计算其灰度总和
    distances = {k: abs(v-darkness) for k, v in avgs.items()}
    # distances是一个关于int k 和关键字灰度值差值的字典
    """
    avgs :  defaultdict(<class 'float'>, {0: 135.8691200324412, 1: 59.660347503522367, 2: 116.29430385718599, 
    3: 110.87092451235051, 4: 95.220739703385476, 5: 100.34646166222814, 6: 107.38312020298929, 7: 89.878834541062801,
    8: 117.88056410703221, 9: 96.045558559292303})
    """
    return min(distances, key=distances.get)  # 获得abs(v-darkness)值最小的k值，返回的是一个int类型的数

if __name__ == "__main__":
    main()
