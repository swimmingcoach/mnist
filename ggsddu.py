import os
import tensorflow.examples.tutorials.mnist.input_data as input_data
import input_data as in_data
import tensorflow as tf
from mnist_demo import *

# 60000行的训练数据集（mnist.train）和10000行的测试数据集（mnist.test）
# 每一个MNIST数据单元有两部分组成：一张包含手写数字的图片和一个对应的标签。我们把这些图片设为“xs”，把这些标签设为“ys”。
# 训练数据集和测试数据集都包含xs和ys，比如训练数据集的图片是 mnist.train.images ，训练数据集的标签是 mnist.train.labels。

# 在MNIST训练数据集中，mnist.train.images 是一个形状为 [60000, 784] 的张量，第一个维度数字用来索引图片，第二个维度数字用来索引每张图片中的像素点。
# 在此张量里的每一个元素，都表示某张图片里的某个像素的强度值，值介于0和1之间。
# 相对应的MNIST数据集的标签是介于0到9的数字，用来描述给定图片里表示的数字。
# 为了用于这个教程，我们使标签数据是"one-hot vectors"。
# 一个one-hot向量除了某一位的数字是1以外其余各维度数字都是0。
# 所以在此教程中，数字n将表示成一个只有在第n维度（从0开始）数字为1的10维向量。
# 比如，标签0将表示成([1,0,0,0,0,0,0,0,0,0,0])。
# 因此， mnist.train.labels 是一个 [60000, 10] 的数字矩阵。
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist = in_data.read_data_sets("data/", one_hot=True)

# x不是一个特定的值，而是一个占位符placeholder，我们在TensorFlow运行计算时输入这个值。
# 我们希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。
# 我们用2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]。
# （这里的None表示此张量的第一个维度可以是任何长度的。）
x = tf.placeholder(tf.float32, [None, 784])

# 我们的模型也需要权重值和偏置量，当然我们可以把它们当做是另外的输入（使用占位符），但TensorFlow有一个更好的方法来表示它们：Variable 。
# 一个Variable代表一个可修改的张量，存在在TensorFlow的用于描述交互性操作的图中。
# 它们可以用于计算输入值，也可以在计算中被修改。
# 对于各种机器学习应用，一般都会有模型参数，可以用Variable表示。

# 我们赋予tf.Variable不同的初值来创建不同的Variable：在这里，我们都用全为零的张量来初始化W和b。
# 因为我们要学习W和b的值，它们的初值可以随意设置。
# 注意，W的维度是[784，10]，因为我们想要用784维的图片向量乘以它以得到一个10维的证据值向量，每一位对应不同数字类。
# b的形状是[10]，所以我们可以直接把它加到输出上面。
# 权重值
W = tf.Variable(tf.zeros([784, 10]))
# 偏置量
b = tf.Variable(tf.zeros([10]))

# tf.matmul(​​X，W)表示x乘以W
# 现在，我们可以实现我们的模型啦。只需要一行代码！
# 我们知道MNIST的每一张图片都表示一个数字，从0到9。
# 我们希望得到给定图片代表每个数字的概率。
# 比如说，我们的模型可能推测一张包含9的图片代表数字9的概率是80%但是判断它是8的概率是5%（因为8和9都有上半部分的小圆），然后给予它代表其他数字的概率更小的值。
#
# 这是一个使用softmax回归（softmax regression）模型的经典案例。
# softmax模型可以用来给不同的对象分配概率。
# 即使在之后，我们训练更加精细的模型时，最后一步也需要用softmax来分配概率。
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder("float", [None, 10])

# 计算交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

keep_prob = tf.placeholder("float")
dir_name = "test_num"
files = os.listdir(dir_name)
cnt = len(files)
for i in range(cnt):
    files[i] = dir_name + "/" + files[i]
    test_images1, test_labels1 = GetImage([files[i]])
    mnist.test = in_data.DataSet(test_images1, test_labels1, dtype=tf.float32)
    res = y.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    # res=h_pool2.eval()
    # res=accuracy.eval({x: mnist.test.images, y_: mnist.test.labels})
    # print("output:",int(res[0]))
    # print_matrix(res)
    print("output:", res)
