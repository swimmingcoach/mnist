import numpy as np
import tensorflow as tf
import cv2
import os
import random
import logger

CHAR_SET = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# 验证码图片的存放路径
CAPTCHA_IMAGE_PATH = 'E:/Tensorflow/captcha/images/'
# 存放训练好的模型的路径
MODEL_SAVE_PATH = 'E:/Tensorflow/captcha/models/'

# 验证码图片的宽度
CAPTCHA_IMAGE_WIDHT = 160
# 验证码图片的高度
CAPTCHA_IMAGE_HEIGHT = 60
CAPTCHA_LEN = 4
batch_size = 64
CHAR_SET_LEN = len(CHAR_SET)


def get_train_data(data_dir=CAPTCHA_IMAGE_PATH):
    simples = {}
    for file_name in os.listdir(data_dir):
        captcha = file_name.split('.')[0]
        simples[data_dir + '/' + file_name] = captcha
    return simples


simples = get_train_data(CAPTCHA_IMAGE_PATH)
file_simples = list(simples.keys())
num_simples = len(simples)


def get_next_batch():
    # 训练输入X， 64 X 160*60
    batch_x = np.zeros([batch_size, CAPTCHA_IMAGE_WIDHT * CAPTCHA_IMAGE_HEIGHT])
    # 训练输入Y， 64 X 10*4
    batch_y = np.zeros([batch_size, CHAR_SET_LEN * CAPTCHA_LEN])

    # 随机取出64个训练数据
    for i in range(batch_size):
        file_name = file_simples[random.randint(0, num_simples - 1)]
        # 灰化图片
        batch_x[i, :] = np.float32(cv2.imread(file_name, 0)).flatten() / 255
        batch_y[i, :] = text2vec(simples[file_name])
    return batch_x, batch_y


# 将验证码转换为训练时用的标签向量，维数是 40
# 例如，如果验证码是 ‘0296’ ，则对应的标签是
# [1 0 0 0 0 0 0 0 0 0
#  0 0 1 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 1
#  0 0 0 0 0 0 1 0 0 0]
def text2vec(text):
    return [0 if ord(i) - 48 != j else 1 for i in text for j in range(CHAR_SET_LEN)]


# 构建卷积神经网络并训练
def train_data_with_cnn():
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    x = tf.placeholder(tf.float32, [None, CAPTCHA_IMAGE_WIDHT * CAPTCHA_IMAGE_HEIGHT], name='input')
    y_ = tf.placeholder(tf.float32, [None, CHAR_SET_LEN * CAPTCHA_LEN])
    x_image = tf.reshape(x, [-1, CAPTCHA_IMAGE_HEIGHT, CAPTCHA_IMAGE_WIDHT, 1])

    w_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    w_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    w_conv3 = weight_variable([5, 5, 64, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    w_fc1 = weight_variable([8 * 20 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool3_flat = tf.reshape(h_pool3, [-1, 8 * 20 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1) + b_fc1)

    w_fc2 = weight_variable([1024, CHAR_SET_LEN * CAPTCHA_LEN])
    b_fc2 = bias_variable([CHAR_SET_LEN * CAPTCHA_LEN])
    output = tf.add(tf.matmul(h_fc1, w_fc2), b_fc2)

    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=output))
    train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)

    predict = tf.reshape(output, [-1, CAPTCHA_LEN, CHAR_SET_LEN])
    labels = tf.reshape(y_, [-1, CAPTCHA_LEN, CHAR_SET_LEN])
    correct_prediction = tf.equal(tf.argmax(predict, 2, name='predict_max_idx'), tf.argmax(labels, 2))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    for i in range(50000):
        batch_x, batch_y = get_next_batch()
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y})
            logger.error("step %d, training accuracy %g " % (i, train_accuracy))
            if train_accuracy > 0.9999:
                saver.save(sess, MODEL_SAVE_PATH, global_step=i)
                break
        train_step.run(feed_dict={x: batch_x, y_: batch_y})


if __name__ == '__main__':
    logger.error('Training start')
    train_data_with_cnn()
    logger.error('Training finished')
