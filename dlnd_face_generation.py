# coding: utf-8
import math
import matplotlib.pyplot as plt
from glob import glob
import tensorflow as tf
import numpy as np
from PIL import Image


def scale(x, feature_range=(-1, 1)):
    # scale to (0, 1)
    x = ((x - x.min())/(255 - x.min()))

    # scale to feature_range
    Min, Max = feature_range
    x = x * (Max - Min) + Min
    return x


def get_batches(dataset, batch_size):
    index = 0
    while index != -1:
        if index + batch_size < dataset.shape[0]:
            next_index = index + batch_size
        else:
            next_index = -1
        yield scale(dataset[index:next_index])
        index = next_index


def read_pic(images_path):
    data_batch = np.array([np.array(Image.open(image_path)) for image_path in images_path])
    return data_batch


def images_square_grid(images, mode):
    length = int(math.floor(np.sqrt(images.shape[0])))
    show_image = Image.new(mode, (images.shape[1] * length, images.shape[2] * length))
    for i in range(length ** 2):
        image = Image.fromarray(images[i], mode)
        col = i // length
        row = i % length
        show_image.paste(image, (col * images.shape[1], row * images.shape[2]))
    return show_image


def model_inputs(image_width, image_height, image_channels, z_dim):
    inputs_real = tf.placeholder(tf.float32,
                                 shape=(None, image_width, image_height, image_channels), name='inputs_real')
    inputs_z = tf.placeholder(tf.float32, shape=(None, z_dim), name='inputs_z')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    return inputs_real, inputs_z, learning_rate


def discriminator(images, reuse=False, alpha=0.2, drop_rate=0., size_mult=32):
    with tf.variable_scope('discriminator', reuse=reuse):
        # input 96 * 96 * 3
        # images = tf.layers.dropout(images, rate=drop_rate/2.5)
        
        # layer1 48 * 48 * 64
        layer_1 = tf.layers.conv2d(images, size_mult, 5, strides=2, padding='same')
        layer_1 = tf.maximum(alpha * layer_1, layer_1)
        # layer_1 = tf.layers.dropout(layer_1, rate=drop_rate)
        
        # layer2 24 * 24 * 128
        layer_2 = tf.layers.conv2d(layer_1, 2 * size_mult, 5, strides=2, padding='same')
        layer_2 = tf.layers.batch_normalization(layer_2, training=True)
        layer_2 = tf.maximum(alpha * layer_2, layer_2)
        
        # layer3 12 * 12 * 256
        layer_3 = tf.layers.conv2d(layer_2, 4 * size_mult, 5, strides=2, padding='same')
        layer_3 = tf.layers.batch_normalization(layer_3, training=True)
        layer_3 = tf.maximum(alpha * layer_3, layer_3)

        # layer4 6 * 6 * 512
        layer_4 = tf.layers.conv2d(layer_3, 8 * size_mult, 5, strides=2, padding='same')
        layer_4 = tf.layers.batch_normalization(layer_4, training=True)
        layer_4 = tf.maximum(alpha * layer_4, layer_4)

        flat = tf.reshape(layer_4, (-1, 6 * 6 * size_mult * 8))
        logits = tf.layers.dense(flat, 1)
        out = tf.sigmoid(logits)
        
    return out, logits


def generator(z, out_channel_dim, is_train=True, alpha=0.2, size_mult=64):
    with tf.variable_scope('generator', reuse=not is_train):
        # 6 * 6 * 1024
        z_o = tf.layers.dense(z, 6 * 6 * size_mult * 8)
        z_o = tf.reshape(z_o, (-1, 6, 6, size_mult * 8))
        z_o = tf.layers.batch_normalization(z_o, training=is_train)
        # z_o = tf.maximum(alpha * z_o, z_o)
        z_o = tf.nn.relu(z_o)
        
        # layer1 12 * 12 * 512
        layer_1 = tf.layers.conv2d_transpose(z_o, size_mult * 4, 5, strides=2, padding='same')
        layer_1 = tf.layers.batch_normalization(layer_1, training=is_train)
        # layer_1 = tf.maximum(alpha * layer_1, layer_1)
        layer_1 = tf.nn.relu(layer_1)
        
        # layer2 24 * 24 * 256
        layer_2 = tf.layers.conv2d_transpose(layer_1, size_mult * 2, 5, strides=2, padding='same')
        layer_2 = tf.layers.batch_normalization(layer_2, training=is_train)
        # layer_2 = tf.maximum(alpha * layer_2, layer_2)
        layer_2 = tf.nn.relu(layer_2)

        # layer3 48 * 48 * 128
        layer_3 = tf.layers.conv2d_transpose(layer_2, size_mult, 5, strides=2, padding='same')
        layer_3 = tf.layers.batch_normalization(layer_3, training=is_train)
        # layer_3 = tf.maximum(alpha * layer_3, layer_3)
        layer_3 = tf.nn.relu(layer_3)

        # 96 * 96 * 3
        logits = tf.layers.conv2d_transpose(layer_3, out_channel_dim, 5, strides=2, padding='same')
        out = tf.tanh(logits)
    return out


def model_loss(input_real, input_z, out_channel_dim, alpha=0.2, drop_rate=0.):
    g_out = generator(input_z, out_channel_dim, is_train=True, alpha=alpha)
    d_out_real, d_logits_real = discriminator(input_real, reuse=False, alpha=alpha, drop_rate=drop_rate)
    d_out_fake, d_logits_fake = discriminator(g_out, reuse=True, alpha=alpha, drop_rate=drop_rate)
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                labels=tf.ones_like(d_logits_real) * 0.9))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                               labels=tf.zeros_like(d_logits_fake)))
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                               labels=tf.ones_like(d_logits_fake)))
    d_loss = d_loss_real + d_loss_fake
    
    return d_loss, g_loss


def model_opt(d_loss, g_loss, learning_rate, beta1):
    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if var.name.startswith('generator')]
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
    return d_train_opt, g_train_opt


def show_generator_output(sess, n_images, input_z, out_channel_dim, image_mode):
    cmap = None if image_mode == 'RGB' else 'gray'
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples = sess.run(
        generator(input_z, out_channel_dim, False),
        feed_dict={input_z: example_z})

    images_grid = images_square_grid(samples, image_mode)
    plt.imshow(images_grid, cmap=cmap)
    plt.show()


def train(epochs, batch_size, z_dim, learning_rate, beta1, dataset, image_mode):
    if image_mode == 'L':
        channels = 1
    elif image_mode == 'RGB':
        channels = 3
    else:
        channels = dataset.shape[3]
    drop_rate = tf.placeholder_with_default(.5, (), "drop_rate")
    inputs_real, inputs_z, rate = model_inputs(dataset.shape[1], dataset.shape[2], channels, z_dim)
    d_loss, g_loss = model_loss(inputs_real, inputs_z, channels, alpha=0.2, drop_rate=drop_rate)
    d_opt, g_opt = model_opt(d_loss, g_loss, rate, beta1)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epochs):
            for batch_images in get_batches(dataset, batch_size):
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                sess.run([d_opt, g_opt], feed_dict={inputs_real: batch_images, inputs_z: batch_z, rate: learning_rate})
            train_loss_d = sess.run(d_loss, {inputs_z: batch_z, inputs_real: batch_images, drop_rate: 0.})
            train_loss_g = g_loss.eval({inputs_z: batch_z, drop_rate: 0.})
            print("Epoch {}/{}...".format(epoch_i+1, epochs),
              "Discriminator Loss: {:.4f}...".format(train_loss_d),
              "Generator Loss: {:.4f}".format(train_loss_g))
            if epoch_i % 1 == 0:
                show_generator_output(sess, 16, inputs_z, 3, image_mode)


if __name__ == '__main__':
    show_n_images = 25

    pic_path = 'data/faces/*.jpg'
    pic_files = glob(pic_path)

    show_images = read_pic(pic_files[:show_n_images])
    plt.imshow(images_square_grid(show_images, 'RGB'))

    batch_size = 64
    z_dim = 100
    learning_rate = 0.0002
    beta1 = 0.9

    epochs = 50

    pic_dataset = read_pic(pic_files)
    with tf.Graph().as_default():
        train(epochs, batch_size, z_dim, learning_rate, beta1, pic_dataset, 'RGB')

