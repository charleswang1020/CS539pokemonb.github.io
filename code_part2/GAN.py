import os
import tensorflow as tf
import numpy as np
import cv2
import random
import scipy.misc
import tensorflow.contrib.slim as slim

def leak_relu(x, n, leak=0.3): 
    return tf.maximum(x, leak * x, name=n)


def process_data():
    pokemon_dir = '/floyd/input/pikachu'
    images = []
    for each in os.listdir(pokemon_dir):
        images.append(os.path.join(pokemon_dir,each))
  
    all_images = tf.convert_to_tensor(images, dtype = tf.string)
    images_queue = tf.train.slice_input_producer([all_images])
    content = tf.read_file(images_queue[0])
    image = tf.image.decode_jpeg(content, channels = 3)

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta = 0.1)
    image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)
    
    image = tf.image.resize_images(image, [128, 128])
    image.set_shape([128,128,3])
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    
    images_batch = tf.train.shuffle_batch([image], batch_size = 64, num_threads = 4,
                                          capacity = 200 + 3* 64, min_after_dequeue = 200)
    return images_batch, len(images)


def generator(input, random_dim, is_train, reuse=False):
    c4, c8, c16, c32, c64 = 512, 256, 128, 64, 32
    
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()

        w1 = tf.get_variable('w1', shape=[random_dim, 4 * 4 * c4], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.03))
        b1 = tf.get_variable('b1', shape=[c4 * 4 * 4], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        z_conv1 = tf.add(tf.matmul(input, w1), b1, name='z_conv1')
        
        # layer1
        conv1 = tf.reshape(z_conv1, shape=[-1, 4, 4, c4], name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay = 0.9,
                                           updates_collections=None, scope='bn1')
        a_conv1 = tf.nn.relu(bn1, name='a_conv1')
        
        # layer2
        conv2 = tf.layers.conv2d_transpose(a_conv1, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,
                                           updates_collections=None, scope='bn2')
        a_conv2 = tf.nn.relu(bn2, name='a_conv2')
        
        # layer3
        conv3 = tf.layers.conv2d_transpose(a_conv2, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,
                                           updates_collections=None, scope='bn3')
        a_conv3 = tf.nn.relu(bn3, name='a_conv3')
        
        # layer4
        conv4 = tf.layers.conv2d_transpose(a_conv3, c32, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,
                                           updates_collections=None, scope='bn4')
        a_conv4 = tf.nn.relu(bn4, name='a_conv4')
        
        # layer5
        conv5 = tf.layers.conv2d_transpose(a_conv4, c64, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv5')
        bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn5')
        a_conv5 = tf.nn.relu(bn5, name='a_conv5')
        
        # layer6
        conv6 = tf.layers.conv2d_transpose(a_conv5, 3, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv6')
        a_conv6 = tf.nn.tanh(conv6, name='a_conv6')
        return a_conv6


def discriminator(input, is_train, reuse=False):
    c2, c4, c8, c16 = 64, 128, 256, 512
    
    with tf.variable_scope('dis') as scope:
        if reuse:
            scope.reuse_variables()

        #layer1
        conv1 = tf.layers.conv2d(input, c2, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon=1e-5, decay = 0.9,
                                           updates_collections=None, scope = 'bn1')
        a_conv1 = leak_relu(conv1, n='a_conv1')
        
        #layer2
        conv2 = tf.layers.conv2d(a_conv1, c4, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,
                                           updates_collections=None, scope='bn2')
        a_conv2 = leak_relu(bn2, n='a_conv2')
        
        #layer3
        conv3 = tf.layers.conv2d(a_conv2, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,
                                           updates_collections=None, scope='bn3')
        a_conv3 = leak_relu(bn3, n='a_conv3')
        
        #layer4
        conv4 = tf.layers.conv2d(a_conv3, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,
                                           updates_collections=None, scope='bn4')
        a_conv4 = leak_relu(bn4, n='a_conv4')
                
        dim = int(np.prod(a_conv4.get_shape()[1:]))
        fc1 = tf.reshape(a_conv4, shape=[-1, dim], name='fc1')
        
        w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')
        return logits


def save_images(images, size, image_path):
    return scipy.misc.imsave((images+1.)/2., size, image_path)


def train():    
    with tf.variable_scope('input'):
        real_image = tf.placeholder(tf.float32, shape = [None, 128, 128, 3], name='real_image')
        random_input = tf.placeholder(tf.float32, shape=[None, 100], name='rand_input')
        is_train = tf.placeholder(tf.bool, name='is_train')
    
    fake_image = generator(random_input, 100, is_train)
    real_result = discriminator(real_image, is_train)
    fake_result = discriminator(fake_image, is_train, reuse=True)
    
    d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)
    g_loss = -tf.reduce_mean(fake_result)
            
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
    trainer_d = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(g_loss, var_list=g_vars)

    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

    image_batch, samples_num = process_data()
    
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    save_path = saver.save(sess, "/tmp/model.ckpt")
    ckpt = tf.train.latest_checkpoint('./model/pikachu')
    saver.restore(sess, save_path)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(5000):
        print("Running epoch {}/{}...".format(i, 5000))
        for j in range(int(samples_num / 64)):
            d_iters = 5
            g_iters = 1

            train_noise = np.random.uniform(-1.0, 1.0, size=[64, 100]).astype(np.float32)
            for k in range(d_iters):
                train_image = sess.run(image_batch)
                sess.run(d_clip)
                _, dLoss = sess.run([trainer_d, d_loss],
                                    feed_dict={random_input: train_noise, real_image: train_image, is_train: True})
            
            for k in range(g_iters):
                _, gLoss = sess.run([trainer_g, g_loss],
                                    feed_dict={random_input: train_noise, is_train: True})
            
        if i%500 == 0:
            if not os.path.exists('./model/pikachu'):
                os.makedirs('./model/pikachu')
            saver.save(sess, './model/pikachu/' + str(i))  
        
        if i%50 == 0:
            if not os.path.exists('./newPokemon'):
                os.makedirs('./newPokemon')
            sample_noise = np.random.uniform(-1.0, 1.0, size=[64, 100]).astype(np.float32)
            imgtest = sess.run(fake_image, feed_dict={random_input: sample_noise, is_train: False})
            save_images(imgtest, [8,8] ,'./newPokemon' + '/epoch' + str(i) + '.jpg')
            print('train:[%d],d_loss:%f,g_loss:%f' % (i, dLoss, gLoss))
    
    coord.request_stop()
    coord.join(threads)

train()
