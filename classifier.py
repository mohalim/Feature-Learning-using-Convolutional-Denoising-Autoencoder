# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 19:37:25 2019

@author: halimnoor
"""

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
import numpy as np

ae_model = dir_model
ae_fc_model = 'Models/cdae_fc_elu_filters_' + str(latent_filter) + '_fold_' + str(fold) + '/'

is_train = 1
is_test = 1
is_load = 0

y_test = np.copy(test_label)


learning_rate = 0.5 # fold1: 0.7, fold2: 0.5, fold3: 0.7
display_step = 20
min_delta = 0.00001
patience = 100
epoch = 1000    # fold1: 3500, fold2: 3500, fold3: 3500

h1 = 40
h2 = 20

tf.reset_default_graph()

saver = tf.train.import_meta_graph(ae_model+'model.ckpt.meta')
sess = tf.Session()
saver.restore(sess, ae_model+'model.ckpt')
ae_graph = tf.get_default_graph()

act_fn = tf.nn.elu

X_net = ae_graph.get_tensor_by_name('X_noise:0')
y = tf.placeholder(tf.float32, shape=(None, batch_y.shape[1]), name='y')

latent_conv1d = ae_graph.get_tensor_by_name('network/encoder/latent/conv1d/Conv2D:0')
with tf.variable_scope('fully_connected', reuse=None):
    flat = tf.layers.flatten(latent_conv1d, name='latent_flat') # num of neurons 450
    fc1 = tf.layers.dense(flat, h1, activation=act_fn,
                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                          use_bias=True, bias_initializer=tf.zeros_initializer())
    fc1 = tf.layers.dropout(fc1)
    fc2 = tf.layers.dense(fc1, h2, activation=act_fn,
                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                          use_bias=True, bias_initializer=tf.zeros_initializer())
    fc2 = tf.layers.dropout(fc2)
    logits = tf.layers.dense(fc2, batch_y.shape[1], activation=None,
                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                          use_bias=True, bias_initializer=tf.zeros_initializer())
    softmax = tf.nn.softmax(logits)
    

with tf.name_scope('loss'):
    cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=softmax, name='xentropy')
    loss = tf.reduce_mean(cost, name='loss')
    print(cost)
    print(loss)
    
with tf.name_scope('train'):
    train_weights = tf.trainable_variables()
    fc_weights = [w for w in train_weights if 'fully_connected' in w.name]
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    solver = optimizer.minimize(loss, var_list=fc_weights)

pred = tf.argmax(softmax, 1)
correct_pred = tf.equal(pred, tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

if is_train:
    saver = tf.train.Saver()
    
    min_loss = 0
    with tf.name_scope('eval'):
        patience_cnt = 0
        with tf.Session() as sess:
            if is_load:
                saver.restore(sess, ae_fc_model+'model.ckpt')
            else:
                sess.run(tf.global_variables_initializer())
                hist_loss = []
                hist_acc = []
            for e in range(0,epoch):
                avg_loss = 0
                avg_acc = 0
                n_batches = len(x_batches)
                for b in range(0,n_batches):
                    batch_x = x_batches[b]
                    _, loss_val, acc_val = sess.run([solver, loss, accuracy], 
                                           feed_dict={X_net: batch_x, 
                                           y:batch_y})
                    avg_loss = avg_loss + loss_val
                    avg_acc = avg_acc + acc_val
                    
                avg_loss = avg_loss / n_batches
                avg_acc = avg_acc / n_batches
                hist_loss.append(avg_loss)
                hist_acc.append(avg_acc)
                if e % display_step == 0:
                    print('epoch:', '%04d'%(e),
                          'cost:', '{:.6f}'.format(avg_loss),
                          'acc:', '{:.6f}'.format(avg_acc))
                    if min_loss == 0:
                        min_loss = avg_loss
                    else:
                        if avg_loss < min_loss:
                            min_loss = avg_loss
                            saver.save(sess, ae_fc_model+'model.ckpt')

                if epoch > 0 and hist_loss[e-1] - hist_loss[e] > min_delta:
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                if patience_cnt > patience:
                    print('epoch:', '%04d'%(e),
                          'cost:', '{:.6f}'.format(avg_loss),
                          'acc:', '{:.6f}'.format(avg_acc))
                    break
    
if is_test:
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, ae_fc_model+'model.ckpt')
        y_pred = sess.run([softmax], feed_dict={X_net: x_test})[0]
        y_pred = np.argmax(y_pred, axis=1)
    
    acc_score = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred)
    print(acc_score)
    print(conf_mat)
    print(cls_report)
