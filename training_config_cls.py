from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from glob import glob
import random
import numpy as np
from tensorflow.python.ops import array_ops
slim = tf.contrib.slim


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            if g is None:
                continue
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        if len(grads) == 0:
            continue

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def save_model(saver, sess, model_dir, step, flag = ''):

        if not os.path.exists(model_dir):
                os.makedirs(model_dir)

        checkpoint_path = os.path.join(model_dir,  flag +'_model.kept')
        if step<100:
            checkpoint_path = os.path.join(model_dir,  'meta_model.kept')
            w_graph = True
        else:
            w_graph = False

        saver.save(sess, checkpoint_path,  global_step = step, write_meta_graph=w_graph)



class COM_LOSS:

    def __init__(self):
        self.a = 1

    def _get_regulation_loss(self):
       regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) 
       return regularization_loss


    def _get_smooth_cross_entropy_loss(self, logits, one_hot_labels, label_smoothing=0,
                           weight=1.0, scope=None):

          logits.get_shape().assert_is_compatible_with(one_hot_labels.get_shape())
          with tf.name_scope(scope, 'CrossEntropyLoss', [logits, one_hot_labels]):
            num_classes = one_hot_labels.get_shape()[-1].value
            one_hot_labels = tf.cast(one_hot_labels, logits.dtype)
            if label_smoothing > 0:
              smooth_positives = 1.0 - label_smoothing
              smooth_negatives = label_smoothing / num_classes
              one_hot_labels = one_hot_labels * smooth_positives + smooth_negatives
            cross_entropy = tf.contrib.nn.deprecated_flipped_softmax_cross_entropy_with_logits(
                logits, one_hot_labels, name='xentropy')

        #    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = one_hot_labels, logits = logits)
            weight = tf.convert_to_tensor(weight,
                                          dtype=logits.dtype.base_dtype,
                                          name='loss_weight')
            loss = tf.multiply(weight, tf.reduce_mean(cross_entropy), name='value')
            return loss



    #class_value alpha = 0.35, gamma = 2.0
    #class_value alpha = 0.6, gamma = 1.5
    #class_value alpha = 0.6, gamma = 2.0
    def _focal_loss(self, onehot_labels, logits, alpha=0.5,  gamma=1.0, name=None, scope=None):



        with tf.name_scope(scope, 'focal_loss', [logits, onehot_labels]) as sc:

            precise_logits = tf.cast(logits, tf.float32) if (
                    logits.dtype == tf.float16) else logits
            onehot_labels = tf.cast(onehot_labels, precise_logits.dtype)
            predictions = tf.nn.sigmoid(logits)
            predictions_pt = tf.where(tf.equal(onehot_labels, 1), predictions, 1.-predictions)

            epsilon = 1e-8
            alpha_t = tf.scalar_mul(alpha, tf.ones_like(onehot_labels, dtype=tf.float32))
            alpha_t = tf.where(tf.equal(onehot_labels, 1.0), alpha_t, 1-alpha_t)
            losses = tf.reduce_mean(tf.reduce_sum(-alpha_t *
                tf.pow(1. - predictions_pt, gamma) * tf.log(predictions_pt+epsilon),
                    name=name, axis=1))


            return losses



    def _focal_loss_debug(self, labels, logits, alpha=0.5,  gamma=2.0, name=None, scope=None):


        with tf.name_scope(scope, 'focal_loss', [logits, onehot_labels]) as sc:

            one_hot_labels = slim.one_hot_encoding(labels, 2)
            precise_logits = tf.cast(logits, tf.float32) if (
                    logits.dtype == tf.float16) else logits
            onehot_labels = tf.cast(onehot_labels, precise_logits.dtype)
            predictions = tf.nn.sigmoid(logits)
            predictions_pt = tf.where(tf.equal(onehot_labels, 1), predictions, 1.-predictions)


            B_ = tf.expand_dims(labels, 1)
            B_oh_1 = tf.concat([B_, B_], axis=1)
 	    B_oh_1 = tf.cast(B_oh_1, tf.float32)

            epsilon = 1e-8
            alpha_t = tf.scalar_mul(alpha, tf.ones_like(B_oh_1, dtype=tf.float32))
            alpha_t = tf.where(tf.equal(B_oh_1, 1.0), alpha_t, 1-alpha_t)
            losses = tf.reduce_mean(tf.reduce_sum(-alpha_t *
                tf.pow(1. - predictions_pt, gamma) * tf.log(predictions_pt+epsilon),
                    name=name, axis=1))


            return losses



    #alpha = 0.5, gamma=1.0
    def _focal_loss_2(self, labels, logits, alpha=0.5, gamma=1.0):

        logits = tf.squeeze(logits)
	y_pred = tf.nn.sigmoid(logits)
	labels = tf.to_float(labels)

#          loss = -1.2 *labels * alpha * (tf.maximum((1 - y_pred) ** gamma, 0.5)) * tf.log(tf.clip_by_value(y_pred, 1e-8, 1.0)) \
                #  -1.2 *(1 - labels) * (1 - alpha) * (tf.maximum(y_pred ** gamma, 0.5)) * tf.log(tf.clip_by_value(1 - y_pred, 1e-8, 1.0))


        loss = -labels * alpha * ((1 - y_pred) ** gamma) * tf.log(tf.clip_by_value(y_pred, 1e-8, 1.0)) \
                -(1 - labels) * (1 - alpha) * (y_pred ** gamma) * tf.log(tf.clip_by_value(1 - y_pred, 1e-8, 1.0))

        loss = tf.reduce_mean(loss)
	return loss

    def _focal_loss_3(self, labels, logits, alpha=0.5, gamma=1.0):

        logits = tf.squeeze(logits)
	y_pred = tf.nn.sigmoid(logits)
	labels = tf.to_float(labels)

        loss = -labels * ((1 - y_pred) ** gamma) * tf.log(tf.clip_by_value(y_pred, 1e-8, 1.0)) \
                -(1 - labels) * (y_pred ** gamma) * tf.log(tf.clip_by_value(1 - y_pred, 1e-8, 1.0))

        loss = tf.reduce_mean(loss)
	return loss

    def _get_center_loss(self, features, labels, num_classes, alpha = 0.5):



            len_features = features.get_shape()[1]


            centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                    initializer=tf.constant_initializer(0), trainable=False)

            labels = tf.reshape(labels, [-1])
            centers_batch = tf.gather(centers, labels)
            loss = tf.nn.l2_loss(features - centers_batch)
            diff = centers_batch - features

            unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
            appear_times = tf.gather(unique_count, unique_idx)
            appear_times = tf.reshape(appear_times, [-1, 1])
            diff = diff / tf.cast((1 + appear_times), tf.float32)
            diff = alpha * diff
            centers_update_op = tf.scatter_sub(centers, labels, diff)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, centers_update_op)
            return loss, centers, centers_update_op


    def _get_depth_loss(self, true_depth, generate_depth):


         #  generate_depth = tf.nn.sigmoid(generate_depth)

           true_depth = true_depth / 255.0


	   k0 = tf.reshape(tf.constant([[1,0,0], [0,-1,0],[0,0,0]], dtype=tf.float32), [3,3,1,1])
           k1 = tf.reshape(tf.constant([[0,1,0], [0,-1,0],[0,0,0]], dtype=tf.float32), [3,3,1,1])
           k2 = tf.reshape(tf.constant([[0,0,1], [0,-1,0],[0,0,0]], dtype=tf.float32), [3,3,1,1])
           k3 = tf.reshape(tf.constant([[0,0,0], [1,-1,0],[0,0,0]], dtype=tf.float32), [3,3,1,1])
           k4 = tf.reshape(tf.constant([[0,0,0], [0,-1,1],[0,0,0]], dtype=tf.float32), [3,3,1,1])
           k5 = tf.reshape(tf.constant([[0,0,0], [0,-1,0],[1,0,0]], dtype=tf.float32), [3,3,1,1])
           k6 = tf.reshape(tf.constant([[0,0,0], [0,-1,0],[0,1,0]], dtype=tf.float32), [3,3,1,1])
           k7 = tf.reshape(tf.constant([[0,0,0], [0,-1,0],[0,0,1]], dtype=tf.float32), [3,3,1,1])

           all_kernel = [k0, k1, k2, k3, k4, k5, k6, k7]

           loss_d =  tf.sqrt(tf.nn.l2_loss([true_depth-generate_depth]))
        #   loss_d = tf.reduce_mean((true_depth-generate_depth)**2) 

           for k in all_kernel:
               true_depth_k = tf.nn.depthwise_conv2d(true_depth, k, [1,1,1,1], 'SAME')
               generate_depth_k = tf.nn.depthwise_conv2d(generate_depth, k, [1,1,1,1], 'SAME')
               loss_d += tf.sqrt(tf.nn.l2_loss([true_depth_k-generate_depth_k]))
          #     loss_d += tf.reduce_mean((true_depth_k-generate_depth_k)**2)

        #   img_shape = generate_depth.get_shape().as_list()
        #   loss_d = loss_d /(img_shape[0] * img_shape[1] * img_shape[2] * img_shape[3])
           loss_d = loss_d * 0.001
        #   loss_d = loss_d
           tf.add_to_collection('train_depth_loss_pos',1.0)
           tf.add_to_collection('train_depth_loss_neg',1.0)

	   return loss_d



class COM_ACC:

    def __init__(self):
        self.a = 1


    def _get_train_acc(self, logits, labels, label_num=2, name = "metrics_train"):

            if label_num == 1:
                logits = tf.squeeze(logits)
                logits_sig = tf.nn.sigmoid(logits, name = 'one_logits')
                predicted_labels_r = tf.round(logits_sig)
                predicted_labels = tf.cast(predicted_labels_r, tf.int64)
            else:
                predicted_labels = tf.argmax(logits, 1)

       #     accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, labels), tf.float32))


            tf_accuracy, tf_accuracy_op = tf.metrics.accuracy(labels, predicted_labels, name=name)
            tf_precision, tf_precision_op = tf.metrics.precision(labels, predicted_labels, name=name)
            tf_recall, tf_recall_op = tf.metrics.recall(labels, predicted_labels, name=name)
            
            auc_pred = tf.nn.sigmoid(tf.squeeze(logits))
            tf_auc, tf_auc_op = tf.metrics.auc(labels, auc_pred, name=name)
            tf_f1_score = 2 * tf_precision_op * tf_recall_op / (tf_precision_op + tf_recall_op)



            return [tf_accuracy_op, tf_precision_op, tf_recall_op, tf_auc_op, tf_f1_score], predicted_labels

























