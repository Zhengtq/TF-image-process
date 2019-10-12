from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from glob import glob
import random
import numpy as np
import math
import cv2

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops



def apply_with_random_selector(x, func, num_cases):

  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)

  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]




def tf_resizes_image(image, height, width):
    num_resize_cases = 4
#      image = apply_with_random_selector(
        #  image,
        #  lambda x, method: tf.image.resize_images(x, [height, width], method),
        #  num_cases=num_resize_cases)
    image = tf.image.resize_images(image, [height, width])
    return image


def tf_ran_avgfilter(image, ran_ratio = 0.02, class_num = 2):

    def averagefilter(image, k):
           k = k+2
           with tf.variable_scope('average_' + str(k), reuse=tf.AUTO_REUSE):
               w = tf.get_variable('average_w', [k, k, image.get_shape().as_list()[-1], 1],
                       initializer=tf.constant_initializer(1./(k*k)), trainable=False)
               image = tf.expand_dims(image, 0)
               image = tf.nn.depthwise_conv2d(image, w, [1,1,1,1], 'SAME')
               image = tf.squeeze(image)
           return image

    def sel_filter(image, class_num = 2):
        image = apply_with_random_selector(image, lambda x, ordering: averagefilter(x, ordering), num_cases=class_num)
        return image

    ran = tf.random_uniform([])
    image = tf.cond(ran<ran_ratio, lambda: sel_filter(image, class_num = class_num), lambda: image)
    return image

def gray_color_tur(image, ran_ratio = 0.02):

    def turn_gray(image):
        image = tf.image.rgb_to_grayscale(image)
        image = tf.stack([image, image, image], axis=3)
        image = tf.squeeze(image)
        return image

    ran = tf.random_uniform([])
    image = tf.cond(ran<ran_ratio, lambda: turn_gray(image), lambda: image)
    return image

def g_noise_tur(image, ran_ratio = 0.1, ratio = 1.0):

    def add_g_noise(input_layer, std):
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
        return input_layer + noise

    level=5*ratio
    ran = tf.random_uniform([])
    ran_std = tf.random_uniform([], maxval=level)
    image = tf.cond(ran<ran_ratio, lambda: add_g_noise(image, ran_std), lambda: image)
    return image

def tf_rotate_img(image, ran_ratio = 0.9):

    def rotate_img(image):
        rotate_angle = tf.random_uniform([], minval=-0.25, maxval=0.25)
        ran = tf.random_uniform([])
#          image = tf.cond(ran<0.5, lambda:tf.contrib.image.rotate(image, rotate_angle,  interpolation='NEAREST'),
                #  lambda:tf.contrib.image.rotate(image, rotate_angle,  interpolation='BILINEAR'))
        image = tf.contrib.image.rotate(image, rotate_angle,  interpolation='BILINEAR')
        return image

    ran = tf.random_uniform([])
    image = tf.cond(ran<ran_ratio, lambda: rotate_img(image), lambda: image)
    return image



def ran_face_size_tur_3(image, ran_ratio=0.03):
    def generate_face_mask(img_channel):
        img_height = tf.constant(320, dtype=tf.float32)/2
        center_ratio = tf.random_uniform([], minval=0.4, maxval=1)
        center_height_top = tf.cast(img_height*center_ratio, dtype=tf.int32)
        center_ratio = tf.random_uniform([], minval=0.4, maxval=1)
        center_height_bottom = tf.cast(img_height*center_ratio, dtype=tf.int32)
        center_ratio = tf.random_uniform([], minval=0.4, maxval=1)
        center_width_left = tf.cast(img_height*center_ratio, dtype=tf.int32)
        center_ratio = tf.random_uniform([], minval=0.4, maxval=1)
        center_width_right = tf.cast(img_height*center_ratio, dtype=tf.int32)
        mask = tf.ones((center_height_top + center_height_bottom, center_width_left + center_width_right, img_channel))
        pad_bottom = tf.cast((160 - center_height_bottom), tf.int32)
        pad_top = tf.cast(320 - pad_bottom - center_height_top - center_height_bottom, tf.int32)
        pad_left = tf.cast((160 - center_width_left), tf.int32)
        pad_right = tf.cast(320 - pad_left - center_width_right - center_width_left, tf.int32)
        mask = tf.pad(mask, [[pad_bottom, pad_top], [pad_left, pad_right], [0,0]])
        mask = tf.reshape(mask, [320, 320, img_channel])
        return mask

    ran = tf.random_uniform([])
    img_channel = image.get_shape().as_list()[-1]
    image = tf.cond(ran<ran_ratio, lambda: image*generate_face_mask(img_channel), lambda: image)
    return image



def all_yuv_tur(image_in, ran_ratio=0.03, ratio = 0.8):

    def yuv_tur(image , case):
        image = image / 255
        image = tf.image.rgb_to_yuv(image)
        yuv_channels = tf.split(image, 3, 2)
        y = yuv_channels[0]
        uv = tf.concat(yuv_channels[1:],2)
        shape1 = uv.get_shape().as_list()
        level=int(8*ratio)
        d1  = tf.random_uniform([], maxval=level, dtype=tf.int32)
        d2  = tf.random_uniform([], maxval=level, dtype=tf.int32)
        if case == 0:
            uv = tf.slice(uv, [0,0,0], [shape1[0]-d1, shape1[1]-d2, shape1[2]])
            uv = tf.pad(uv, [[d1,0], [d2, 0], [0,0]])
        elif case == 1:
            uv = tf.slice(uv, [d1,d2,0], [shape1[0]-d1, shape1[1]-d2, shape1[2]])
            uv = tf.pad(uv, [[0, d1], [0, d2], [0,0]])
        elif case == 2:
            uv = tf.slice(uv, [d1,0,0], [shape1[0]- d1, shape1[1]-d2, shape1[2]])
            uv = tf.pad(uv, [[0, d1], [d2, 0], [0,0]])
        elif case == 3:
            uv = tf.slice(uv, [0,d2,0], [shape1[0]-d1, shape1[1]-d2, shape1[2]])
            uv = tf.pad(uv, [[d1, 0], [0, d2], [0,0]])
        yuv_image = tf.concat([y, uv], 2)
        image = tf.image.yuv_to_rgb(yuv_image)
        image = image * 255
        return image

    def yuv_tur_sel(image):
        image = apply_with_random_selector(image,
            lambda x, ordering: yuv_tur(x, ordering), num_cases=4)
        return image

    ran = tf.random_uniform([])
    image = tf.cond(ran<ran_ratio, lambda: yuv_tur_sel(image_in), lambda: image_in)
    return image

def transform_perspective(image, ran_ratio=0.02):
    def x_y_1():
        x = tf.random_uniform([], minval=-0.25, maxval=-0.05)
        y = tf.random_uniform([], minval=-0.25, maxval=-0.05)
        return x, y

    def x_y_2():
        x = tf.random_uniform([], minval=0.05, maxval=0.25)
        y = tf.random_uniform([], minval=0.05, maxval=0.25)
        return x, y

    def trans(image):
        x = tf.random_uniform([], minval=-0.3, maxval=0.3)
        x_com = tf.random_uniform([], minval=1-x-0.1, maxval=1-x+0.1)
        y = tf.random_uniform([], minval=-0.3, maxval=0.3)
        y_com = tf.random_uniform([], minval=1-y-0.1, maxval=1-y+0.1)
        transforms =  [x_com, x,0,y,y_com,0,0,0]
        ran = tf.random_uniform([])
#          image = tf.cond(ran<0.5, lambda:tf.contrib.image.transform(image,transforms,interpolation='NEAREST', name=None),
                #  lambda:tf.contrib.image.transform(image,transforms,interpolation='BILINEAR', name=None))
        image = tf.contrib.image.transform(image,transforms,interpolation='BILINEAR', name=None)
        return image

    ran = tf.random_uniform([])
    image = tf.cond(ran<ran_ratio, lambda: trans(image), lambda:image)
    return image

def ran_img_motion_blur(image, ran_ratio, minval, maxval):

    def motion_blur(image, minval, maxval):
        degree = np.random.randint(minval, maxval)
        angle = np.random.uniform(-180, 180)
        image = np.array(image)
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)
        return blurred

    ran = tf.random_uniform([])
    image = tf.cond(ran<ran_ratio, lambda:tf.cast(tf.py_func(motion_blur, [image, minval, maxval], tf.uint8), tf.float32),
            lambda:image)
    return image

def tf_ran_resize_tur123(image, ran_ratio = 0.5, resize_ratio = 0.5):

    def inner_fun(image, resize_ratio = 0.5):

        origin_height, origin_width,_ = image.get_shape().as_list()
        resize_dest = tf.random_uniform([], minval=int(origin_height * 0.6), maxval=int(origin_height * 0.9))
        image = tf_resizes_image(image, resize_dest, resize_dest)
        image = tf_resizes_image(image, origin_height, origin_width)
        return image
    ran = tf.random_uniform([])
    image = tf.cond(ran<ran_ratio, lambda:inner_fun(image, resize_ratio = resize_ratio), lambda:image)

    return image


def tf_ran_jpeg_compress(image, ran_ratio = 0.05, the_range = [1, 100]):


    print(the_range)
    def inner_fun_jc(image, the_range):
        image = tf.cast(image, tf.uint8, name=None)
        image = tf.image.random_jpeg_quality(image, min_jpeg_quality = the_range[0], max_jpeg_quality = the_range[1])
        image = tf.reshape(image, (280, 280, 3))
        image = tf.cast(image, tf.float32, name=None)
        return image

    ran = tf.random_uniform([])
    image = tf.cond(ran<ran_ratio, lambda:inner_fun_jc(image, the_range), lambda:image)
    return image


def tf_random_crop(image, minval,maxval,dest_size, channel = 3):

    def produce_ran_img(image, minval, maxval, dest_size, channel):
        ran_range = tf.random_uniform([], minval=minval, maxval=maxval, dtype=tf.int32)
        image = tf.random_crop(image, size=[ran_range, ran_range, channel])
        image = tf_resizes_image(image, dest_size, dest_size)
        return image

    ran = tf.random_uniform([])
    image = tf.cond(ran<0.5, lambda: produce_ran_img(image, minval, maxval, dest_size, channel),
            lambda:tf.random_crop(image, size=[dest_size, dest_size, channel]))
    return image


def tf_ran_rot90(image, ran_ratio=0.03):

   def inner_fun(image):
       ran_k = tf.random_uniform([], minval = 1, maxval = 4, dtype = tf.int32)
       image = tf.image.rot90(image, ran_k)
       return image

   ran = tf.random_uniform([])
   image = tf.cond(ran<ran_ratio, lambda:inner_fun(image), lambda:image)
   return image

def ran_flip_up_down(image, ran_ratio=0.02):

    ran = tf.random_uniform([])
    image = tf.cond(ran<ran_ratio, lambda: tf.image.flip_up_down(image), lambda:image)
    return image

def random_color_hue(image,ran_ratio = 0.05,ratio=0.4):

    color_range = 30*ratio
    hue_level = 0.1*ratio
    def all_color_channel_tur(image):
        def add_color_mask_test(image):
            img_shape = image.get_shape().as_list()
            img_shape = [img_shape[0], img_shape[1], 1]
            ran_color_range_0 = tf.random_uniform([], minval=-color_range, maxval=color_range,dtype=tf.float32)
            color_mask_0 = tf.ones(img_shape, dtype = tf.float32) * ran_color_range_0
            ran_color_range_1 = tf.random_uniform([], minval=-color_range, maxval=color_range,dtype=tf.float32)
            color_mask_1 = tf.ones(img_shape, dtype = tf.float32) * ran_color_range_1
            ran_color_range_2 = tf.random_uniform([], minval=-color_range, maxval=color_range,dtype=tf.float32)
            color_mask_2 = tf.ones(img_shape, dtype = tf.float32) * ran_color_range_2
            sep_channels = tf.split(image, 3, 2)
            sep_channels[0] = tf.add(sep_channels[0], color_mask_0)
            sep_channels[1] = tf.add(sep_channels[1], color_mask_1)
            sep_channels[2] = tf.add(sep_channels[2], color_mask_2)
            image = tf.concat(sep_channels,2)
            return image

#          ran = tf.random_uniform([])
        #  image = tf.cond(ran<0.5, lambda:add_color_mask_test(image), lambda: image)

        image = add_color_mask_test(image)
        return image


    def ran_adjust_hue(image):
      image = tf.image.random_hue(image, max_delta=hue_level)
      return image

    def produce_img_1(image):
        ran = tf.random_uniform([])
        image = tf.cond(ran<0.5, lambda: ran_adjust_hue(image), lambda: all_color_channel_tur(image))
        return image

    ran_1 = tf.random_uniform([])
    image = tf.cond(ran_1<ran_ratio, lambda: produce_img_1(image), lambda: image)
    return image

def distort_color(image, color_ordering=0, fast_mode=False, ratio=1.0, scope=None):

  lower_1 = 1 - 0.6*ratio
  higher_1 = 1+0.6*ratio
  lower_2 = 1 - 0.4 * ratio
  higher_2 = 1+0.4*ratio
  bright_level = 35



  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=bright_level)
        image = tf.image.random_saturation(image, lower=lower_1, upper=higher_1)
      else:
        image = tf.image.random_saturation(image, lower=lower_1, upper=higher_1)
        image = tf.image.random_brightness(image, max_delta=bright_level)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=bright_level)
        image = tf.image.random_saturation(image, lower=lower_1, upper=higher_1)
    #    image = random_color_hue(image)
        image = tf.image.random_contrast(image, lower=lower_2, upper=higher_2)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=lower_1, upper=higher_1)
        image = tf.image.random_brightness(image, max_delta=bright_level)
        image = tf.image.random_contrast(image, lower=lower_2, upper=higher_2)
   #     image = random_color_hue(image)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=lower_2, upper=higher_2)
    #    image = random_color_hue(image)
        image = tf.image.random_brightness(image, max_delta=bright_level)
        image = tf.image.random_saturation(image, lower=lower_1, upper=higher_1)
      elif color_ordering == 3:
    #    image = random_color_hue(image)
        image = tf.image.random_saturation(image, lower=lower_1, upper=higher_1)
        image = tf.image.random_contrast(image, lower=lower_2, upper=higher_2)
        image = tf.image.random_brightness(image, max_delta=bright_level)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    return tf.clip_by_value(image, 0.0, 255.0)

def img_color_tur(image, ratio=1.0, fast_mode=False):
    num_distort_cases = 1 if fast_mode else 4
    image = apply_with_random_selector(image,
        lambda x, ordering: distort_color(x, ordering, fast_mode=fast_mode, ratio=ratio), num_cases=num_distort_cases)
    return image

def ran_img_color_tur(image, ran_ratio=0.1, ratio = 0.8):
    ran = tf.random_uniform([])
    image = tf.cond(ran<ran_ratio, lambda: img_color_tur(image, ratio=ratio, fast_mode=False), lambda: image)
    return image


def random_erase_np_v2(image, ran_ratio = 0.1):

    def inner_fun(img, sl = 0.02, sh = 0.02, r1 = 0.5):
	height = img.shape[0]
	width = img.shape[1]
	channel = img.shape[2]
	area = width * height

	erase_area_low_bound = np.round( np.sqrt(sl * area * r1) ).astype(np.int)
	erase_area_up_bound = np.round( np.sqrt((sh * area) / r1) ).astype(np.int)
	if erase_area_up_bound < height:
	    h_upper_bound = erase_area_up_bound
	else:
	    h_upper_bound = height
	if erase_area_up_bound < width:
	    w_upper_bound = erase_area_up_bound
	else:
	    w_upper_bound = width

	h = np.random.randint(erase_area_low_bound, h_upper_bound)
	w = np.random.randint(erase_area_low_bound, w_upper_bound)

	x1 = np.random.randint(0, height - h)
	y1 = np.random.randint(0, width - w)
	img[x1:x1+h, y1:y1+w, :] = np.random.randint(0, 255, size=(h, w, channel)).astype(np.float32)
	return img

    ran = tf.random_uniform([])
    image = tf.cond(ran<ran_ratio, lambda:tf.py_func(inner_fun, [image], tf.float32), lambda:image)

    return image

def tf_lower_img_bright(image, ran_ratio = 0.08, bound = [-70, -20]):
	ran = tf.random_uniform([])
	bright_range = tf.random_uniform([], minval=bound[0], maxval=bound[1],dtype=tf.float32)
	image = tf.cond(ran<ran_ratio, lambda:tf.image.adjust_brightness(image, bright_range), lambda:image)
	return image












def distort_color_neg(image, color_ordering=0, fast_mode=False, ratio=1.0, scope=None):

  lower_1 = 1-0.6*ratio
  higher_1 = 1
  lower_2 = 1-0.4*ratio
  higher_2 = 1
  bright_level = 35



  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=bright_level)
        image = tf.image.random_saturation(image, lower=lower_1, upper=higher_1)
      else:
        image = tf.image.random_saturation(image, lower=lower_1, upper=higher_1)
        image = tf.image.random_brightness(image, max_delta=bright_level)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=bright_level)
        image = tf.image.random_saturation(image, lower=lower_1, upper=higher_1)
    #    image = random_color_hue(image)
        image = tf.image.random_contrast(image, lower=lower_2, upper=higher_2)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=lower_1, upper=higher_1)
        image = tf.image.random_brightness(image, max_delta=bright_level)
        image = tf.image.random_contrast(image, lower=lower_2, upper=higher_2)
   #     image = random_color_hue(image)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=lower_2, upper=higher_2)
    #    image = random_color_hue(image)
        image = tf.image.random_brightness(image, max_delta=bright_level)
        image = tf.image.random_saturation(image, lower=lower_1, upper=higher_1)
      elif color_ordering == 3:
    #    image = random_color_hue(image)
        image = tf.image.random_saturation(image, lower=lower_1, upper=higher_1)
        image = tf.image.random_contrast(image, lower=lower_2, upper=higher_2)
        image = tf.image.random_brightness(image, max_delta=bright_level)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    return tf.clip_by_value(image, 0.0, 255.0)

def img_color_tur_neg(image, ratio=1.0, fast_mode=False):
    num_distort_cases = 1 if fast_mode else 4
    image = apply_with_random_selector(image,
        lambda x, ordering: distort_color(x, ordering, fast_mode=fast_mode, ratio=ratio), num_cases=num_distort_cases)
    return image

def ran_img_color_tur_neg(image, ran_ratio=0.1, ratio = 0.8):
    ran = tf.random_uniform([])
    image = tf.cond(ran<ran_ratio, lambda: img_color_tur(image, ratio=ratio, fast_mode=False), lambda: image)
    return image




