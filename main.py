#coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
def distort_color(image,color_ordering = 0):
    if color_ordering ==0:
        image = tf.image.random_brightness(image,max_delta=32./255.)            #调整图像亮度、
        image = tf.image.random_saturation(image,lower = 0.5,upper=1.5)         #调整图像的饱和度
        image = tf.image.random_hue(image,max_delta= 0.2)                       #调整图像的色相
        image = tf.image.random_contrast(image,lower = 0.5,upper = 1.5)         #调整图像对比度

    elif color_ordering ==1:
        image = tf.image.random_contrast(image,lower=0.5,upper = 1.5)
        image = tf.image.random_saturation(image,lower = 0.5,upper = 1.5)
        image = tf.image.random_brightness(image,max_delta= 32./255.)
        image = tf.image.random_hue(image,max_delta=0.2)
    return tf.clip_by_value(image,0.0,1.0)
def preprocess_for_train(image,height,width,bbox):
    #如果没有没出标注框，那么整幅图像就是要注意的部分
    if bbox is None:
        bbox = tf.constant([0.0,0.0,1.0,1.0],dtype=tf.float32,shape=[1,1,4])
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image,dtype=tf.float32)
        #随机截取图像
    bbox_begin,bbox_size,_ = tf.image.sample_distorted_bounding_box(tf.shape(image),bounding_boxes=bbox)
    print(bbox_begin,bbox_size)
    distort_image = tf.slice(image,bbox_begin,bbox_size)
    #裁剪图片
    # distort_image = tf.image.resize_images(distort_image,height,width,method= np.random.randint(4))
    #左右翻转图片
    distort_image = tf.image.random_flip_left_right(distort_image)
    #调用distort_color
    distorted_image = distort_color(distort_image,np.random.randint(1))
    return distorted_image
img_raw_data = tf.read_file("./1.jpg","r")
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(img_raw_data)
    # image = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    boxes = tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
    result = preprocess_for_train(img_data,299,299,boxes)
    plt.imshow(result.eval())
    plt.show()







