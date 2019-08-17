from __future__ import absolute_import, division, print_function, unicode_literals
#导入所需要的包
import os
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import tensorflow_hub as hub
from keras_preprocessing.image import ImageDataGenerator
#训练集cats文件夹下的图片数量
num_cats_tr = len(os.listdir('F:/PycharmProjects/img-rec/transfer_train/cats'))
#训练集dogs文件夹下的图片数量
num_dogs_tr = len(os.listdir('F:/PycharmProjects/img-rec/transfer_train/dogs'))
#总共的训练用的图片数量
total_train = num_cats_tr + num_dogs_tr
BATCH_SIZE = 20                      #每一次训练的数量
IMAGE_RES = 224                      #输入层的图片大小
#模型网址
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
#读取训练所需要的文件
image_gen_val = ImageDataGenerator(rescale=1./255)
train_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE,
                                                 directory='F:/PycharmProjects/img-rec/transfer_train',
                                                 target_size=(IMAGE_RES, IMAGE_RES),
                                                 class_mode='binary')

#从网址下载模型
feature_extractor = hub.KerasLayer(URL,input_shape=(IMAGE_RES, IMAGE_RES,3))
#设置除了输出层外其他层的参数不可辨
feature_extractor.trainable = False
#创建模型
model = tf.keras.Sequential([
  feature_extractor,
  tf.keras.layers.Dense(2, activation='softmax')
])
#编译模型
model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])
#训练模型
epochs=6
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=epochs
)
#储存模型
model.save('transfer_model.h5')