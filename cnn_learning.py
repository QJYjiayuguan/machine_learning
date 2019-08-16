from __future__ import absolute_import, division, print_function, unicode_literals
#导入所需要的包
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
#训练集cats文件夹下的图片数量
num_cats_tr = len(os.listdir('F:/PycharmProjects/img-rec/train/cats'))
#训练集dogs文件夹下的图片数量
num_dogs_tr = len(os.listdir('F:/PycharmProjects/img-rec/train/dogs'))
#验证集dogs文件夹下的图片数量
num_cats_val = len(os.listdir('F:/PycharmProjects/img-rec/validation/cats'))
#验证集dogs文件夹下的图片数量
num_dogs_val = len(os.listdir('F:/PycharmProjects/img-rec/validation/dogs'))
#总共的训练用的图片数量
total_train = num_cats_tr + num_dogs_tr
#总共的验证用的图片数量
total_val = num_cats_val + num_dogs_val
BATCH_SIZE = 100                      #每一次训练的数量
IMG_SHAPE  = 150                      #输入的图片大小
#绘图函数
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()
#图像增强函数，包括旋转，伸缩，平易等
image_gen_train = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
)
#从train文件夹读取指定的文件
train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory='F:/PycharmProjects/img-rec/train',
                                                     shuffle=True,
                                                     target_size=(IMG_SHAPE,IMG_SHAPE),
                                                     class_mode='binary')
#从validation文件夹读取指定的文件
image_gen_val = ImageDataGenerator(rescale=1./255)
val_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE,
                                                 directory='F:/PycharmProjects/img-rec/validation',
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode='binary')
#创建模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
#编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#开始训练模型
epochs=100     #训练的次数
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)
model.save('model.h5')
