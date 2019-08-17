from __future__ import absolute_import, division, print_function, unicode_literals
#导入所需要的包
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
#加载模型
model = tf.keras.models.load_model('model.h5')
#读取测试图片文件
image_gen_val = ImageDataGenerator(rescale=1./255)
test_data_gen = image_gen_val.flow_from_directory(batch_size=3,
                                                 directory='F:/PycharmProjects/img-rec/test/dog',
                                                 target_size=(150, 150),
                                                 class_mode=None)
#进行预测
result = model.predict(test_data_gen)
predicted_class=[]
for i in range(0,515):
     predicted_class.append(np.argmax(result[i], axis=-1))
predicted_class=np.array(predicted_class)
#测试结果中非零的个数
cishu =np.count_nonzero(predicted_class)
#正确率或错误率
print(cishu/np.size(predicted_class))