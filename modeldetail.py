import tensorflow as tf
# from keras.preprocessing import image
import keras.utils as ku
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras import backend as K


# loading TensorFlow SavedModel use api ("/download_model/{taskName}")

model = tf.keras.models.load_model('exported_model_test/temp/iop@iop.com/raw_model')
model.summary()
model_label = ["Lemon","Orange","Apple","Banana","Grape","Strawberry","Pineapple","Watermelon","Peach","Pear"]

# 加载并预处理图像
img_path = 'LemonFB.jpg'
img = ku.load_img(img_path, target_size=(224, 224))
img_array = ku.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)  # 根据您的模型选择合适的预处理函数


# 查找最後一個卷積層的名稱
conv_layers = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
last_conv_layer_name = conv_layers[-1] if conv_layers else None

print("模型中的卷積層名稱：", conv_layers)
print("模型中最後一個卷積層的名稱：", last_conv_layer_name)





def gradcam(model, target_image):
     
    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer(last_conv_layer_name)  
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(target_image)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)

    heatmap_shape = (grads.shape[1], grads.shape[2])

    # ヒートマップの値を正規化
    heatmap_Emphasis = np.maximum(heatmap, 0) # ReLU
    heatmap_Emphasis /= np.max(heatmap_Emphasis) # 正規化
    heatmap_Emphasis = heatmap_Emphasis.reshape(heatmap_shape)
    plt.matshow(heatmap_Emphasis)
    plt.show()

    return heatmap_Emphasis

def plot_heatmap(heatmap, img_path):
   
    
    # 讀取影像
    img = cv2.imread(img_path)
    
    fig, ax = plt.subplots()
    
    im = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (img.shape[1], img.shape[0]))

    # 拉伸 heatmap
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)
    
    # 以 0.6 透明度繪製原始影像
    ax.imshow(im, alpha=0.6)
    
    # 以 0.4 透明度繪製熱力圖
    ax.imshow(heatmap, cmap='jet', alpha=0.4)
    
    
    
    plt.show()
# 确保找到 MobileNetV2 模型中最后一个卷积层名称
if last_conv_layer_name:
    print(img_array)
    heatmap= gradcam(model, img_array)

    plot_heatmap(heatmap, img_path=img_path)
else:
    print("模型中没有找到卷积层")
