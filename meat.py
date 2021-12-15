import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    model = tf.keras.models.load_model('./model')
    image=load_img("./fresh.jpg",target_size=(100,100))
    image=img_to_array(image) 
    image=image/255.0
    prediction_image=np.array(image)
    prediction_image= np.expand_dims(image, axis=0)
    prediction=model.predict(prediction_image)
    value=np.argmax(prediction)
    print("Prediction is {}.".format(prediction))
    print("Value  is {}.".format(value))



