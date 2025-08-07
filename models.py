from django.db import models

# Create your models here.
from django.db import models

# Create your models here.
from django.db import models
from keras.models import load_model
# import cv2
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import json
from PIL import Image



cnn = load_model(r'C:\Users\strydo\Desktop\PROJECTS SEICOM\FIRE DETECTION\FIRE DATASET\front end\fire_CNN.h5')
vgg16 = load_model(r'C:\Users\strydo\Desktop\PROJECTS SEICOM\FIRE DETECTION\FIRE DATASET\front end\fire_VGG16.h5')





def predict(img,algo): 
	file = Image.open(img)
	img = file.convert('RGB')
	img_bgr= img.resize((224, 224))
	img_bgr = np.array(img_bgr)
	
	#res = cv2.resize(img,(224,224), interpolation = cv2.INTER_CUBIC)
	res = img_bgr.reshape([-1,224, 224,3])
	#res = img_array.reshape(-1, 224, 224, 3)
	print(res.shape)
	if algo=='cnn':
		y_pred=cnn.predict(res)
		return y_pred[0]
	else:
		y_pred=vgg16.predict(res)
		return y_pred[0]

