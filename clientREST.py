#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install cv2')
get_ipython().system('pip install imutils')
get_ipython().system('pip install tensorflow')


# In[8]:


import requests
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
import tensorflow as tf
import json

def get_predict(predict_text):
    strr=predict_text.split()
    predict=float(strr[4])
    if predict>0.8:
        return "Positivo"
    else:
        return "Negativo"
def proprocesa_img(img):
    #Es necesario someter a la imagen al mismo procedimiento que se sometieron
    #las imagenes con la que se entreno el modelo
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY) [1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    ADD_PIXELS = 0
    new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
    n_img = cv2.resize(new_img, dsize = IMG_SIZE,interpolation=cv2.INTER_CUBIC)
    return n_img

IMG_SIZE = (224,224)
endpoint = "http://127.0.0.1:8501/v1/models/RETOM2:predict"
img = cv2.imread("/home/benja/Python/2.jpg")
plt.imshow(img)
imagen = proprocesa_img(img)
json_data = { "inputs" : [imagen.tolist()] }
header={"content type":"application/json"}
response = requests.post(endpoint, json=json_data,headers=header)
print(response.text)
print("Diagnostico: ",get_predict(response.text))
 


# In[ ]:





# In[ ]:




