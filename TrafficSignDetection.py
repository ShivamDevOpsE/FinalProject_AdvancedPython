# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 08:26:02 2020

@author: shiva
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils as np_utils
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score


data = []  # creating a list
labels = []
classes = 43   #it shows the different types of signs

cur_path = "C:\\Users\\shiva\\Desktop\\TrafficSign_Database"

print(cur_path)            

for i in range(classes):
    path = os.path.join(cur_path, 'Train', str(i))
    images = os.listdir(path)
    
    for a in images:
        try:
            image = Image.open(path + '\\' + a)
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)     # store all the images in a list
            labels.append(i)       # store all the class labels in a list
            
        except:
            print("Something went wrong")
 
# convert the list into numpy array for model           
data = np.array(data)      
labels = np.array(labels)

# it must print no. of images in the data with no of pixel and RGB value
print(data.shape, labels.shape) 

# split the data into training and testing set with 80 to 20 ratio
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, 
                                                    random_state = 42)

# X indicates independent features and y dependent features
# it returns the no of images, pixel values and XGB color code
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


y_train = np_utils.to_categorical(y_train, 43)    # to change labeled data into binary metrix
y_test = np_utils.to_categorical(y_test, 43)


#Building CNN model to classify images into respective categories
model = models.Sequential()    # build a simple model in layers

''' adding 4 convolution layers to manage input images in 2D metrics
    filters indicated no of nodes in the layers depends on size of datasets
    kernel_size indicated 5*5 filter matrix.
    ReLU for neural network
'''
model.add(layers.Conv2D(32, (5, 5), activation = 'relu',
                 input_shape = X_train.shape[1:]))
model.add(layers.Conv2D(32, (5, 5), activation = 'relu'))
model.add(layers.MaxPool2D((2,2)))    # to downsample the input
model.add(layers.Dropout(rate = 0.25))  # using dropout to reduce overfitting
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPool2D((2, 2)))

model.add(layers.Dropout(rate = 0.25))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dropout(rate = 0.5))
model.add(layers.Dense(43, activation = 'softmax'))

model.summary()

#Model compilation for multiple classes to categorise
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])

#in 15 iterations, it is getting around 96% accuracy
hist = model.fit(X_train, y_train, batch_size = 256, epochs = 10,
          validation_data = (X_test, y_test))

plt.figure(0)  #create image object
plt.plot(hist.history['accuracy'], label = 'training accuracy')  #training set accuracy
plt.plot(hist.history['val_accuracy'], label = 'val accuracy')   #validation set accuracy
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.figure(1)
plt.plot(hist.history['loss'], label = 'training loss')    #training set loss
plt.plot(hist.history['val_loss'], label = 'val loss')     #validation set loss
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()


# testing of our model on test dataset
y_test = pd.read_csv("C:\\Uni_Regensburg\\SS2020\\Advanced_Python\\\
Final_Project\\Traffic_Signs_Detection\TrafficSign_Database\\Test.csv")

#Extracting image labels and paths from Test.csv file 
labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data = []

for img in imgs:
    image = Image.open(img)
    # resizing the data to predict the model
    image = image.resize((30, 30)) 
    # Storing all the images in numpy array
    data.append(np.array(image))
    
X_test = np.array(data, dtype = np.float32)
pred = model.predict_classes(X_test) #to predict multiclass vector(image and labels)

#Accuracy of predicted labels vs actual labels
print(accuracy_score(labels, pred))

#Save the model in HDF5 file as a dictionary
model.save('traffic_detector.h5')  

#loading the trained model
model = load_model('traffic_detector.h5')

import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

#class dictionary to label all the traffic signs
classes = {1: 'Speed limit (20km/h)',
           2: 'Speed limit (30km/h)',
           3: 'Speed limit (50km/h)',
           4: 'Speed limit (60km/h)',
           5: 'Speed limit (70km/h)',
           6: 'Speed limit (80km/h)',
           7: 'End of Speed limit (80km/h)',
           8: 'Speed limit (100km/h)',
           9: 'Speed limit (120km/h)',
           10: 'No Passing',
           11: 'No passing weigh over 3.5 tons',
           12: 'Right-of-way at intersection',
           13: 'Priority road',
           14: 'Yield',
           15: 'Stop',
           16: 'No Vehicle',
           17: 'Weigh > 3.5 tons prohibited',
           18: 'No entry',
           19: 'General Caution',
           20: 'Dangerous curve left',
           21: 'Dangerous curve right',
           22: 'Double curve',
           23: 'Bumpy road',
           24: 'Slippery Road',
           25: 'Road narrow on right',
           26: 'Road work',
           27: 'Traffic signals',
           28: 'Pedestrians',
           29: 'Children crossing',
           30: 'Bicycle crossing',
           31: 'Beware of Ice/Snow ',
           32: 'Wild animals crossing',
           33: 'End speed + passing limit',
           34: 'Turn right ahead',
           35: 'Turn left ahead',
           36: 'Ahead',
           37: 'Go straight or right',
           38: 'Go straight or left',
           39: 'Keep right',
           40: 'Keep left',
           41: 'Roundabout mandatory',
           42: 'End of no passing',
           43: 'End no passing weigh > 3.5 tons' }

# Initialize GUI
top = tk.Tk()      #creating main window
top.geometry('800x600')    #with given width and height
top.title('Traffic Sign Detection')
top.configure(background = '#ABCDEF')  #using hex color light blue
              
label = Label(top, background = '#ABCDEF', font = ('arial', 15, 'bold'))
sign_image = Label(top)

#classification of image by classify() function
def classify(file_path):    #dynamic image path to get the image
    global label_packed
    image = Image.open(file_path)
    image = image.resize((30, 30))      #resizing to match the image dimension as we used when training
    image = np.expand_dims(image, axis = 0)   #to load the actual image as array
    image = np.array(image, dtype = np.float32)  #we need to use single precision bit
    pred = model.predict_classes([image])[0]      #prediction of image wrt to its class(1-43)
    sign = classes[pred + 1]
    print(sign)
    label.configure(foreground = '#380116', text = sign)

#this function is used to maintain the button classify image                    
def classify_button(file_path):
    #using lambda function for the callback of classify func
    classify_b = Button(top, text = 'Classify Image', command = lambda: classify(
            file_path), padx = 10, pady = 5)
    classify_b.configure(background = '#404d66', foreground = 'black', font = 
                         ('arial', 10, 'bold'))
    classify_b.place(relx = 0.79, rely = 0.46)    #Horizontal and vertical offsets
    
def upload_image():
    try:
        file_path = filedialog.askopenfilename()     #user entered image path
        uploaded = Image.open(file_path)
        #dividing it by specific number to adjust it in the tkinter window
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        
        sign_image.configure(image = im)
        sign_image.image = im
        label.configure(text = '')
        classify_button(file_path)
        
    except:
        pass

#adding a widget with adjusting spaces from right, left and top, bottom    
upload = Button(top, text = 'Upload Image', command = upload_image, padx = 10, pady = 5)
upload.configure(background = '#380116', foreground = 'white', font = ('arial', 10, 'bold'))
  
#packing the parent widget wrt to bottom
upload.pack(side = BOTTOM, pady = 50)
sign_image.pack(side = BOTTOM, expand = True)
label.pack(side = BOTTOM, expand = True)

heading = Label(top, text = "Predict Traffic Sign", pady = 20, font = ('arial', 20, 'bold'))
heading.configure(background = '#ABCDEF', foreground = '#380116')

heading.pack()
top.mainloop()    #the run the application until the window will not get closed
    