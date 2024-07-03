#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initialising the CNN
CNN_Classifier=Sequential();

#Step 1 - Convolution
CNN_Classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation="relu"))

#Step 2 - Pooling
CNN_Classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step 1 - Convolution
CNN_Classifier.add(Conv2D(16,(3,3),activation="relu"))

#Step 2 - Pooling
CNN_Classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step 3 - Flattening
CNN_Classifier.add(Flatten())



#Step 4 - Full Connection
CNN_Classifier.add(Dense(units=128,activation='relu'))
CNN_Classifier.add(Dense(units=1,activation='sigmoid'))

#Compiling the CNN
CNN_Classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[6]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory("C:/Users/DELL/Downloads/facedata/Face Mask Dataset/Test",
                                                target_size = (64,64),
                                                batch_size = 32,
                                                class_mode = 'binary')

test_set = test_datagen.flow_from_directory("C:/Users/DELL/Downloads/facedata/Face Mask Dataset/Test",
                                                target_size = (64,64),
                                                batch_size = 32,
                                                class_mode = 'binary')

CNN_Classifier.fit_generator(training_set,
                            steps_per_epoch = 7000,
                            epochs = 10,
                            validation_data = test_set,
                            validation_steps = 2000)


# In[9]:


import numpy as np
from keras.preprocessing import image
test_image=image.load_img("C:/Users/DELL/Downloads/facedata/Face Mask Dataset/Validation/WithoutMask/54.png",target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=CNN_Classifier.predict(test_image)
training_set.class_indices
if result[0][0]==1:
    prediction='face without mask'
    print(prediction)
else:
    prediction='face with mask'
    print(prediction)


# In[ ]:




