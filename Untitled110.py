#!/usr/bin/env python
# coding: utf-8

# # Bulding a CNN for image classification- CIFAR-10

# IMPORT LIBRARIES

# In[12]:


import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Flatten,Activation,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.models import load_model
import os


# Load the CIFAR-10 datesets

# In[13]:


(x_train,y_train),(x_test,y_test)=cifar10.load_data()


# Data shape 

# In[14]:


print('x_train.shape:',x_train.shape)
print('x_test.shape:',x_test.shape)
print('train_sample',x_train.shape[0])
print('x_test_sample',x_test.shape[0])


# format datatype and normalize

# In[15]:


x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train /=255
x_test /=255
batch_size = 32
num_classes = 18
epochs = 10
data_augmentation = True


# one hot encode for targets data

# In[16]:


y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)


# Models Bulding

# In[17]:


model=Sequential()
# padding ='same' result in padding the input such that
#output has same length as the orginal input

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

#intialize RMSprop optimizer and configure some parametrs
opt=keras.optimizers.rmsprop(lr=0.0001,decay=1e-6)


#compile the model
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
model.summary()
        


# train the model

# In[20]:


hist=model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test,y_test),shuffle=True)


# In[21]:


model.save('E')
score=model.evaluate(x_test,y_test,verbose=1)
print('test loss',score[0])
print('test accuracy',score[1])


# In[22]:


import cv2
import numpy as np
from keras.models import load_model

im_rows,im_cols,im_depth=32,32,3
classifier=load_model('E')
color=True
scale=8
def draw_test(nam,res,input_im,scale,im_rows,im_cols):
    BLACK=[0,0,0]
    res=int(res)
    if res==0:
        pred='airplane'
    if res==1:
        pred='automobile'
    if res==2:
        pred='bird'
    if res==3:
        pred='cat'
    if res==4:
        pred='deer'
    if res==5:
        pred='dog'
    if res==6:
        pred='frog'
    if res==7:
        pred='horse'
    if res==8:
        pred='ship'
    if res==9:
        pred='truck'
    expended_image=cv2.copyMakeBorder(input_im,0,0,0,imageL.shape[0]*2,cv2.BORDER_CONSTANT,value=BLACK)
    if color==False:
        expended_image=cv2.cvtColor(expended_image,cv2.COLOR_GRAY2BGR)
    cv2.putText(expended_image,str(pred),(300,80),cv2.FONT_HERSHEY_COMPLEX_SMALL,3,(0,255,0),2)
    cv2.imshow(nam,expended_image)


# In[ ]:





# In[ ]:


for i in range(0,10): 
    rand=np.random.randint(0,len(x_test))
    input_im=x_test[rand]
    imageL=cv2.resize(input_im,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    input_im=input_im.reshape(1,im_rows,im_cols,im_depth)
    
    res=str(classifier.predict_classes(input_im,1,verbose=0)[0])
    draw_test("prediction",res,imageL,scale,im_rows,im_cols)
    

    cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




