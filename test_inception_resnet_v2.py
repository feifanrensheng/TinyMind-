import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
#%matplotlib inline
    
from glob import glob
from tqdm import tqdm
    
import cv2
from PIL import Image
from imgaug import augmenters as iaa
#from keras.callbacks import TensorBoard
#from keras.utils import plot_model

train_path = 'visual_china_train1.csv'
train_df = pd.read_csv(train_path)
train_df.head()

train_df.shape

for i in range(35000):
    train_df['img_path'].iloc[i] = train_df['img_path'].iloc[i].split('/')[-1]

img_paths = list(train_df['img_path'])

def hash_tag(filepath):
    fo = open(filepath, "r",encoding='utf-8')
    hash_tag = {}
    i = 0
    for line in fo.readlines():     
        line = line.strip()      
        hash_tag[i] = line
        i += 1
    return hash_tag

def load_ytrain(filepath):  
    y_train = np.load(filepath)
    y_train = y_train['tag_train']
    
    return y_train

def arr2tag(arr):
    tags = []
    for i in range(arr.shape[0]):
        tag = []
        index = np.where(arr[i] > 0.3)  
        index = index[0].tolist()
        tag =  [hash_tag[j] for j in index]

        tags.append(tag)
    return tags

filepath = "valid_tags.txt"
hash_tag = hash_tag(filepath)

hash_tag[1]

y_train = load_ytrain('tag_train.npz')
y_train.shape
#(35000, 6941)

nub_train = 35000  
X_train = np.zeros((nub_train,299,299,3),dtype=np.uint8)
i = 0

for img_path in img_paths[:nub_train]:
    img = Image.open('train/' + img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((299,299))
    arr = np.asarray(img)
    X_train[i,:,:,:] = arr
    i += 1

seq = iaa.Sequential([
    iaa.CropAndPad(percent=(-0.1, 0.1)), 
    iaa.Sometimes(0.5,
    iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    iaa.ContrastNormalization((0.75, 1.5)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255)),
], random_order=True) 
imglist=[]
imglist.append(X_train)
images_aug = seq.augment_images(X_train)

from sklearn.model_selection import train_test_split
X_train2,X_val,y_train2,y_val = train_test_split(X_train, y_train[:nub_train], test_size=0.01, random_state=2018)

from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
from keras.models import load_model
from keras.applications.inception_resnet_v2 import InceptionResNetV2,preprocess_input
from keras.utils.training_utils import multi_gpu_model
#from keras.applications.inception_resnet_v2 import InceptionResNetV2,preprocess_input
#from keras.applications.densenet import DenseNet201,preprocess_input

#base_model = DenseNet201(weights='imagenet', include_top=False)
base_model = InceptionResNetV2(weights='imagenet',include_top=False)
#base_model = InceptionResNetV2(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x) 
x = Dense(1024,activation='relu')(x)
predictions = Dense(6941,activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
#plot_model(model,to_file='model.png')

model.summary()

import keras.backend as K

def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.
    if beta < 0:
    	raise ValueError('The lowest choosable beta is zero (only precision).')
    
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)

def setup_to_transfer_learning(model, base_model):
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',fmeasure,recall,precision])

def setup_to_fine_tune(model,base_model):
    GAP_LAYER = 17
    for layer in base_model.layers[:GAP_LAYER+1]:
        layer.trainable = False
    for layer in base_model.layers[GAP_LAYER+1:]:
        layer.trainable = True
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',fmeasure,recall,precision])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(width_shift_range = 0.1, 
                                 height_shift_range = 0.1, 
                                 zoom_range = 0.1,
                                 horizontal_flip= True)
val_datagen = ImageDataGenerator()   

batch_size = 128

train_generator = train_datagen.flow(X_train2,y_train2,batch_size=batch_size,shuffle=False) 
val_generator = val_datagen.flow(X_val,y_val,batch_size=batch_size,shuffle=False)

checkpointer = ModelCheckpoint(filepath='weights_best_simple_model.hdf5', 
                            monitor='val_fmeasure',verbose=1, save_best_only=True, mode='max')
reduce = ReduceLROnPlateau(monitor='val_fmeasure',factor=0.5,patience=2,verbose=1,min_lr=1e-4)

model = multi_gpu_model(model, gpus=4)
setup_to_transfer_learning(model, base_model)
history_t1 = model.fit_generator(train_generator,
                                steps_per_epoch=274,
                                validation_data = val_generator,
                                epochs=10,
                                callbacks=[reduce],
                                verbose=1
                               )


setup_to_fine_tune(model,base_model)
history_ft = model.fit_generator(train_generator,
                                steps_per_epoch=274,
                                epochs=8,
                                validation_data=val_generator,
                                validation_steps=10,
                                callbacks=[reduce],
                                verbose=1)


nub_test = len(glob('test/*'))
X_test = np.zeros((nub_test,299,299,3),dtype=np.uint8)
path = []
i = 0
for img_path in tqdm(glob('test/*')):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((299,299))
    arr = np.asarray(img)
    X_test[i,:,:,:] = arr
    i += 1

y_pred = model.predict(X_test)

model.save('model1.h5')

from pandas import DataFrame

data= DataFrame(y_pred)
data.to_csv('data1.csv')


y_tags = arr2tag(y_pred)

import os
img_name = os.listdir('test/')
img_name[:10]

df = pd.DataFrame({'img_path':img_name, 'tags':y_tags})
for i in range(df['tags'].shape[0]):
    df['tags'].iloc[i] = ','.join(str(e) for e in  df['tags'].iloc[i])
df.to_csv('submit2.csv',index=None)

df.head()
