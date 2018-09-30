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

def arr2tag(arr1, arr2):
    tags = []
    for i in range(arr1.shape[0]):
        tag = []
        
        index1 = np.where(arr1[i] > 0.3 )
        
        index2 = np.where(arr2[i] > 0.3 )
        index1 = index1[0].tolist()
        index2 = index2[0].tolist()
        index = list(set(index1).union(set(index2)))
        #index = np.where((arr1[i] > 0.3) or (arr2[i] > 0.3) )
        #index = index[0].tolist()
        tag =  [hash_tag[j] for j in index]

        tags.append(tag)
    return tags

filepath = "valid_tags.txt"
hash_tag = hash_tag(filepath)

hash_tag[1]

from keras.models import *

model = load_model('model1.h5')
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

y_pred1 = model.predict(X_test)

del model
model = load_model('model2.h5')
y_pred2 = model.predict(X_test)

#y_pred = y_pred1 + y_pred2
y_tags = arr2tag(y_pred1, y_pred2)

import os
img_name = os.listdir('test/')
img_name[:10]

df = pd.DataFrame({'img_path':img_name, 'tags':y_tags})
for i in range(df['tags'].shape[0]):
    df['tags'].iloc[i] = ','.join(str(e) for e in  df['tags'].iloc[i])
df.to_csv('submit2.csv',index=None)

df.head()



