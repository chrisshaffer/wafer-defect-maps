import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical

# Loads and cleans a sample of wafer data
def load_data(frac):
	# Load data into pandas df
	file_loc = '/home/chris/DSI/wafer-defect-maps/data/wm-811k/LSWMD.pkl'
	df=pd.read_pickle(file_loc)
	df = df.rename(columns={"trianTestLabel": "trainTestLabel"})

	# Drop extraneous columns
	df = df.drop(columns=['dieSize', 'lotName', 'waferIndex'])

	# Sample data for testing
	df = df.sample(frac=frac)

	# Map labels for fault and train/test
	mapping_type={'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}
	mapping_traintest={'Training':0,'Test':1}
	df=df.replace({'failureType':mapping_type, 'trainTestLabel':mapping_traintest})

	# Select only failed wafers
	df = df[(df.failureType >= 0) & (df.failureType < 8)]

	# Split into train/test
	train = df[df.trainTestLabel == 0]
	test = df[df.trainTestLabel == 1]

	X_train, Y_train = reshape_for_tf(train)
	X_test, Y_test = reshape_for_tf(test)
 
	return X_train, X_test, Y_train, Y_test

# Converts df into X and y, where X is formatted for tensorflow
def reshape_for_tf(df):
    X = df['waferMap'].values
    y = df['failureType']
    
    # convert class vectors to binary class matrices (don't change)
    Y = to_categorical(y, 8)
    
    # Resize to uniform img shape
    X_resize = np.array([cv2.resize(img, dsize=(24,24)) for img in X])
    
    # Reshape for tf
    X_tf = X_resize.reshape(X_resize.shape[0], 24, 24, 1).astype('float32')
    
    return X_tf, Y
