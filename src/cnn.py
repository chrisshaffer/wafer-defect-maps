from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import cv2

np.random.seed(1)  # for reproducibility

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


def define_model(nb_filters, kernel_size, input_shape, pool_size):
    model = Sequential() # model is a linear stack of layers

    model.add(Conv2D(nb_filters,(kernel_size[0], kernel_size[1]),
                        padding='valid',
                        input_shape=input_shape)) #first conv. layer  KEEP
    model.add(Activation('relu')) # Activation specification necessary for Conv2D and Dense layers

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), padding='valid')) #2nd conv. layer KEEP
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=pool_size)) # decreases size, helps prevent overfitting
    model.add(Dropout(0.5)) # zeros out some fraction of inputs, helps prevent overfitting

    model.add(Flatten()) # necessary to flatten before going into conventional dense layer  KEEP
    print('Model flattened out to ', model.output_shape)

    # now start a typical neural network
    model.add(Dense(32)) # (only) 32 neurons in this layer, really?   KEEP
    model.add(Activation('relu'))

    model.add(Dropout(0.5)) # zeros out some fraction of inputs, helps prevent overfitting

    model.add(Dense(nb_classes)) # 10 final nodes (one for each class)  KEEP
    model.add(Activation('softmax')) # softmax at end to pick between classes 0-9 KEEP

    # many optimizers available, see https://keras.io/optimizers/#usage-of-optimizers
    # suggest you KEEP loss at 'categorical_crossentropy' for this multiclass problem,
    # and KEEP metrics at 'accuracy'
    # suggest limiting optimizers to one of these: 'adam', 'adadelta', 'sgd'
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # important inputs to the model: don't changes the ones marked KEEP
    batch_size = 32  # number of training samples used at a time to update the weights
    nb_classes = 8   # number of output possibilites
    nb_epoch = 2     # number of passes through the entire train dataset before weights "final"
    img_rows, img_cols = 24, 24  # the size of the MNIST images KEEP
    input_shape = (img_rows, img_cols, 1)  # 1 channel image input (grayscale) KEEP
    nb_filters = 32  # number of convolutional filters to use
    pool_size = (2, 2) # pooling decreases image size, reduces computation, adds translational invariance
    kernel_size = (5, 5) # convolutional kernel size, slides over image to learn features
    frac = 1 # Fraction of data to sample

    X_train, X_test, Y_train, Y_test = load_data(frac)

    model = define_model(nb_filters, kernel_size, input_shape, pool_size)

    # during fit process watch train and test error simultaneously
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
            verbose=1, validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1]) # this is the one we care about