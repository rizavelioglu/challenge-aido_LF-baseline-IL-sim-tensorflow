# -*- coding: utf-8 -*-

### Imports

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam, SGD
from keras.losses import mean_squared_error as MSE
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model

import pickle
from _loggers import Reader

"""### Plot Losses"""

# Function to plot model's validation loss and validation accuracy
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(25,8))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.savefig(STORAGE_LOCATION+'/VGG16#8_model_history.png')
    
    plt.show()

"""### Config"""

# configuration zone
BATCH_SIZE = 32
EPOCHS = 50
# here we assume the observations have been resized to 60x80
OBSERVATIONS_SHAPE = (None, 60, 80, 3)
ACTIONS_SHAPE = (None, 2)
SEED = 1234
# TODO: change thepaths
STORAGE_LOCATION = "the path of the model that will be saved  ->  desktop/trained_models/"
DATA_LOCATION = "the path of training data   ->   desktop/data/"

"""### Load Data"""

reader = Reader(DATA_LOCATION+'train-v10.log')

observations, actions = reader.read()
actions = np.array(actions)
observations = np.array(observations)

x_train, x_validate, y_train, y_validate = train_test_split(observations, actions, test_size = 0.2, random_state = 2)

train_datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0, # Randomly zoom image 
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

train_datagen.fit(x_train)

validation_datagen = ImageDataGenerator()
validation_datagen.fit(x_validate)

"""### Model"""

base_model = VGG16(classes=2, input_shape=(60,80,3), weights=None, include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
predictions = Dense(2)(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# Define the optimizer
optimizer = SGD(lr=0.01, momentum=0.001, nesterov=False)
# optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# Compile the model
model.compile(optimizer = optimizer ,
              loss = MSE, 
              metrics=["accuracy"])

es = EarlyStopping(monitor='val_loss', verbose=1, patience=30)
mc = ModelCheckpoint(STORAGE_LOCATION+'VGG16#8.h5', monitor='val_loss', save_best_only=True)

history = model.fit_generator(train_datagen.flow(x_train,y_train, batch_size=BATCH_SIZE),
                              validation_data = validation_datagen.flow(x_validate,y_validate, batch_size=BATCH_SIZE),
                              epochs = EPOCHS,
                              verbose=2,  # for hiding print statements
                              steps_per_epoch=observations.shape[0] // BATCH_SIZE,
                              callbacks=[es, mc],
                              shuffle=True)
plot_model_history(history)

"""### Evaluate Model"""

from keras.models import load_model
model = load_model(STORAGE_LOCATION + 'VGG16#8.h5')
model.evaluate(x_validate,y_validate)

num = 20
preds = model.predict(x_validate[:num])

for i in range(num):
  print("Pred: ", preds[i], "\tGT: ", y_validate[i])
