#building slmodel
from keras import layers
from keras import models
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import random


# loading images 
n_images= 1000
all_stimlus = np.zeros((n_images+ n_images, 224, 224, 3))

for i in range(n_images):
	img_path ='.figures/stimuli/feature_'+str(i+1)+'.png'
	img= image.load_img(img_path, target_size=(224, 224))
	img = image.img_to_array(img)
	img = np.expand_dims(img, axis=0)
	all_stimlus [i,:,:,:] = img

	
for j in range(n_images):
	img_path ='.figures/distractors/nofeature_'+str(i+1)+'.png'
	img = image.load_img(img_path, target_size= (224, 224))
	img = image.img_to_array(img)
	img = np.expand_dims(img, axis= 0)
	all_stimlus[i+j,:,:,:] = img

all_labels = np.vstack(np.zeros(n_images).reshape(n_images, 1), np.zeros(n_images).reshape(n_images, 1))

#concatenate data together and permute the data 
c =list(all_stimuli, all_labels)
random.shuffle(c)
a, b= zip(*c)
a_array, b_array = np.array(a), np.array(b)

# partition data into train, validation, and test data
train_stim=a_array[: 1500,:,:,:]
val_stim = a_array[1501:1751,:,:,:]
test_stim =a_array[1751:1999,:,:,:]
train_labels = b_array[:1500]
val_labels = b_array[:1500]
test_labels = b_array[:1500]

#building custom model for the analysis.

for epochs in [100, 500, 1000, 1500]:
	for n_hidden in range(5, 10):
		cont_model =models.Sequential()
		cont_model.add(layers.Conv2D(2**n_hidden , (3,3), activation='relu', input_shape=(224, 224, 3))
		cont_model.add(layers.MaxPooling2D((2,2)))
		cont_model.add(layers.Conv2D(2**(n_hidden +1), (3, 3), activation ='relu'))
		cont_model.add(layers.MaxPooling2D((2, 2)))
		cont_model.add(layers.Con2D(2**(n_hidden+2), (3,3), activation ='relu'))
		cont_model.add(layers.MaxPooling2D((2,2)))
		cont_model.add(layers.Conv2D(2**(n_hidden+3), (3,3), activation='relu')
		cont_model.add(layers.MaxPooling2D((2,2)))
		cont_model.add(layers.Flatten())
		cont_model.add(layers.Dropout(0.5))
		cont_model.add(layers.Dense(512, activation='relu'))
		cont_model.add(layers.Dense(1, activation='sigmoid'))

		cont_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


#training the  model.
		cont_model_history = cont_model.fit(train_stim, train_labels, epochs= epochs, batch_size=100, validation_data=(val_stim, val_labels))
		plt.plot(cont_model_history.history['loss'], -- ,cont_model_history.history['val_loss'], cont_model_history.history['acc'], '-',cont_model_history.history['val_acc'], '-')
		plt.savefig('.figure/results/train_epoch_'+str(epochs)+'_'+str(n_hidden)+'.png')


