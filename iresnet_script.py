
# load modules for task
from keras.applications import InceptionResNetV2 as iresnet 
import h5py as h5py
from keras.preprocessing  import image
from keras.models import Model
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time 

start= time.time()

# load xception model
model = iresnet(weights='imagenet', include_top = False)

# define function for cmoputing area under ROC
def aucROC(sample1, sample2):
    sample_list=np.array([sample1, sample2 ] )
    list_usample = np.unique(np.around(sample_list,2))
    sorted_list = sorted(list_usample)
    x = np.zeros((list_usample.size+2, 1))
    y = np.zeros((list_usample.size+2, 1))
    for i,j in enumerate(sorted_list):
        x[i+1,]=len(np.where(sample1> j)[0])
        y[i+1,]=len(np.where(sample2> j)[0])
    x = x/sample1.size
    y = y/sample2.size 
    x[0]=1 ; y[0]=1
    auc = np.around(sum(-np.diff(x, axis=0)*(y[1:]+ y[:-1])/2),3)
    return auc[0] if auc>=0.5 else 1 - auc[0]


#load image data into array and define activation model
n_images= 100
stim= np.zeros((n_images, 224, 224, 3))
dis= np.zeros((n_images, 224, 224, 3))


for i in range(100):

	img_path = '/0/abib/figures/circlefeature/feature_'+str(i+1)+'.png'
	img = image.load_img(img_path, target_size=(224, 224))
	img = image.img_to_array(img)
	img =np.expand_dims(img, axis=0)
	stim[i]= img

for k in range(100):

	img_path = '/0/abib/figures/nonfeature/nofeature_'+str(k+1)+'.png'
	img = image.load_img(img_path, target_size=(224, 224))
	img = image.img_to_array(img)
	img =np.expand_dims(img, axis=0)
	dis [k]=  img

##define act model
outs = [ layer.output for layer in model.layers[:780]]
act_model= Model (inputs= model.inputs, outputs= outs)


## compute activations for all gimages data for the network model
stim_acts = act_model.predict(stim)
dis_acts = act_model.predict(dis)

# loop through layers and filters and compute auc for the data at each filter.
max_auc= np.zeros(len(model.layers[:780]))
for layer_index in range(len(model.layers[:780])):
	auc_filters = np.zeros((stim_acts[layer_index].shape[3]))
	for filter_index in range(stim_acts[layer_index].shape[3]):
		stim_sample = np.around(stim_acts[layer_index][:,:,:,filter_index].astype('float'), decimals=2)
		dis_sample = np.around(dis_acts[layer_index][:,:,:,filter_index].astype('float'), decimals=2)
		auc= aucROC(stim_sample, dis_sample)
		auc_filters[filter_index] = auc

	plt.plot(auc_filters, 'b-')
	plt.xlabel('Filters Index')
	plt.ylabel('auRoC')
	plt.grid(True)
	plt.title('InceptionResNetV2'+str(model.layers[layer_index].name)+': layer '+str(layer_index) )
	plt.savefig('/0/abib/cluster_inceptionresnet_plots/inceptionresnetlayer_'+str(layer_index)+'.png')
	max_auc= max(auc_filters)	

plt.plot(max_auc, 'b-')
plt.xlabel('Layer Index')
plt.ylabel('auRoC')
plt.grid(True)
plt.title(' Xception Layer ' )
plt.savefig('/0/abib/cluster_inceptionresnet_plots/inceptionresnetlayer_maxauc.png')


#write the time taken to file.
t_taken= time.time()- start
fh = open("/0/abib/cluster_inceptionresnet_plots/time_taken.txt","w")
fh.write(t_taken)
fh.close()
