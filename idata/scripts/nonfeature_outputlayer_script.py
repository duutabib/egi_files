
#load model
from keras.applications import VGG19
from keras import backend as K

model = VGG19(weights='imagenet',
              include_top=False)

#layer_name = 'block3_conv1'
#filter_index = 0

#layer_output = model.get_layer(layer_name).output  # extract layer output
#loss = K.mean(layer_output[:, :, :, filter_index]) #this returns the computes the mean of along an axis.

# We preprocess the image into a 4D tensor
from keras.preprocessing import image
import numpy as np
from functools import reduce
from tqdm import tqdm
import glob, operator, time 

def deprocess_image(x):  # converts a tensor to a vaild image
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def generate_pattern(layer_name, filter_index, size=150): #putting all the code together
    # Build a loss function that maximizes the activation
    # of the nth filter of the layer considered.
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # Compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, model.input)[0]

    # Normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5) 

    # This function returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])
    
    # We start from a gray image with some noise
    #input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    input_img_data = img_tensor
    # Run gradient ascent for 40 steps
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)


start = time.time()
nf_block1_conv1_list=[]; nf_block2_conv1_list=[]; nf_block3_conv1_list=[]; nf_block4_conv1_list=[]; nf_block5_conv1_list=[]

for fig_path in tqdm(glob.glob('/0/abib/idata/nonfeature_stimulusdata/Figure*[0-9].png' )): #.need to change this line'85
    img = image.load_img(fig_path, target_size=(64, 64))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    # Remember that the model was trained on inputs
    # that were preprocessed in the following way:
    img_tensor /= 255.
    for layer in [ 'block3_conv1', 'block4_conv1', 'block5_conv1']:
            for i in tqdm(range(8)):
                for j in tqdm(range(8)):
                    pattern = generate_pattern(layer, i+(j*8), size =64)
                    pattern = pattern.reshape(1, reduce(operator.mul, pattern.shape))
                    if layer =='block1_conv1':
                        nf_block1_conv1_list.append(pattern)
		    if layer =='block2_conv1':
                        nf_block2_conv1_list.append(pattern)
		    if layer =='block3_conv1':
                        nf_block3_conv1_list.append(pattern)
                    elif layer =='block4_conv1':
                        nf_block4_conv1_list.append(pattern)                            
                    else:
                        nf_block5_conv1_list.append(pattern)

print((time.time() -start)/3600)}

#collecting files
numpy.savetxt('nfb1c1_list.txt',nf_block1_conv1_list, fmt='%s', delimiter=',')
numpy.savetxt('nfb2c1_list.txt',nf_block2_conv1_list, fmt='%s', delimiter=',')
numpy.savetxt('nfb5c1_list.txt',nf_block5_conv1_list, fmt='%s', delimiter=',')
numpy.savetxt('nfb3c1_list.txt',nf_block3_conv1_list, fmt='%s', delimiter=',')
numpy.savetxt('nfb4c1_list.txt',nf_block4_conv1_list, fmt='%s', delimiter=',')


