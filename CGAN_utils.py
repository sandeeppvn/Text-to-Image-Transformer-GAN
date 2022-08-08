

########################################################################################################################
#                                            CLIP HELPERS                                                              #            
########################################################################################################################                                                              `

import clip
def clip_embed(text):
    text = "This is " +text
    return clip.tokenize(text)

########################################################################################################################
#                                            GAN HELPERS                                                               #
########################################################################################################################

import torch
# custom weights initialization called on gen and disc model
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)    


import torch.nn as nn
def generator_loss(fake_output, label):
    adversarial_loss = nn.BCELoss() 
    gen_loss = adversarial_loss(fake_output, label)
    #print(gen_loss)
    return gen_loss

def discriminator_loss(output, label):
    adversarial_loss = nn.BCELoss() 
    disc_loss = adversarial_loss(output, label)
    return disc_loss

########################################################################################################################
#                                            GENERAL HELPERS                                                           #
########################################################################################################################

import os
def create_directories(exp_name):
    if not os.path.exists('Experiments/'+exp_name):
        os.makedirs('Experiments/'+exp_name)

    # Create saved models directory if it doesn't exist
    if not os.path.exists('Experiments/'+exp_name+'/saved_models'):
        os.makedirs('Experiments/'+exp_name+'/saved_models')

    # Create Generator folder in saved models directory if it doesn't exist
    if not os.path.exists('Experiments/'+exp_name+'/saved_models/Generator'):
        os.makedirs('Experiments/'+exp_name+'/saved_models/Generator')

    # Create Discriminator folder in saved models directory if it doesn't exist
    if not os.path.exists('Experiments/'+exp_name+'/saved_models/Discriminator'):
        os.makedirs('Experiments/'+exp_name+'/saved_models/Discriminator')

    # Create best model directory if it doesn't exist
    if not os.path.exists('Experiments/'+exp_name+'/best_model'):
        os.makedirs('Experiments/'+exp_name+'/best_model')


    # Create saved images directory if it doesn't exist
    if not os.path.exists('Experiments/'+exp_name+'/saved_images'):
        os.makedirs('Experiments/'+exp_name+'/saved_images')

    # Create saved logs directory if it doesn't exist
    if not os.path.exists('Experiments/'+exp_name+'/saved_logs'):
        os.makedirs('Experiments/'+exp_name+'/saved_logs')

    # Create saved plots directory if it doesn't exist
    if not os.path.exists('Experiments/'+exp_name+'/saved_plots'):
        os.makedirs('Experiments/'+exp_name+'/saved_plots')

    return exp_name


import numpy as np
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy()
    #print(image.shape)
    image = image.squeeze()
    #print(image.shape)
    image = image.transpose(1,2,0)
    image = image * np.array((0.485, 0.456, 0.406)) + np.array((0.229, 0.224, 0.225))
    image = image.clip(0, 1)


    return image



