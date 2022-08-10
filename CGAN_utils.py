

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
from numpy.random import random
def smooth_positive_labels(y):
    return y - 0.3 + (random(y.shape) * 0.5)


def smooth_negative_labels(y):
	return y + random(y.shape) * 0.3

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
    # Use reduce_mean on sigmoid cross entropy loss to average across batch
    adversarial_loss = nn.BCELoss() 
    gen_loss = adversarial_loss(fake_output.double(), label)
    #print(gen_loss)
    return gen_loss

def discriminator_loss(output, label):
    # adversarial_loss = nn.BCEWithLogitsLoss()
    adversarial_loss = nn.BCELoss()
    disc_loss = adversarial_loss(output.double(), label)
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
    image = tensor.to("cpu").clone().detach().numpy().squeeze().transpose(1,2,0)

    # Denormalize image
    image = image * np.array((0.485, 0.456, 0.406)) + np.array((0.229, 0.224, 0.225))
    image = image.clip(0, 1)

    return image


import matplotlib.pyplot as plt
def save_image(fake_img, real_img, caption, path):
    fake_img = im_convert(fake_img)
    real_img = im_convert(real_img)

    fig, axs = plt.subplots(1, 2, figsize=(15, 15))
    axs[0].imshow(fake_img)
    axs[1].imshow(real_img)

    axs[0].axis('off')
    axs[1].axis('off')

    plt.suptitle(caption, fontsize=18)
    
    plt.savefig(path)

def save_loss_plot(G_loss, D_real_loss, D_fake_loss, D_total_loss, path):
    plt.figure(figsize=(10,10))
    # Use different colors and markers for each loss
    plt.plot(G_loss, c='#1f77b4', label='Generator loss')
    plt.plot(D_real_loss, c='#ff7f0e', label='Discriminator loss on real images')
    plt.plot(D_fake_loss, c='#2ca02c', label='Discriminator loss on fake images')
    plt.plot(D_total_loss, c='#d62728', label='Total discriminator loss')

    # Add axis labels and legend
    plt.xlabel('epochs')
    plt.ylabel('loss')

    plt.legend()
    plt.savefig(path)
