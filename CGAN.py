import torch
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms


import CGAN_utils as utils
from CGAN_Model import Generator, Discriminator, TextModel
from CGAN_Dataset import CGAN_Dataset
import CGAN_DataTransforms as data_transforms

import os
import glob
import gdown
import pandas as pd
from tqdm import tqdm
import pickle

import time

# Set current working directory to the folder where the script is located
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# congif file
import configparser

config_path = 'CGAN_Config.ini'
config = configparser.ConfigParser()
config.read(config_path)

data_path = config['PATH']['data_path']
exp_name = config['PATH']['exp_name']


########################################################################################################################
#                                            Data Preparation                                                          #
########################################################################################################################

print('Creating directories')
# Create Directories if they don't exist
utils.create_directories(exp_name)


# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)

print('Downloading data')
# Download the TallyQA data from google drive if it doesn't exist
if not os.path.exists(data_path):
    gdown.download(
        "https://drive.google.com/uc?id=1eqSWu_BiK0KUJBwWM9L1D2An89gthjs_",
        data_path
    )

print('Loading data')
# Load the data from pickle
data = pd.read_pickle(data_path)
# convert the pandas dataframe to a numpy array
# Select only the image_embeddings and annot1 columns

# data = pd.read_pickle(data_path)
data['count_annotation'] = data.apply(lambda x: " ".join([str(x['answer']),x['animal']]), axis=1)
data = data[['image_embeddings', 'annot1','count_annotation']].to_numpy()

# use the first 100 images for training
train_data = data
print('Data Shape:', train_data.shape)

#Hyperparameters 
BATCH_SIZE = int(config['HYPERPARAMETERS']['batch_size'])
EPOCHS = int(config['HYPERPARAMETERS']['num_epochs'])
GENERATOR_LEARNING_RATE = float(config['HYPERPARAMETERS']['generator_learning_rate'])
DISCRIMINATOR_LEARNING_RATE = float(config['HYPERPARAMETERS']['discriminator_learning_rate'])

save_interval = int(config['TRAIN']['save_interval'])

# Pick text Embedding method
text_model_checkpoint = config['MODEL']['text_model_checkpoint']
text_embeddings_fn = data_transforms.custom_text_embeddings_fn if text_model_checkpoint else data_transforms.pytorch_text_embeddings_fn


# Create the dataset
train_dataset = CGAN_Dataset(
    data=train_data,
    augmentation_fn = None,
    preprocessing_fn=None,
    text_embeddings_fn=text_embeddings_fn,
    text_model_checkpoint=text_model_checkpoint
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

## SHOULD WE USE TRAIN/VAL/TEST SPLIT?


########################################################################################################################
#                                     DEFINE THE MODEL AND PARAMETERS                                                  #
########################################################################################################################



# Text model
text_model = TextModel(text_model_checkpoint).to(device)
text_embedding_dim = text_model.model.config.hidden_size

# Model Architecture parameters
noise_dim = int(config['MODEL']['noise_dim'])

print('Creating model')
# Model
generator = Generator(noise_dim=noise_dim,text_embedding_dim=text_embedding_dim).to(device)
generator.apply(utils.weights_init)
print('Number of parameters in generator:', sum(p.numel() for p in generator.parameters() if p.requires_grad))


discriminator = Discriminator(text_embedding_dim=text_embedding_dim).to(device)
discriminator.apply(utils.weights_init)
print('Number of parameters in discriminator:', sum(p.numel() for p in discriminator.parameters() if p.requires_grad))

# Define the optimizers
generator_optimizer = optim.Adam(generator.parameters(), lr=GENERATOR_LEARNING_RATE, betas=(0.5, 0.999))
# discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=DISCRIMINATOR_LEARNING_RATE, betas=(0.5, 0.999))
# Use momentum optimizer for discriminator
discriminator_optimizer = optim.SGD(discriminator.parameters(), lr=DISCRIMINATOR_LEARNING_RATE, momentum=0.5)

########################################################################################################################
#                                            Load latest Model and Best Model States                                   #
########################################################################################################################

CURRENT_EPOCH = 0

# Set best loss to infinity
BEST_GENERATOR_LOSS = float("inf")
BEST_DISCRIMINATOR_LOSS = float("inf")

generator_loss_list = []
discriminator_loss_list = []
discriminator_real_loss_list  = []
discriminator_fake_loss_list = []

# If we want to resume training
if config['TRAIN']['resume'] != 'no':
    print("Resuming training...")

    # Load the model with the highest epoch number from the Generator and Discriminator directories in sorted order    
    generator_file = torch.load(sorted(glob.glob(os.path.join('Experiments', exp_name, 'saved_models', 'Generator', '*.pth')))[-1])
    
    CURRENT_EPOCH = generator_file['epoch']
    
    generator.load_state_dict(generator_file['model_state_dict'])
    # generator_optimizer.load_state_dict(generator_file['optimizer_state_dict'])
    # Load generator loss list as a pickle file
    with open(os.path.join('Experiments', exp_name, 'saved_models', 'Generator', 'generator_loss_list.pkl'), 'rb') as f:
        generator_loss_list = pickle.load(f)
    
    generator.to(device)

    # discriminator_file = torch.load(sorted(glob.glob(os.path.join('Experiments', exp_name, 'Discriminator', '*.pth')))[-1])
    # discriminator.load_state_dict(discriminator_file['model_state_dict'])
    # discriminator_optimizer.load_state_dict(discriminator_file['optimizer_state_dict'])
    # discriminator.to(device)

    print("Loaded Generator and Discriminator from epoch", CURRENT_EPOCH)


    # Load best model if exists
    if os.path.isfile(os.path.join('Experiments', exp_name, 'best_generator.pth')):
        print("Loading best generator...")
        best_generator_file = torch.load(os.path.join('Experiments', exp_name, 'best_generator.pth'))
        BEST_GENERATOR_LOSS = best_generator_file['loss']

    # if os.path.isfile(os.path.join('Experiments', exp_name, 'best_discriminator.pth')):
    #     print("Loading best discriminator...")
    #     best_discriminator_file = torch.load(os.path.join('Experiments', exp_name, 'best_discriminator.pth'))
    #     BEST_DISCRIMINATOR_LOSS = best_discriminator_file['loss']

########################################################################################################################
#                                        Load the specific discriminator model if required                             #
########################################################################################################################
pick_discriminator = config['MODEL']['pick_discriminator']
if pick_discriminator != 'no':
    print("Loading discriminator model at epoch", pick_discriminator)
    print('!!!CAUTION: This will override the old discriminator model starting from epoch:', pick_discriminator)
    print('The best discriminator model still be retained until a new best discriminator model is found starting from this checkpoint')
    print('The generator model will continue to be trained from the last checkpoint, epoch:', CURRENT_EPOCH)

    # Check if the user wantes to continue or stop
    print("Do you want to continue training? (y/n)")
    choice = input()
    if choice != 'y':
        print("Exiting...")
        exit()


    discriminator_file = torch.load(os.path.join('Experiments', exp_name, 'Discriminator', '_epoch_' + str(pick_discriminator) + '.pth'))
    discriminator.load_state_dict(discriminator_file['model_state_dict'])
    discriminator_optimizer.load_state_dict(discriminator_file['optimizer_state_dict'])
    CURRENT_DISCRIMINATOR_LOSS = discriminator_file['loss']
    discriminator.to(device)
    print("Loaded specific discriminator model at epoch:", pick_discriminator)


########################################################################################################################
#                                            TRAINING LOOP                                                           #
########################################################################################################################



print("Starting training...")
# STart timer

# Setup Training Loop
for epoch in tqdm(range(CURRENT_EPOCH+1, EPOCHS)):

    epoch_generator_loss = 0
    epoch_discriminator_loss = 0
    epoch_discriminator_real_loss = 0
    epoch_discriminator_fake_loss = 0

    start_time = time.time()
    for index, input in tqdm(enumerate(train_loader)):

        image = input['image'].to(device)
        text = input['text']
        count = input['count']
        text_tokens = input['text_tokens'] if input['text_tokens'] is not None else None
        input_ids = text_tokens["input_ids"].to(device)  
        token_type_ids = text_tokens["token_type_ids"].to(device) 
        attention_mask = text_tokens["attention_mask"].to(device)

        with torch.no_grad():
            text_embeddings = text_model(input_ids, token_type_ids, attention_mask)

        text_embeddings = text_embeddings.to(device)


        discriminator_optimizer.zero_grad()

        real_target = Variable(torch.ones(image.size(0), 1))
        fake_target = Variable(torch.zeros(image.size(0), 1))

        # Apply Label Smoothing
        real_target = utils.smooth_positive_labels(real_target).to(device)
        fake_target = utils.smooth_negative_labels(fake_target).to(device)

        # Mix 10 of the data of real_target and fake_target
        # real_target = torch.cat((real_target, real_target, real_target, real_target, real_target, real_target, real_target, real_target, real_target, real_target), 0)

        # Generate a noise vector for fake image generation
        noise_vector = torch.randn(image.size(0), noise_dim, device=device).to(device)

       # Generate fake image
        generated_image = generator(noise_vector, text_embeddings)

        # Apply gaussian blur to the generated image
        generated_image = transforms.GaussianBlur(kernel_size=(3,3))(generated_image)

        # Get the discriminator loss for the fake image
        D_fake_loss = utils.discriminator_loss(
            discriminator(generated_image.detach(), text_embeddings), 
            fake_target
        )
        epoch_discriminator_fake_loss += D_fake_loss.item()

        # Get the discriminator loss for the real image
        D_real_loss = utils.discriminator_loss(
            discriminator(image, text_embeddings),
            real_target
        )
        epoch_discriminator_real_loss += D_real_loss.item()

        # Get the average discriminator loss and backpropagate
        D_total_loss = (D_real_loss + D_fake_loss) / 2
        epoch_discriminator_loss += D_total_loss.item()

        
        D_total_loss.backward()
        discriminator_optimizer.step()

        # Compute the generator loss and backpropagate it
        generator_optimizer.zero_grad()
        G_loss = utils.generator_loss(
            discriminator(generated_image,text_embeddings), 
            real_target
        )
        epoch_generator_loss += G_loss.item()

        # Get number of learning parameters in the generator
        # generator_parameters = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        # print("Generator parameters:", generator_parameters)

        print('Losses:', D_total_loss.item(), G_loss.item())

        G_loss.backward()
        generator_optimizer.step()

    # Get the average losses for the epoch
    epoch_generator_loss /= len(train_loader)
    epoch_discriminator_loss /= len(train_loader)
    epoch_discriminator_real_loss /= len(train_loader)
    epoch_discriminator_fake_loss /= len(train_loader)

    # Add the losses to the list
    generator_loss_list.append(epoch_generator_loss)
    discriminator_loss_list.append(epoch_discriminator_loss)
    discriminator_real_loss_list.append(epoch_discriminator_real_loss)
    discriminator_fake_loss_list.append(epoch_discriminator_fake_loss)



    ##################################################################################################################
    #                                            SAVE MODEL CHECKPOINTS AND BEST MODEL                               #
    ##################################################################################################################

    if epoch % save_interval == 0:

        # Save model checkpoints
        torch.save({
            'epoch': epoch,
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': generator_optimizer.state_dict(),
            'loss': epoch_generator_loss
        }, os.path.join('Experiments', exp_name, 'saved_models', 'Generator', 'Epoch_' + str(epoch) + '.pth'))
        torch.save({
            'epoch': epoch,
            'model_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': discriminator_optimizer.state_dict(),
            'loss': epoch_discriminator_loss
        }, os.path.join('Experiments', exp_name, 'saved_models', 'Discriminator', 'Epoch_' + str(epoch) + '.pth'))
        print("Saved model checkpoints at epoch", epoch)


        # Save fake and real images for visualization with text as caption
        utils.save_image(
            generated_image[0],
            image[0],
            text[0]+"__"+count[0],
            os.path.join('Experiments', exp_name, 'saved_images', 'Epoch_' + str(epoch)+ '_Batch_' + str(BATCH_SIZE))
        )
        print("Saved images")


        # Generate and save plots of the loss functions
        utils.save_loss_plot(
            generator_loss_list,
            discriminator_real_loss_list,
            discriminator_fake_loss_list,
            discriminator_loss_list,
            os.path.join('Experiments', exp_name, 'saved_plots', 'Loss.png')
        )
        print("Saved loss plots at " +os.path.join('Experiments', exp_name, 'saved_plots', 'Loss.png'))
        
        # Save best discriminator model
        if epoch_discriminator_loss < BEST_DISCRIMINATOR_LOSS:
            torch.save({
                'epoch': epoch,
                'model_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': discriminator_optimizer.state_dict(),
                'loss': epoch_discriminator_loss,
            }, os.path.join('Experiments', exp_name, 'best_discriminator.pth'))
            BEST_DISCRIMINATOR_LOSS = epoch_discriminator_loss
            print("Saved best discriminator model")
            
        # Save best generator model
        if epoch_generator_loss < BEST_GENERATOR_LOSS:
            torch.save({
                'epoch': epoch,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': generator_optimizer.state_dict(),
                'loss': epoch_generator_loss,
            }, os.path.join('Experiments', exp_name, 'best_generator.pth'))
            BEST_GENERATOR_LOSS = epoch_generator_loss
            print("Saved best generator model")

        # Save the losses list as a pickle file
        with open(os.path.join('Experiments', exp_name, 'saved_logs', 'discriminator_loss_list.pkl'), 'wb') as f:
            pickle.dump(discriminator_loss_list, f)
        with open(os.path.join('Experiments', exp_name, 'saved_logs', 'generator_loss_list.pkl'), 'wb') as f:
            pickle.dump(generator_loss_list, f)
        with open(os.path.join('Experiments', exp_name, 'saved_logs', 'discriminator_real_loss_list.pkl'), 'wb') as f:
            pickle.dump(discriminator_real_loss_list, f)
        with open(os.path.join('Experiments', exp_name, 'saved_logs', 'discriminator_fake_loss_list.pkl'), 'wb') as f:
            pickle.dump(discriminator_fake_loss_list, f)
        print("Saved loss lists")
        

        