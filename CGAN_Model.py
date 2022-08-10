import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self,noise_dim,text_embedding_dim):
        super(Generator, self).__init__()

        self.latent_layer = nn.Sequential(
            nn.Linear(noise_dim, 4*4*512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.linear_text_embedding = nn.Sequential(
            nn.Linear(text_embedding_dim, 4*4*512)
        )

        self.model =nn.Sequential(
            nn.ConvTranspose2d(1024  , 512 , 4 ,2,1,bias = False),
            nn.BatchNorm2d(64*8, momentum=0.1,  eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64*8, 64*4, 4, 2, 1,bias=False),
            nn.BatchNorm2d(64*4, momentum=0.1,  eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64*4, 64*2, 4, 2, 1,bias=False),
            nn.BatchNorm2d(64*2, momentum=0.1,  eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64*2, 64*1, 4, 2, 1,bias=False),
            nn.BatchNorm2d(64*1, momentum=0.1,  eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64*1, 64//2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64//2, momentum=0.1,  eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64//2, 3, 4, 2, 1, bias=False),
            
            nn.Tanh()
        )

    def forward(self,  noise_vector,text_embeddings):
        
        latent_output = self.latent_layer(noise_vector)
        latent_output = latent_output.view(-1, 512,4,4)

        text_embeddings_output = self.linear_text_embedding(text_embeddings)
        text_embeddings_output = text_embeddings_output.view(-1, 512, 4, 4)
        
        concat = torch.cat((latent_output, text_embeddings_output), dim=1)
        
        
        image = self.model(concat)
        return image


    
class Discriminator(nn.Module):
    def __init__(self,text_embedding_dim):
        super(Discriminator, self).__init__()

        self.linear_text_embedding = nn.Sequential(
            nn.Linear(text_embedding_dim, 1*256*256)
        )
        
        self.model = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1, bias=False),
            nn.Dropout(0.1),
            nn.BatchNorm2d(64, momentum=0.1,  eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64*2, 4, 3, 2, bias=False),
            nn.Dropout(0.1),
            nn.BatchNorm2d(64*2, momentum=0.1,  eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64*2, 64*4, 4, 3, 2, bias=False),
            nn.Dropout(0.1),
            nn.BatchNorm2d(64*4, momentum=0.1,  eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Dropout(0.2),
            
            nn.Linear(256 *15 * 15,1),
            nn.Sigmoid()
        )

    def forward(self, image, text_embeddings):

        text_embeddings = self.linear_text_embedding(text_embeddings)
        text_embeddings = text_embeddings.view(-1, 1, 256, 256)

        image = image.view(-1, 3, 256, 256)

        concat = torch.cat((image, text_embeddings), dim=1)

        output = self.model(concat)
        return output


# Custom Model Class to be used in the training process for obtaining text embeddings
from transformers import AutoModel
class TextModel(nn.Module):
    def __init__(self,checkpoint): 
        super(TextModel,self).__init__() 
        
        #Load Model with given checkpoint and extract its body
        self.model = AutoModel.from_pretrained(checkpoint)
    
    def forward(self, input_ids, token_type_ids, attention_mask):
        with torch.no_grad():
            last_hidden_state = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).last_hidden_state
        #Return Vector for [CLS] token
        return last_hidden_state[:,0,:].cpu() #outputs a tensor of shape (batch_size, hidden_size) for the first token in the sequence