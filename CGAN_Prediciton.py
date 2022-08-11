import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


TEXT = 'Image of 2 trees'


# Generate the quantitaive text
# from CGAN_utils import process_prompt
# from transformers import pipeline
from transformers import AutoTokenizer
# from torch import topk
# quantitative_text = process_prompt(TEXT)
quantitative_text = '2 trees'
# qa = pipeline("question-answering", model='deepset/roberta-base-squad2', tokenizer='deepset/roberta-base-squad2')

# Load Config file
import configparser

config_path = 'CGAN_Config.ini'
config = configparser.ConfigParser()
config.read(config_path)


text_model_checkpoint = config['MODEL']['text_model_checkpoint']
# checkpoint = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(text_model_checkpoint)
from transformers import AutoModel
model = AutoModel.from_pretrained(text_model_checkpoint).to(device)

def extract_hidden_states(batch):
    # Place model inputs on the GPU
    inputs = {k:v.to(device) for k,v in batch.items() if k in tokenizer.model_input_names and isinstance(v, list)==False}
    # Extract Last Hidden State 
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    #Return Vector for [CLS] token
    return {'hidden_state': last_hidden_state[:,0]}

def get_embedding(sentence1,sentence2=None):
    if sentence2:
        tokens = tokenizer(sentence1,sentence2 ,return_tensors='pt')
    else:
        tokens = tokenizer(sentence1,return_tensors='pt')
    hidden_state = extract_hidden_states(tokens)
    return hidden_state

hidden_state_te = get_embedding(TEXT,quantitative_text)
text_embeddings=hidden_state_te['hidden_state']


from CGAN_Model import Generator


noise_dim = int(config['MODEL']['noise_dim'])
noise_vector = torch.randn(1, noise_dim, device=device).to(device)
text_embedding_dim = 384

# Download the best_generator using gdown if not present
import gdown
import os
if not os.path.exists('best_generator.pth'):
    gdown.download(
        "https://drive.google.com/uc?id=1VLPuqbuuTCXPayPm_MG6JbxJ9gTqTG4R", 
        output='best_generator.pth'
    )

generator = Generator(noise_dim=noise_dim,text_embedding_dim=text_embedding_dim).to(device)
generator.load_state_dict(torch.load('best_generator.pth')['model_state_dict'])
generator.eval()
generated_image = generator(noise_vector, text_embeddings)

from CGAN_utils import im_convert
import matplotlib.pyplot as plt
generated_image = im_convert(generated_image)
plt.imshow(generated_image)
plt.show()

# Save the image
plt.imsave('generated_image.png', generated_image)


