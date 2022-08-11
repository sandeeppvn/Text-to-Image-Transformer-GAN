import pandas as pd
import numpy as np
import os
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import gdown
import pandas as pd
import numpy as np 
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.graph_objects as go
import plotly.express as px


print('Downloading data')
# Download the TallyQA data from google drive if it doesn't exist
data_path = './animal_curated_dataset.csv'


if not os.path.exists(data_path):
    gdown.download(
        "https://drive.google.com/uc?id=1BY8RRtYDeTUWIr-udImISYY1YieJUJcs",
        data_path
    )

tqdm.pandas()
data = pd.read_csv(data_path)
data =data[(data['answer']>=1) & (data['answer']<=9) & (data['animal'] != 'broccolis')]

print(f' Distribution of Answer: \n{data["answer"].value_counts()}')
ploting_distribution = data['answer'].value_counts()
fig = px.bar(ploting_distribution, x=ploting_distribution.index, y=ploting_distribution.values, labels={'x':'Number of Animals in a Image', 'y':'Count'})
fig.update_layout(title_text="Distribution of Count of Animals in a Picture", xaxis_title="Number of Animals in a Picture", yaxis_title="Count", font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
fig.show()


print(f'Distribution of Animal Count: \n{data["animal"].value_counts()}')
ploting_count = data['animal'].value_counts()
fig = px.bar(ploting_count, x=ploting_count.index, y=ploting_count.values)
fig.update_layout(title_text="Distribution of Animals in the Dataset", xaxis_title="Animal", yaxis_title="Count", bargap=0.1, font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
fig.show()




##### THIS PART OF THE CODE WAS USED TO COPY IMAGE FILES TO CLOUD
image_list = data['image'].apply(lambda x: x.split('/')[-1]).to_list()
destination = '/Volumes/GoogleDrive/Shared drives/MSML612 DeepLearningProject/data/animal_images'

print(f'Loading {len(image_list)} images')

def load_image(img_path, max_size=600, shape=None):
  full_file_name = os.path.join(destination, img_path)
  image = Image.open(full_file_name).convert('RGB')
  #print(image.size)
  if max(image.size) > max_size:
    size = max_size
  else:
    size = max(image.size)

  in_transform= transforms.Compose([
       transforms.Resize(size=(256,256)),
       transforms.ToTensor()])
  image = in_transform(image).unsqueeze(0)

  #print((image.size()))
  return image

print(f'Loading images from {destination}')
data['image_embeddings']=data['image'].progress_apply(lambda x: load_image(x.split('/')[-1], max_size=600, shape=None))
print('done')
print(f' Data type of image embeddings: \n{data["image_embeddings"].iloc[0].dtype}')
print(f' Shape of image embeddings: {data["image_embeddings"].shape}')
print(f'Saving the file to the cloud')

## This command saves the dataframe to the cloud which will be used in the next part of the code
#data.to_pickle('/Volumes/GoogleDrive/Shared drives/MSML612 DeepLearningProject/data/animal_text_curated_embeddings_updated_20220810.pkl')