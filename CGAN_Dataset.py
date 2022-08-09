from torch.utils.data import Dataset
import torch
import numpy as np



class CGAN_Dataset(Dataset):
    '''
        Read Text and Image
        Perform necessary augmentation and preprocessin on the image.
        Apply relevant custom text embeddings on the text
    '''

    def __init__(
        self,
        data,
        augmentation_fn=None, 
        preprocessing_fn=None,
        text_embeddings_fn=None,
        text_model_checkpoint=None
    ):

        self.data = data
        self.augmentation_fn = augmentation_fn
        self.preprocessing_fn = preprocessing_fn
        self.text_embeddings_fn = text_embeddings_fn
        self.text_model_checkpoint = text_model_checkpoint

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        # Get the image and text
        # image = self.data.iloc[idx]['image_embeddings']
        # text = self.data.iloc[idx]['annot1']

        image = self.data[idx,0].squeeze()
        text1 = self.data[idx,1]
        text2 = self.data[idx,2]


        # Apply augmentation
        if self.augmentation_fn:
            image = self.augmentation_fn(image)

        # Apply preprocessing
        if self.preprocessing_fn:
            image = self.preprocessing_fn(image)

        # Apply custom embeddings
        if self.text_embeddings_fn:
            text_tokens = self.text_embeddings_fn(text1,text2, self.text_model_checkpoint)

        return {
            'image': image,
            'text': text1,
            'count': text2,
            'text_tokens': text_tokens
        }


        