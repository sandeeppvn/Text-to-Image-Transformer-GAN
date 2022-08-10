
########################################################################################################################
#                                            DATA TRANFORM HELPERS                                                     #
########################################################################################################################
from torchvision import transforms
def augmentation_fn(image):
    # Apply Gauusian blur to the image
    return transforms.GaussianBlur(kernel_size=(3,3))(image)
from torchvision.transforms import Normalize
def preprocessing_fn(image):
    # Normalize the image
    result = Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )(image)

    return result

from transformers import AutoTokenizer
def custom_text_embeddings_fn(text1, text2, text_model_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(text_model_checkpoint)
    encoding = tokenizer(text1,text2, truncation = True,
        # pad_to_max_length = True,
        padding = 'max_length',
        return_attention_mask = True,
        max_length = 50,
        return_tensors = 'pt'
    )

    return {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'token_type_ids': encoding['token_type_ids'].flatten()
    }

from torch.nn import Embedding
import torch
def pytorch_text_embeddings_fn(text):
    VOCAB_SIZE = 30522
    EMBEDDING_DIM = 384

    # Vectorize the text
    text_vectorized = torch.LongTensor(text)
    # Create the embedding layer
    embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM)
    # Embed the text
    text_embeddings = embedding(text_vectorized)
    return text_embeddings

