import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from .utils import image_transformation, standarize_text
from torch.nn.utils.rnn import pad_sequence

class FlickrDataset(Dataset):
    """Dataset class for loading Flickr images and captions"""
    
    def __init__(self, image_dir, image_names, captions, vocab):
        """
        Args:
            image_dir (str): Directory containing images
            image_names (list): List of image filenames
            captions (list): List of corresponding captions
            vocab (Vocabulary): Vocabulary object for tokenization
        """
        self.image_dir = image_dir
        self.vocab = vocab
        self.image_names = image_names
        self.captions = captions
        
        assert len(self.image_names) == len(self.captions), "Number of images and captions must match"

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        caption = self.captions[idx]
        
        # Load and transform image
        image_path = os.path.join(self.image_dir, img_name)
        image = Image.open(image_path).convert('RGB')
        image = image_transformation(image)
        
        # Convert caption to tensor
        caption = standarize_text(caption)
        caption_tensor = self.vocab.cap2tensor(caption)
        
        return image, caption_tensor

def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences
    Args:
        batch: List of tuples (image, caption)
    """
    # Separate images and captions
    images, captions = zip(*batch)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Pad captions to same length
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    
    return images, captions
