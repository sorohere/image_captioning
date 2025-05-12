import cv2
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
import numpy as np
from PIL import Image
from .utils import image_transformation, set_cuda
from torchvision.models import ResNet101_Weights

class Vocabulary:
    """
    Vocabulary class to handle tokenization and numericalization of captions.
    
    Args:
        caption_list (list): List of captions as strings.
        freq_threshold (int): Minimum frequency for a word to be added to the vocabulary.
    """
    def __init__(self, caption_list, freq_threshold):
        self.caption_list = caption_list
        self.threshold = freq_threshold
        # Adding special tokens
        self.idx2wrd = {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unknown>'}
        self.wrd2idx = {word: idx for idx, word in self.idx2wrd.items()}
        self.create_vocab()

    def create_vocab(self):
        """Creates a vocabulary based on the frequency threshold."""
        all_tokens = [word for caption in self.caption_list for word in caption.split()]
        word_counts = Counter(all_tokens)
        index = len(self.idx2wrd)  # Start adding to the dict after the special tokens index
        for word, count in word_counts.items():
            if count >= self.threshold and word not in self.wrd2idx:
                self.wrd2idx[word] = index
                self.idx2wrd[index] = word
                index += 1

    def cap2tensor(self, caption):
        """
        Converts a caption to a tensor of token IDs.

        Args:
            caption (str): Input caption as a string.

        Returns:
            torch.Tensor: Numericalized caption as a tensor of token IDs.
        """
        numericalized_caption = [self.wrd2idx['<start>']]
        for word in caption.split():
            if word in self.wrd2idx:
                numericalized_caption.append(self.wrd2idx[word])
            else:
                numericalized_caption.append(self.wrd2idx['<unknown>'])
        numericalized_caption.append(self.wrd2idx['<end>'])
        return torch.tensor(numericalized_caption)

    def __len__(self):
        return len(self.wrd2idx)

class Image_encoder(nn.Module):
    """
    Image encoder using ResNet101 for feature extraction.

    Args:
        device (torch.device): Device to load the model on.
    """
    def __init__(self, device):
        super(Image_encoder, self).__init__()
        self.device = device
        self.resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        for param in self.resnet.parameters():
            param.requires_grad_(False)
        self.layers_list = list(self.resnet.children())[:-2]  # Remove the last classification layer and its FC
        self.Resnet = nn.Sequential(*self.layers_list)

    def forward(self, image_tensor):
        """
        Forward pass to extract image features.

        Args:
            image_tensor (torch.Tensor): Batch of images.

        Returns:
            torch.Tensor: Extracted image features of shape (batch_size, 49, 2048).
        """
        features = self.Resnet(image_tensor)  # Shape: (batch_size, 2048, 7, 7)
        features = features.permute(0, 2, 3, 1)  # Shape: (batch_size, 7, 7, 2048)
        features = features.view(features.size(0), -1, features.size(-1))  # Flatten to a single tensor
        return features

class AttentionLayer(nn.Module):
    """
    Attention layer for soft attention mechanism.

    Args:
        features_dims (int): Dimensions of image features.
        hidden_state_dims (int): Dimensions of the hidden state.
        attention_dims (int): Dimensions of the attention space.
    """
    def __init__(self, features_dims, hidden_state_dims, attention_dims):
        super().__init__()
        self.U = nn.Linear(features_dims, attention_dims)
        self.W = nn.Linear(hidden_state_dims, attention_dims)
        self.A = nn.Linear(attention_dims, 1)

    def forward(self, img_features, hidden_state):
        """
        Forward pass for the attention mechanism.

        Args:
            img_features (torch.Tensor): Image features of shape (batch_size, 49, features_dims).
            hidden_state (torch.Tensor): Decoder hidden state of shape (batch_size, hidden_state_dims).

        Returns:
            torch.Tensor: Attention weights (batch_size, 49).
            torch.Tensor: Context vector (batch_size, features_dims).
        """
        u_hs = self.U(img_features)  # Shape: (batch_size, 49, attention_dims)
        w_hs = self.W(hidden_state)  # Shape: (batch_size, attention_dims)
        combined_states = torch.tanh(u_hs + w_hs.unsqueeze(1))
        attention_scores = self.A(combined_states).squeeze(2)  # Shape: (batch_size, 49)
        alpha = F.softmax(attention_scores, dim=1)
        context_vector = (img_features * alpha.unsqueeze(2)).sum(dim=1)
        return alpha, context_vector

class Attention_Based_Decoder(nn.Module):
    """
    Decoder with attention mechanism.

    Args:
        features_dims (int): Dimensions of image features.
        hidden_state_dims (int): Dimensions of the hidden state.
        attention_dims (int): Dimensions of the attention space.
        word_emb_dims (int): Dimensions of the word embeddings.
        vocab_size (int): Size of the vocabulary.
        drop_prob (float): Dropout probability.
        device (torch.device): Device to load the model on.
    """
    def __init__(self, features_dims, hidden_state_dims, attention_dims, word_emb_dims, vocab_size, drop_prob, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.device = device
        self.dropout_layer = nn.Dropout(drop_prob)
        self.attention_layer = AttentionLayer(features_dims, hidden_state_dims, attention_dims)
        self.tokens_embedding = nn.Embedding(vocab_size, word_emb_dims)
        self.hidden_state_init = nn.Linear(features_dims, hidden_state_dims)
        self.cell_state_init = nn.Linear(features_dims, hidden_state_dims)
        self.lstm_cell = nn.LSTMCell(word_emb_dims + features_dims, hidden_state_dims, bias=True)
        self.fcl = nn.Linear(hidden_state_dims, vocab_size)

    def init_hidden_state(self, image_features_tensor):
        """
        Initializes the hidden and cell states for the LSTM.

        Args:
            image_features_tensor (torch.Tensor): Image features of shape (batch_size, 49, features_dims).

        Returns:
            torch.Tensor: Initialized hidden state.
            torch.Tensor: Initialized cell state.
        """
        features_mean = image_features_tensor.mean(dim=1)
        h = self.hidden_state_init(features_mean)
        c = self.cell_state_init(features_mean)
        return h, c

    def forward(self, batch_images_features, batch_captions_tensors):
        """
        Forward pass of the decoder.

        Args:
            batch_images_features (torch.Tensor): Image features of shape (batch_size, 49, features_dims).
            batch_captions_tensors (torch.Tensor): Tokenized captions as tensors.

        Returns:
            torch.Tensor: Predicted token probabilities.
            torch.Tensor: Attention weights.
        """
        captions_len = len(batch_captions_tensors[0]) - 1
        batch_size = batch_captions_tensors.size(0)
        features_size = batch_images_features.size(1)
        embedded_tokens = self.tokens_embedding(batch_captions_tensors)
        hidden_state, cell_state = self.init_hidden_state(batch_images_features)
        preds = torch.zeros(batch_size, captions_len, self.vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, captions_len, features_size).to(self.device)

        for wrd_index in range(captions_len):
            alpha, context_vector = self.attention_layer(batch_images_features, hidden_state)
            current_token_emb = embedded_tokens[:, wrd_index]
            lstm_input = torch.cat((current_token_emb, context_vector), dim=1)
            hidden_state, cell_state = self.lstm_cell(lstm_input, (hidden_state, cell_state))
            tokens_probs = self.fcl(self.dropout_layer(hidden_state))
            preds[:, wrd_index] = tokens_probs
            alphas[:, wrd_index] = alpha
        return preds, alphas

class Encoder_Decoder_Model(nn.Module):
    """
    Encoder-Decoder model combining image encoder and attention-based decoder.

    Args:
        features_dims (int): Dimensions of image features.
        hidden_state_dims (int): Dimensions of the hidden state.
        attention_dims (int): Dimensions of the attention space.
        word_emb_dims (int): Dimensions of the word embeddings.
        vocab_size (int): Size of the vocabulary.
        drop_prob (float): Dropout probability.
        device (torch.device): Device to load the model on.
    """
    def __init__(self, features_dims, hidden_state_dims, attention_dims, word_emb_dims, vocab_size, drop_prob, device):
        super().__init__()
        self.device = device
        self.img_encoder = Image_encoder(device)
        self.decoder = Attention_Based_Decoder(
            features_dims, 
            hidden_state_dims, 
            attention_dims, 
            word_emb_dims, 
            vocab_size, 
            drop_prob,
            device
        )

    def forward(self, batch_images, batch_tokenized_captions):
        """
        Forward pass through the encoder-decoder model.

        Args:
            batch_images (torch.Tensor): Batch of images.
            batch_tokenized_captions (torch.Tensor): Tokenized captions as tensors.

        Returns:
            torch.Tensor: Predicted token probabilities.
            torch.Tensor: Attention weights.
        """
        image_features = self.img_encoder(batch_images)
        probs, alphas = self.decoder(image_features, batch_tokenized_captions)
        return probs, alphas

    def predict(self, image, vocab, max_cap_len=20, debugging=False):
        """
        Generate a caption for a given image.

        Args:
            image (np.array): Input image in numpy array format.
            vocab (Vocabulary): Vocabulary object for decoding.
            max_cap_len (int): Maximum caption length.
            debugging (bool): If True, prints debugging information.

        Returns:
            list: Attention weights.
            list: Generated caption as a list of tokens.
        """
        self.eval()
        with torch.no_grad():
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = image_transformation(img).unsqueeze(0).to(self.device)
            image_features = self.img_encoder(img)
            if debugging:
                print(f"predict input image_Shape:{image_features.shape}")
            hidden_state, cell_state = self.decoder.init_hidden_state(image_features)
            caption = [vocab.idx2wrd[1]]
            token = torch.tensor(vocab.wrd2idx["<start>"]).unsqueeze(0).to(self.device)
            attentions = []

            for i in range(max_cap_len):
                alpha, context_vector = self.decoder.attention_layer(image_features, hidden_state)
                if debugging:
                    print(i, "-attention map for token:", vocab.idx2wrd[token.item()], "is", alpha.shape)
                attentions.append(alpha.cpu().detach().numpy())
                current_token_emb = self.decoder.tokens_embedding(token)
                lstm_input = torch.cat((current_token_emb.squeeze(1), context_vector), dim=1)
                hidden_state, cell_state = self.decoder.lstm_cell(lstm_input, (hidden_state, cell_state))
                tokens_prob = self.decoder.fcl(hidden_state)
                next_token = tokens_prob.argmax(dim=1).item()
                next_word = vocab.idx2wrd[next_token]
                caption.append(next_word)
                if next_word == "<end>":
                    break
                token = torch.tensor([next_token]).unsqueeze(0).to(self.device)

            if debugging:
                print("attention shape:", np.array(attentions).shape)
                print("caption_length:", len(caption))
            return attentions, caption