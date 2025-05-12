import torch
from construct.architecture import Encoder_Decoder_Model, Vocabulary
from construct.dataloader import FlickrDataset
from construct.trainer import ModelTrainer
from construct.utils import set_cuda
import json
import re
from sklearn.model_selection import train_test_split

def main():
    """
    Main function to train an image captioning model using a Flickr30k dataset.
    
    Steps:
        1. Load configuration parameters from a JSON file.
        2. Set up the computation device (CPU or GPU).
        3. Read and process the captions file.
        4. Create vocabulary from captions with a frequency threshold.
        5. Split the dataset into training and validation sets.
        6. Initialize and train the Encoder-Decoder model.
        7. Save the weights and vocabulary after training.
    """
    # Load configuration
    with open("scripts/config.json", "r") as f:
        config = json.load(f)

    # Set device for computation
    device = set_cuda()

    # Extract configuration parameters
    FEATURES_DIMS = config["FEATURES_DIMS"]
    HIDDEN_STATE_DIMS = config["HIDDEN_STATE_DIMS"]
    ATTENTION_DIMS = config["ATTENTION_DIMS"]
    WORD_EMB_DIMS = config["WORD_EMB_DIMS"]
    VOCAB_FREQ_THRESHOLD = config["VOCAB_FREQ_THRESHOLD"]
    DROP_PROB = config["DROP_PROB"]
    BATCH_SIZE = config["BATCH_SIZE"]
    LEARNING_RATE = config["LEARNING_RATE"]
    NUM_EPOCHS = config["NUM_EPOCHS"]
    IMAGE_DIR = config["IMAGE_DIR"]
    CAPTIONS_FILE = config["CAPTIONS_FILE"]
    VOCAB_SAVE_PATH = config["VOCAB_SAVE_PATH"]
    MODEL_SAVE_PATH = config["MODEL_SAVE_PATH"]

    # Initialize lists to store image names and captions
    image_names = []
    captions_list = []

    # Regular expression pattern to match image name and caption pairs
    pattern = r'([a-zA-Z0-9]+\.[a-zA-Z]+),\s*(.*)'

    # Read and process captions file
    with open(CAPTIONS_FILE, 'r') as text:
        for line in text:
            match = re.match(pattern, line.strip())
            if match:
                img_name = match.group(1)
                caption = match.group(2)
                image_names.append(img_name)
                captions_list.append(caption)

    # Create vocabulary
    vocab = Vocabulary(captions_list, VOCAB_FREQ_THRESHOLD)
    VOCAB_SIZE = len(vocab)

    # Split data into training and validation sets
    train_indices, val_indices = train_test_split(
        range(len(image_names)), 
        test_size=0.2, 
        random_state=42
    )

    train_dataset = FlickrDataset(
        IMAGE_DIR,
        [image_names[i] for i in train_indices],
        [captions_list[i] for i in train_indices],
        vocab
    )

    val_dataset = FlickrDataset(
        IMAGE_DIR,
        [image_names[i] for i in val_indices],
        [captions_list[i] for i in val_indices],
        vocab
    )

    # Initialize the Encoder-Decoder model
    model = Encoder_Decoder_Model(
        FEATURES_DIMS,
        HIDDEN_STATE_DIMS,
        ATTENTION_DIMS,
        WORD_EMB_DIMS,
        VOCAB_SIZE,
        DROP_PROB,
        device
    )

    # Initialize the training handler
    trainer = ModelTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )

    # Train the model
    trainer.train(num_epochs=NUM_EPOCHS)

    # Save the vocabulary for later use
    torch.save(vocab, VOCAB_SAVE_PATH)
    torch.save(model, MODEL_SAVE_PATH)
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
