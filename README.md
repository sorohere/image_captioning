# capage

A CNN-LSTM-based image captioning model trained on the Flickr30k dataset. To learn more about how this project works, check out the [documentation](https://drive.google.com/file/d/1H-p_hxG4sENpcs0h-nOQMAWnKsHj9D2s/view?usp=sharing).

https://github.com/user-attachments/assets/1a59807a-c5a4-48cb-baf5-faf1ec225e1e

### Installation

1. clone the repository: `https://github.com/sorohere/capage.git`
2. download weights and dataset:
   - run `run.sh` script, or
   - download manually from [flickr30k](https://github.com/sorohere/flickr-dataset?tab=readme-ov-file#flickr30k)

3. install dependencies: `pip install -r requirement.txt`

   
### Dataset  

the dataset used for this project is **Flickr30k**, which consists of:  
- around **31,000 unique images**.  
- each image is paired with **5 captions**, resulting in approximately **151,000 image-caption pairs**.  

#### Dataset Structure: 
i. **images folder**: contains all the images used for training and evaluation.  
ii. **captions file**:
   - `captions.txt`: A text file mapping each image to its corresponding caption.  
   - each line follows the format: `image_name, caption`
   
#### Training:
To start training the model: `python scripts/train.py`, the vocabulary and trained model will be saved in ```scripts/checkpoints/```.

#### Inference:

Ensure you have a trained model and vocabulary saved. If not, train the model yourself(checkout the scripts), to generate captions for images: `python inference.py`
