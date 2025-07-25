# Image Captioning Model

This repository contains an Image Captioning model built using TensorFlow. The model uses a combination of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) to generate descriptive captions for images. It has been trained on the Flickr8K dataset and evaluated using the SacreBLEU metric.

## Features
- **CNN for Feature Extraction:** The model uses a pre-trained Convolutional Neural Network (Xception) to extract visual features from images.
- **RNN for Caption Generation:** A Recurrent Neural Network (LSTM) processes the extracted features and generates captions.
- **Tokenizer and Embeddings:** The textual data is tokenized and embedded for effective processing by the RNN.
- **BLEU Score Evaluation:** The generated captions are evaluated using the SacreBLEU metric.

## Dataset
The model is trained on the [(**Flickr8K Dataset**)](https://github.com/awsaf49/flickr-dataset), which contains 8,000 images with five captions each. The dataset is preprocessed by tokenizing the captions and encoding them for training the sequence model.

## Model Architecture
1. **Feature Extraction:**
   - A CNN (Xception) extracts feature vectors from input images.
2. **Text Processing:**
   - Tokenization and word embeddings are applied to the captions.
   - The sequences are padded to maintain uniformity.
3. **Caption Generation:**
   - The extracted image features and text sequences are passed through an LSTM-based RNN.
   - The final output is a sequence of words forming a caption.

## Evaluation
The model is evaluated using the **SacreBLEU** metric, which measures the similarity between generated and reference captions.


