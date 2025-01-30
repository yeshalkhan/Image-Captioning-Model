import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        
        # get the pretrained resnet model
        resnet = models.resnet101(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # define an embedding layer
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features # -> embedded image feature vector
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super().__init__()
        
        # define an embedding layer
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        # difine the LSTM
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # define a dropout layer
        self.dropout = nn.Dropout(p=0.5)
        
        # define the final, fully-connected output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # initialize the weights
        self.initialize_weights()
        
    def forward(self, features, captions):
        
        captions = captions[:, :-1]
        # create embedded word vectors for each word in caption
        embed = self.word_embeddings(captions)
        # concatenate the image feature and caption embeds
        embed = torch.cat((features.unsqueeze(1), embed), dim=1) 
        # get prediction (output) and the hidden states
        output, (hidden, cell) = self.lstm(embed)
        output = self.dropout(output)
        output = self.fc(output)
     
        return output

    def initialize_weights(self):
#         nn.init.xavier_normal_(self.fc.weight)
        
        # set FC bias to a small constant 
        self.fc.bias.data.fill_(0.01)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-1, 1)
            
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)
        
    def sample(self, inputs, states=None, max_len=20):
        '''Accepts pre-processed image tensor (inputs) and returns predicted sentence (list of ids of length max_len)'''  
        unique_index = None
        predicted_sentence = []
        for i in range(max_len):
            
            output_lstm, states = self.lstm(inputs, states)
            
            output = self.fc(output_lstm)
            
            _, tensor_index = torch.max(output, 2)
            
            # get numerical value from a tensor 
            unique_index = tensor_index.item()
            
            # append unique index for each word to list
            predicted_sentence.append(unique_index)
            
            if unique_index == 1:
                break
            
            inputs = self.word_embeddings(tensor_index)
              
        return predicted_sentence