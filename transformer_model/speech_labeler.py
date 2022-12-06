# Helpful tutorial
# at https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py

from typing import Dict
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from helper_functions import make_onehot_vectors, make_label_vectors, split_sentences


class SpeechLabeler(torch.nn.Module):
    def __init__(self,
                 word_vocabulary: Dict[str, int],
                 embedding_size: int,
                 d_hid: int,
                 number_of_encoder_layers: int,
                 attention_head_size: int,
                 device: str,
                 character_level: bool,
                 dropout: float):
        
        super(SpeechLabeler, self).__init__()

        self.model_type = 'Transformer'

        # remember device, vocabulary and character_level
        self.device = device
        self.word_vocabulary = word_vocabulary
        self.character_level = character_level
        self.embedding_size = embedding_size

        # Encoder
        self.encoder = nn.Embedding(len(word_vocabulary), embedding_size)

        # Encode the positional information
        self.pos_encoder = PositionalEncoding(embedding_size, dropout)
        encoder_layers = TransformerEncoderLayer(embedding_size, attention_head_size, d_hid, dropout)

        # Transformer
        self.transformer_encoder = TransformerEncoder(encoder_layers, number_of_encoder_layers)

        # Decoder
        self.decoder = nn.Linear(embedding_size, 1)

        # Activation function
        self.sigmoid = torch.nn.Sigmoid()

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, batch_sentences):
        # Preprocessing
        batch_split_sentences = split_sentences(batch_sentences, self.character_level)

        # Create Vector
        vector_sentences = make_onehot_vectors(batch_split_sentences, self.word_vocabulary).to(self.device)

        # Make embeddings
        embedded_sentences = self.encoder(vector_sentences) * math.sqrt(self.embedding_size)

        # Add positional features
        embedded_sentences = self.pos_encoder(embedded_sentences)

        # Go through transformers
        output = self.transformer_encoder(embedded_sentences)
        output = self.decoder(output)

        return output

    def make_prediction(self, prediction):
        # Create linear layer based on length
        linear = torch.nn.Linear(prediction.size(1), 1)

        # Send through linear layer
        batch_label_space = linear(prediction.squeeze()).flatten()

        # Send through activation
        prediction = self.sigmoid(batch_label_space)

        return prediction
    
    def compute_loss(self, prediction, batch_tags):
        prediction = self.make_prediction(prediction)

        # Make Classification Vector
        vector_tags = make_label_vectors(batch_tags).to(self.device)

        # compute the loss
        loss_function = torch.nn.CrossEntropyLoss()
        loss = loss_function(prediction, vector_tags.to(torch.float32))

        with torch.no_grad():
            true_positive_prediction = 0
            true_negative_prediction = 0
            false_positive_prediction = 0
            false_false_prediction = 0

            for sentence_index, sentence in enumerate(prediction):  # Batch

                # 0: False, 1: True
                # Classification is Positive
                if vector_tags[sentence_index].item() == 1:
                    # Classification: +, Result: +
                    if torch.round(sentence).item() == 1:
                        true_positive_prediction += 1
                    # Classification: +, Result: -
                    else:
                        false_false_prediction += 1
                # Classification is Negative
                else:
                    # Classification: -, Result: -
                    if torch.round(sentence).item() == 0:
                        true_negative_prediction += 1
                    # Classification: -, Result: +
                    else:
                        false_positive_prediction += 1

        # Put together
        accuracy_data = [true_positive_prediction, true_negative_prediction, false_positive_prediction, false_false_prediction]

        return loss, accuracy_data

    def evaluate(self, test_data):
        self.eval()

        # evaluate the model
        true_predictions: int = 0
        false_predictions: int = 0
        aggregate_loss_sum = 0

        with torch.no_grad():

            # Todo: Decide, either batch test or all at once
            # go through all test data points
            all_senteces = []
            all_tags = []
            for sentence in test_data:
                all_senteces.append(sentence[0])
                all_tags.append(sentence[1])

            # send the data point through the model and get a prediction
            output = self.forward(all_senteces)

            # Calculate loss
            loss, accuracy_data = self.compute_loss(output, all_tags)
            aggregate_loss_sum += loss.item()

            # Add to accuracy variables
            true_predictions += accuracy_data[0] + accuracy_data[1]
            false_predictions += accuracy_data[2] + accuracy_data[3]

        accuracy = true_predictions / (true_predictions + false_predictions)
        f1_score = (2 * accuracy_data[0] / (2 * accuracy_data[0] + accuracy_data[2] + accuracy_data[3]))
        aggregate_loss = aggregate_loss_sum / len(test_data)

        self.train()
        return accuracy, f1_score, accuracy_data, math.exp(aggregate_loss), aggregate_loss

    def classify(self, string):
        # send the data point through the model and get a prediction
        self.eval()
        _, hidden = self.forward([string])
        prediction_torch = self.make_prediction(hidden)
        prediction_int = torch.round(prediction_torch).item()
        prediction = False if prediction_int == 0 else True

        return prediction, torch.round(prediction_torch, decimals=4).item()


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        x = x.permute(1, 0, 2)
        return self.dropout(x)