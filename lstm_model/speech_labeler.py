# Helpful tutorial
# at https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py

from typing import Dict
import math
import torch
import torch.nn.functional as F
from helper_functions import make_onehot_vectors, make_label_vectors


class SpeechLabeler(torch.nn.Module):
    def __init__(self,
                 word_vocabulary: Dict[str, int],
                 rnn_hidden_size: int,
                 embedding_size: int,
                 device: str = 'cpu'):
        
        super(SpeechLabeler, self).__init__()
        
        # remember device, vocabulary
        self.device = device
        self.word_vocabulary = word_vocabulary

        # Initialize encoder with an embedding layer and an LSTM
        self.word_embedding = torch.nn.Embedding(len(word_vocabulary), embedding_size)
        self.lstm = torch.nn.LSTM(embedding_size,
                                  rnn_hidden_size,
                                  batch_first=True,
                                  num_layers=1,
                                  bidirectional=True
                                  )
        
        # Hidden2tag linear layer takes the LSTM output and projects to tag space
        self.linear = torch.nn.Linear(rnn_hidden_size * 2, 1)

    def forward(self, batch_sentences, hidden=None):
        # Create Vector
        vector_sentences = make_onehot_vectors(batch_sentences, self.word_vocabulary).to(self.device)

        # Make embeddings
        embedded_sentences = self.word_embedding(vector_sentences)

        # Send through LSTM (Initial hidden is defaulted to 0)
        lstm_out, hidden = self.lstm(embedded_sentences, hidden)

        return lstm_out, hidden

    def make_prediction(self, hidden):
        # Concatenate the hidden state
        hidden = hidden[0]
        batch_size = hidden.size(1)
        hidden_concatenated = hidden.transpose(1, 0).contiguous().view(batch_size, -1)

        # Send through
        batch_label_space = self.linear(hidden_concatenated)
        print(hidden_concatenated.shape)

        # Send through activation
        # prediction = torch.nn.Sigmoid(batch_label_space)
        prediction = F.log_softmax(batch_label_space, dim=1)

        return prediction
    
    def compute_loss(self, hidden, batch_tags):
        # Make prediction
        prediction = self.make_prediction(hidden)

        # Make Classification Vector
        vector_tags = make_label_vectors(batch_tags).to(self.device)

        # compute the loss
        loss = F.nll_loss(prediction, vector_tags)

        with torch.no_grad():
            for sentence_index, sentence in enumerate(prediction):  # Batch
                true_positive_prediction = 0
                true_negative_prediction = 0
                false_positive_prediction = 0
                false_false_prediction = 0

                # 0: False, 1: True
                # Classification is Positive
                if vector_tags[sentence_index].item() == 1:
                    if torch.round(sentence).item() == 1:
                        true_positive_prediction += 1
                    else:
                        false_false_prediction += 1
                # Classification is Negative
                else:
                    if torch.round(sentence).item() == 0:
                        true_negative_prediction += 1
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
            for instance in test_data:
                # send the data point through the model and get a prediction
                lstm_out, hidden = self.forward([instance[0]])

                # Calculate loss
                loss, accuracy_data = self.compute_loss(lstm_out, [instance[1]])
                aggregate_loss_sum += loss.item()

                # Add to accuracy variables
                true_predictions += accuracy_data[0] + accuracy_data[1]
                false_predictions += accuracy_data[2] + accuracy_data[3]

        accuracy = true_predictions / (true_predictions + false_predictions)
        aggregate_loss = aggregate_loss_sum / len(test_data)

        self.train()
        return accuracy, accuracy_data, math.exp(aggregate_loss), aggregate_loss