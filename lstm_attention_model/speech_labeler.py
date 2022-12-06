# Helpful tutorial
# at https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py

from typing import Dict
import math
import torch
import torch.nn.functional as F
from helper_functions import make_onehot_vectors, make_label_vectors,split_sentences


class SpeechLabeler(torch.nn.Module):
    def __init__(self,
                 word_vocabulary: Dict[str, int],
                 rnn_hidden_size: int,
                 embedding_size: int,
                 device: str,
                 character_level: bool):
        
        super(SpeechLabeler, self).__init__()
        
        # remember device, vocabulary and character_level
        self.device = device
        self.word_vocabulary = word_vocabulary
        self.character_level = character_level
        self.rnn_hidden_size = rnn_hidden_size

        # Initialize encoder with an embedding layer and an LSTM
        self.word_embedding = torch.nn.Embedding(len(word_vocabulary), embedding_size)
        self.lstm = torch.nn.LSTM(embedding_size,
                                  rnn_hidden_size,
                                  batch_first=True,
                                  num_layers=1,
                                  bidirectional=True
                                  )
        
        # Hidden2tag linear layer takes the LSTM output and projects to tag space
        self.linear = torch.nn.Linear(rnn_hidden_size * 2, 2)
        self.sigmoid = torch.nn.Sigmoid()

    def attention_net(self, lstm_out, final_state):
        # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        hidden = final_state.view(-1, self.rnn_hidden_size * 2, 1)

        attn_weights = torch.bmm(lstm_out, hidden).squeeze(2) # attn_weights : [batch_size, n_step]

        soft_attn_weights = F.softmax(attn_weights, 1)

        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_out.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return context, soft_attn_weights.data.numpy() # context : [batch_size, n_hidden * num_directions(=2)]

    def forward(self, batch_sentences, hidden=None):
        # Preprocessing
        batch_split_sentences = split_sentences(batch_sentences, self.character_level, self.training)

        # Create Vector
        vector_sentences = make_onehot_vectors(batch_split_sentences, self.word_vocabulary).to(self.device)

        # Make embeddings
        embedded_sentences = self.word_embedding(vector_sentences)

        # Send through LSTM (Initial hidden is defaulted to 0)
        lstm_out, hidden = self.lstm(embedded_sentences, hidden)

        # Get attention
        attn_output, attention = self.attention_net(lstm_out, hidden[0])

        return self.linear(attn_output), attention

    def make_prediction(self, output):
        prediction = output.data.max(1, keepdim=True)[1]

        # Send through activation
        # prediction = self.sigmoid(batch_label_space)

        return prediction
    
    def compute_loss(self, output, batch_tags):

        # Make Classification Vector
        vector_tags = make_label_vectors(batch_tags).to(self.device)

        # compute the loss
        loss_function = torch.nn.CrossEntropyLoss()
        loss = loss_function(output, vector_tags.to(torch.long))

        # Make prediction
        prediction = self.make_prediction(output)

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
            output, attention = self.forward(all_senteces)

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
        output, hidden = self.forward([string])
        prediction_torch = self.make_prediction(output)
        prediction_int = torch.round(prediction_torch).item()
        prediction = False if prediction_int == 0 else True

        return prediction, torch.round(prediction_torch, decimals=4).item()
