# Helpful tutorial
# at https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py

from typing import Dict
import math
import torch
import torch.nn.functional as F
from helper_functions import make_batch_vector, make_label_vectors


class SpeechLabeler(torch.nn.Module):
    def __init__(self,
                 tag_vocabulary: Dict[str, int],
                 rnn_hidden_size: int,
                 device: str = 'cpu'):
        
        super(SpeechLabeler, self).__init__()
        
        # remember device, vocabulary
        self.device = device
        self.tag_vocabulary = tag_vocabulary

        # Initialize encoder with an embedding layer and an LSTM
        self.lstm = torch.nn.LSTM(516, # output of SentenceTransformer
                                  rnn_hidden_size,
                                  batch_first=True,
                                  num_layers=1,
                                  bidirectional=True
                                  )
        
        # Hidden2tag linear layer takes the  LSTM output and projects to tag space
        self.hidden2tag = torch.nn.Linear(2*rnn_hidden_size, len(tag_vocabulary))

    def forward(self, batch_files, hidden=None):

        # Create Vector
        batch_vector = make_batch_vector(batch_files).to(self.device)

        # Send through LSTM (Initial hidden is defaulted to 0)
        lstm_out, hidden = self.lstm(batch_vector, hidden)

        return lstm_out, hidden
                 
    def tag_files(self, file: list):
        # sentence = sentence.strip().split(" ")

        file_pre_processed = []

        for line in file:
            # default is loading the corpus as sequence of words
            file_pre_processed.append(line.strip().split(""))

        lstm_out, hidden  = self.forward([file_pre_processed])
        prediction = self.make_prediction(lstm_out)

        for index, predicted_tag in enumerate(prediction[0]):
            # Todo: Iterate through every line of file, chop off its tags from the prediction and combine it with the XML tags
            file_tagged = file

        return file_tagged

        '''
        # Todo: Build tagging mechanism
        for index, predicted_tag in enumerate(prediction[0]):
            if torch.argmax(predicted_tag).item() == 1:
                coordinate_insert = [index][0] + 1 + error_count
                error_count += 1
                sentence.insert(coordinate_insert, "<ERR>")

                
        separator = ' '
        sentence = separator.join(sentence)
        return sentence
        '''
        # Durch die Wörterliste iterieren
    
    def compute_loss(self, lstm_out, batch_tags):
        prediction = self.make_prediction(lstm_out)
        vector_tags = make_label_vectors(batch_tags, self.tag_vocabulary).to(self.device)

        # compute the loss
        criterion = torch.nn.MultiLabelSoftMarginLoss()
        loss = criterion(prediction, vector_tags.float())

        # compute tp, fp
        tp: int = 0
        fp: int = 0
        tp_list = [0] * len(self.tag_vocabulary)
        fp_list = [0] * len(self.tag_vocabulary)

        for batch_index, batch_item in enumerate(prediction): # Batch
            for vector_index, predicted_tag_vector in enumerate(batch_item): # Line: Predicted Tag Vector
                tp_line: int = 0
                fp_line: int = 0

                for tag_index, predicted_tag in enumerate(predicted_tag_vector): # Predicted Tag
                    # print(torch.argmax(predicted_tag).item(), " - ", s[batch_index][index].item())
                    # Todo: Totaler Quatsch ist das natürlich.
                    # Das gibt nur die Wahrscheinlichkeit zurück, dass dieses Tag exisitiert
                    # Ersteres gibt ja immer immer 0 zurück, oder?
                    if torch.round(predicted_tag).item() == vector_tags[batch_index][vector_index][tag_index].item():
                        tp_line += 1
                        tp_list[tag_index] += 1
                    else:
                        fp_line += 1
                        fp_list[tag_index] += 1

                if fp_line < 1:
                    tp += 1
                else:
                    fp += 1

        accuracy_data = [tp, fp, tp_list, fp_list]

        return loss, accuracy_data
    
    def make_prediction(self, lstm_out):
        batch_tag_space = self.hidden2tag(lstm_out)
        prediction = torch.sigmoid(batch_tag_space)

        return prediction

    def evaluate(self, test_data):
        self.eval()

        # evaluate the model
        tp: int = 0
        fp: int = 0
        aggregate_loss_sum = 0
        tag_list_accuracy_pos = [0] * len(self.tag_vocabulary)
        tag_list_accuracy_neg = [0] * len(self.tag_vocabulary)

        with torch.no_grad():

            # go through all test data points
            for instance in test_data:
                # send the data point through the model and get a prediction
                lstm_out, hidden = self.forward([instance[0]])

                # Calculate loss
                loss, accuracy_data = self.compute_loss(lstm_out, [instance[1]])
                aggregate_loss_sum += loss.item()

                # Add to accuracy variables
                tp += accuracy_data[0]
                fp += accuracy_data[1]
                tag_list_accuracy_pos = [x + y for (x,y) in zip(tag_list_accuracy_pos, accuracy_data[2])]
                tag_list_accuracy_neg = [x + y for (x,y) in zip(tag_list_accuracy_neg, accuracy_data[3])]

        accuracy = tp / (tp + fp)
        aggregate_loss = aggregate_loss_sum / len(test_data)

        self.train()
        return accuracy, [tag_list_accuracy_pos, tag_list_accuracy_neg], math.exp(aggregate_loss), aggregate_loss