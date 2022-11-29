import time
from pathlib import Path
import torch
import random
import torch.optim as optim
import numpy
import os # To count files within a folder
import pickle

from helper_functions import plot_loss_curves, make_label_dictionary, make_sbert_encoding
from speech_labeler import SpeechLabeler


torch.manual_seed(1)

# Variables
data = Path("../data/data.pkl")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = torch.device("cpu")
    
# All hyperparameters
learning_rate = 0.05
number_of_epochs = 30
rnn_hidden_size = 500
mini_batch_size = 4

# model_save_name = "experiments/best-grammar-corrector.pt"
model_save_name = "models/best-speech-tagger.pt"

print(f"Training speech labeler with \n - rnn_hidden_size: {rnn_hidden_size}\n - learning rate: {learning_rate}"
      f" \n - max_epochs: {number_of_epochs} \n - mini_batch_size: {number_of_epochs} ")

# -- STEP 1: LOAD TRAINING DATA
training_data = []

with open(data, 'rb') as pickle_file:
    file = pickle.load(pickle_file)

for observation in file:
    try:
        string = observation[0]
        is_speech = observation[1]

        # Encoding step

        # Encoded
        encoding = training_data.append([string, string], dtype=numpy.float32)

    except:
        print("An exception occurred in")

training_data.append((file_encoded_texts, file_tags))


# create training, testing and validation splits
corpus_size = len(training_data)
validation_data = training_data[-round(corpus_size / 5):-round(corpus_size / 10)]
test_data = training_data[-round(corpus_size / 10):]
training_data = training_data[:-round(corpus_size / 5)]

# some helpful output
print(f"\nTraining corpus has "
      f"{len(training_data)} train, "
      f"{len(validation_data)} validation and "
      f"{len(test_data)} test files")

# -- STEP 2: MAKE TAG DICTIONARY
all_tags = [pair[1] for pair in training_data]
tag_vocabulary = make_label_dictionary(all_tags)

# -- Step 3: initialize model and send to device
model = SpeechLabeler(  tag_vocabulary = tag_vocabulary,
                        rnn_hidden_size = rnn_hidden_size,
                        device = device
                        )

model.to(device)
print(model)

# -- Step 4: Do a training loop
# define a simple SGD optimizer with a learning rate of 0.1
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)

# log the losses and accuracies
train_loss_per_epoch = []
validation_loss_per_epoch = []
validation_perplexity_per_epoch = []
accuracy_per_epoch = []

# remember the best model
best_model = None
best_epoch = 0
best_accuracy = 0.

# Go over the training dataset multiple times
# Go over the training dataset multiple times
for epoch in range(number_of_epochs):

    print(f"\n - Epoch {epoch}")
    start = time.time()

    # shuffle training data at each epoch
    random.shuffle(training_data)

    train_loss = 0.
    hidden = None

    for batch in more_itertools.chunked(training_data, mini_batch_size):
        # Remember that PyTorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Run our forward pass.
        batch_sentences = [pair[0] for pair in batch]
        batch_tags = [pair[1] for pair in batch]

        lstm_out, hidden = model.forward(batch_sentences)
        loss, _ = model.compute_loss(lstm_out, batch_tags)

        # remember loss and backpropagate
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss /= len(batch)

    # Evaluate and print accuracy at end of each epoch
    accuracy, tag_list_accuracy, validation_perplexity, validation_loss \
        = model.evaluate(test_data)

    # remember best model:
    if accuracy > best_accuracy:
        print("new best model found!")
        best_epoch = epoch
        best_accuracy = accuracy

        # always save best model
        torch.save(model, model_save_name)

    # print losses
    print(f"training loss: {train_loss}")
    print(f"validation loss: {validation_loss}")
    print(f"validation perplexity: {validation_perplexity}")
    print(f"accuracy: {accuracy}")

    for item_index, item in enumerate(tag_list_accuracy[0]):
        print(list(tag_vocabulary.keys())[list(tag_vocabulary.values()).index(item_index)],
              "- true:", tag_list_accuracy[0][item_index],
              " false:", tag_list_accuracy[1][item_index],
              " acurracy:", tag_list_accuracy[0][item_index] / (tag_list_accuracy[0][item_index] + tag_list_accuracy[1][item_index]))


    # append to lists for later plots
    train_loss_per_epoch.append(train_loss)
    validation_loss_per_epoch.append(validation_loss)
    validation_perplexity_per_epoch.append(validation_perplexity)
    accuracy_per_epoch.append(accuracy)

    end = time.time()
    print(f'{round(end - start, 3)} seconds for this epoch')
    
# do final test:
# load best model and do final test
best_model = torch.load(model_save_name)
test_accuracy, tag_list_accuracy, _, _ = best_model.evaluate(validation_data)

# print final score
print("\n -- Training Done --")
print(f" - using model from epoch {best_epoch} for final evaluation")
print(f" - final score: {test_accuracy}")

for item_index, item in enumerate(tag_list_accuracy[0]):
    print(list(tag_vocabulary.keys())[list(tag_vocabulary.values()).index(item_index)],
          "- true:", tag_list_accuracy[0][item_index],
          " false:", tag_list_accuracy[1][item_index],
          " acurracy:",
          tag_list_accuracy[0][item_index] / (tag_list_accuracy[0][item_index] + tag_list_accuracy[1][item_index]))

# make plots
plot_loss_curves(train_loss_per_epoch,
                 validation_loss_per_epoch,
                 accuracy_per_epoch,
                 approach_name="Sequence Labeler Model",
                 validation_label='Accuracy',
                 hyperparams={"rnn_hidden_size": rnn_hidden_size,
                              "lr": learning_rate})
