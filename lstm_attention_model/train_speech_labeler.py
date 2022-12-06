import time
from pathlib import Path
import torch
import random
import torch.optim as optim
import more_itertools
import pickle

from helper_functions import plot_loss_curves, make_dictionary
from speech_labeler import SpeechLabeler


# Variables
data = Path("../data/data.pkl")

# model_save_name
model_save_name = "models/best-speech-tagger.pt"

# All hyperparameters
learning_rate = 0.05
number_of_epochs = 10
embedding_size = 20
rnn_hidden_size = 50
mini_batch_size = 10
torch.manual_seed(1)
unk_threshold = 1
character_level = True

if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#    device = "mps"
else:
    device = torch.device("cpu")
    

print(f"Training speech labeler with \n - rnn_hidden_size: {rnn_hidden_size}\n - learning rate: {learning_rate}"
      f" \n - max_epochs: {number_of_epochs} \n - mini_batch_size: {mini_batch_size} ")

# -- STEP 1: LOAD TRAINING DATA
with open(data, 'rb') as pickle_file:
    file = pickle.load(pickle_file)

training_data = []
for observation in file:
    string = observation[0]
    classification = observation[1]

    training_data.append((string, classification))

# Shuffle training data
random.shuffle(training_data)

# create training, testing and validation splits
corpus_size = len(training_data)
validation_data = training_data[-round(corpus_size / 5):-round(corpus_size / 10)]
test_data = training_data[-round(corpus_size / 10):]
training_data = training_data[:-round(corpus_size / 5)]


# some helpful output
print(f"\nTraining corpus has "
      f"{len(training_data)} train, "
      f"{len(validation_data)} validation and "
      f"{len(test_data)} test sentences")

# -- STEP 2: MAKE DICTIONARIES
all_sentences = [pair[0] for pair in training_data]
all_tags = [pair[1] for pair in training_data]

word_vocabulary = make_dictionary(all_sentences, unk_threshold)

# some helpful output
print(f"\nTraining corpus has "
      f"{len(training_data)} train, "
      f"{len(validation_data)} validation and "
      f"{len(test_data)} test files")


# -- Step 3: initialize model and send to device
model = SpeechLabeler(
    word_vocabulary=word_vocabulary,
    rnn_hidden_size=rnn_hidden_size,
    embedding_size=embedding_size,
    device=device,
    character_level=character_level
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
f1_score_per_epoch = []

# remember the best model
best_model = None
best_epoch = 0
best_f1_score = 0

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

        output, attention = model.forward(batch_sentences)
        loss, _ = model.compute_loss(output, batch_tags)

        # remember loss and backpropagate
        train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

    train_loss /= len(batch)

    # Evaluate and print accuracy at end of each epoch
    accuracy, f1_score, accuracy_data, validation_perplexity, validation_loss = model.evaluate(test_data)

    # remember best model:
    if f1_score > best_f1_score:
        print("new best model found!")
        best_epoch = epoch
        best_f1_score = f1_score

        # always save best model
        torch.save(model, model_save_name)

    # print losses
    print(f"training loss: {train_loss}")
    print(f"validation loss: {validation_loss}")
    print(f"validation perplexity: {validation_perplexity}")
    print(f"validation accuracy: {accuracy}")
    print(f"F1 score: {f1_score}")
    print(f"tp: {accuracy_data[0]} tn: {accuracy_data[1]} fp: {accuracy_data[2]} fn: {accuracy_data[3]}")

    # append to lists for later plots
    train_loss_per_epoch.append(train_loss)
    validation_loss_per_epoch.append(validation_loss)
    validation_perplexity_per_epoch.append(validation_perplexity)
    f1_score_per_epoch.append(f1_score)

    end = time.time()
    print(f'{round(end - start, 3)} seconds for this epoch')
    
# do final test:
# load best model and do final test
best_model = torch.load(model_save_name)
test_accuracy, f1_score, accuracy_data, _, _ = best_model.evaluate(validation_data)

# print final score
print("\n -- Training Done --")
print(f" - using model from epoch {best_epoch} for final evaluation")
print(f"accuracy: {test_accuracy}")
print(f" - final score: {round(f1_score, 4)}"
      f"\n tp: {accuracy_data[0]} tn: {accuracy_data[1]} fp: {accuracy_data[2]} fn: {accuracy_data[3]}")


# make plots
plot_loss_curves(train_loss_per_epoch,
                 validation_loss_per_epoch,
                 f1_score_per_epoch,
                 approach_name="Sequence Labeler Model",
                 validation_label='F1-Score',
                 hyperparams={"rnn_hidden_size": rnn_hidden_size,
                              "lr": learning_rate})
