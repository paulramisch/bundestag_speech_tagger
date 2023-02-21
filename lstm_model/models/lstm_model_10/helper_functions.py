import math
from typing import Dict
import re
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from textwrap import wrap

def read_data_from_file(path):
    data = []
    with open(path) as file:
        for line in file.readlines():
            line = line.strip()
            label = line.split(" ")[0][9:]
            text = line.split(" ")[1:]
            data.append((text, label))

    return data


def split_string(string):
    string_split = re.findall(r"[\u00C0-\u017Fa-zA-Z']+|[\(\)\[\]\*.,!?;€]", string)
    return string_split


def create_encoding(word):
    encoding = ""

    return encoding


def make_batch_vector(batch_files):
    batch = []

    for file in batch_files:
        tensor = torch.stack(file)
        batch.append(tensor)

    batch_vector = pad_sequence(batch, batch_first = True)

    return batch_vector

def make_dictionary(data, unk_threshold: int = 0) -> Dict[str, int]:
    '''
    Makes a dictionary of words given a list of tokenized sentences.
    :param data: List of (sentence, label) tuples
    :param unk_threshold: All words below this count threshold are excluded from dictionary and replaced with UNK
    :return: A dictionary of string keys and index values
    '''

    # First count the frequency of each distinct ngram
    word_frequencies = {}
    for sentence in data:
        for word in sentence:
            if word not in word_frequencies:
                word_frequencies[word] = 0
            word_frequencies[word] += 1

    # Assign indices to each distinct ngram
    word_to_ix = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in word_frequencies.items():
        if freq > unk_threshold:  # only add words that are above threshold
            word_to_ix[word] = len(word_to_ix)

    # Print some info on dictionary size
    print(f"At unk_threshold={unk_threshold}, the dictionary contains {len(word_to_ix)} words")
    return word_to_ix


def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        if word in word_to_ix:
            vec[word_to_ix[word]] += 1
    return vec.view(1, -1)


def make_label_vectors(batch):
    label_mini_batch = []

    for sentence in batch:
        # 0: False, 1: True
        onehot_for_sentence = 1 if sentence else 0
        label_mini_batch.append(onehot_for_sentence)

    return torch.tensor(label_mini_batch)


def make_onehot_vectors(sentences, word_to_ix):
    onehot_mini_batch = []

    longest_sequence_in_batch = max([len(sentence) for sentence in sentences])

    for sentence in sentences:

        onehot_for_sentence = []

        # move a window over the text
        for word in sentence:

            # look up ngram index in dictionary
            if word in word_to_ix:
                onehot_for_sentence.append(word_to_ix[word])
            else:
                onehot_for_sentence.append(word_to_ix["UNK"] if "UNK" in word_to_ix else 0)

        for i in range(longest_sequence_in_batch - len(sentence)):
            onehot_for_sentence.append(0)

        onehot_mini_batch.append(onehot_for_sentence)

    return torch.tensor(onehot_mini_batch)


def plot_loss_curves(loss_train,
                     loss_val,
                     accuracy_val,
                     approach_name: str,
                     hyperparams,
                     validation_label='Validation accuracy'):
    last_finished_epoch = len(loss_train)
    epochs = range(1, last_finished_epoch + 1)
    hyperparam_pairs = [f"{key}{hyperparams[key]}" for key in hyperparams]

    file_name = f"models/loss-curves-{approach_name}-" + "-".join(hyperparam_pairs).replace("/", "-") + ".png"
    title_text = ", ".join([f"{key}:{hyperparams[key]}" for key in hyperparams])

    fig, ax1 = plt.subplots()

    color = 'g'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, loss_train, 'r', label='Training loss')
    ax1.plot(epochs, loss_val, 'g', label='Validation loss')
    ax1.tick_params(axis='y', labelcolor=color)
    title = ax1.set_title("\n".join(wrap(title_text, 60)))
    title.set_y(1.05)
    fig.subplots_adjust(top=0.8)
    ax1.grid()

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'k'  # k := black
    ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(epochs, accuracy_val, 'black', label=validation_label)
    ax2.tick_params(axis='y', labelcolor=color)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.xticks(range(5, math.floor((last_finished_epoch + 1) / 5) * 5, 5))
    plt.savefig(file_name)
    plt.show()
