import math
import torch
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from textwrap import wrap
import re

import fasttext.util
fasttext.util.download_model('de', if_exists='ignore')  # English
ft = fasttext.load_model('cc.de.300.bin')

def read_data_from_file(path):
    data = []
    with open(path) as file:
        for line in file.readlines():
            line = line.strip()
            label = line.split(" ")[0][9:]
            text = line.split(" ")[1:]
            data.append((text, label))

    return data


def make_batch_vector(batch_files):
    batch = []

    for file in batch_files:
        tensor = torch.stack(file)
        batch.append(tensor)

    batch_vector = pad_sequence(batch, batch_first = True)

    return batch_vector


def make_label_vectors(batch):
    label_mini_batch = []

    for sentence in batch:
        # 0: False, 1: True
        onehot_for_sentence = 1 if sentence else 0
        label_mini_batch.append(onehot_for_sentence)

    return torch.tensor(label_mini_batch)


def split_sentences(sentences):
    sentences_split = []

    for sentence in sentences:
        sentence_split = re.findall(r"[\u00C0-\u017Fa-zA-Z']+|[\(\)\[\]\*.,!?;€]", sentence)
        sentences_split.append(sentence_split)

    return sentences_split


def make_ft_vectors(raw_sentences):
    sentences = split_sentences(raw_sentences)

    longest_sequence_in_batch = max([len(sentence) for sentence in sentences])
    ft_vector_batch = []

    for sentence in sentences:
        sentence_vector = []
        for word in sentence:
            word_vector = ft.get_word_vector(word)
            sentence_vector.append(word_vector)

        for i in range(longest_sequence_in_batch - len(sentence)):
            sentence_vector.append(torch.zeros(300))

        ft_vector_batch.append(sentence_vector)

    return torch.tensor(ft_vector_batch)


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
