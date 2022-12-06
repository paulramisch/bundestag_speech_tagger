from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import pickle
import random
import torch

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Get data
with open("../data/data.pkl", 'rb') as pickle_file:
    file = pickle.load(pickle_file)

# Fill list of data
data = []
for observation in file:
    string = observation[0]
    # 0: False, 1: True
    classification = 1 if observation[1] else 0
    data.append((string, classification))

# Shuffle training data
random.shuffle(data)

# create training, testing and validation splits
corpus_size = len(data)

# Create evaluation data
validation_data = data[-round(corpus_size / 5):-round(corpus_size / 10)]
eval_df = pd.DataFrame(validation_data)
eval_df.columns = ["text", "labels"]

# Create training data
train_data = data[:-round(corpus_size / 5)]
train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]

# Create test data
test_data = data[-round(corpus_size / 10):]

# Optional model configuration
model_args = ClassificationArgs(
    num_train_epochs=5)
cuda_available = torch.cuda.is_available()

# Create a ClassificationModel
model = ClassificationModel(
    "bert", "bert-base-cased", args=model_args, use_cuda=cuda_available
)

# Train the model
if __name__ == '__main__':
    model.train_model(train_df)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)

