from pathlib import Path
from speech_labeler import SpeechLabeler
from helper_functions import split_string
import torch
import pickle
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# oad models with and without attention
model_path = Path('models') / '10_best-speech-tagger.pt'
classifier: SpeechLabeler = torch.load(model_path)
print(classifier)

# Open annotated data
with open(Path("../data/comparison_data.pkl"), 'rb') as pickle_file:
    annotated_data = pickle.load(pickle_file)

# Classify training data
# https://stackoverflow.com/questions/47969756/pandas-apply-function-that-returns-two-new-columns
# annotated_data["classifier_output"], annotated_data["int"] = annotated_data.apply(lambda row: classifier.classify(row["string"]), axis=1)

def run_test(df):
    # Get Classification
    c, d, = classifier.classify_batch(df["string"].apply(split_string))
    data = pd.Series({'classifier_output': c, 'int': d})
    df[['classifier_output', 'int']] = data

    # Create dataframe of mistakes
    mistakes = df[df["classifier_output"] != df["is_speech"]]

    # Get mistake count
    true_positive_prediction = len(df[(df["is_speech"] == True) & (df["classifier_output"] == df["is_speech"])])
    true_negative_prediction = len(df[(df["is_speech"] == False) & (df["classifier_output"] == df["is_speech"])])
    false_positive_prediction = len(df[(df["is_speech"] == False) & (df["classifier_output"] != df["is_speech"])])
    false_false_prediction = len(df[(df["is_speech"] == True) & (df["classifier_output"] != df["is_speech"])])

    # Calculate f1 score
    f1_score = (2 * true_positive_prediction / (2 * true_positive_prediction + false_positive_prediction + false_false_prediction))

    # Print results
    print(len(mistakes), " - ", round(len(mistakes) / len(df), 4))
    print(f"Final F1 score: {round(f1_score, 4)}"
          f"\ntp: {true_positive_prediction} tn: {true_negative_prediction} fp: {false_positive_prediction} fn: {false_false_prediction}")


run_test(annotated_data)
