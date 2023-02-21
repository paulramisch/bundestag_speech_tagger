from pathlib import Path
from speech_labeler import SpeechLabeler
import torch
import pickle
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# oad models with and without attention
model_path = Path('models') / 'best-speech-tagger.pt'
classifier: SpeechLabeler = torch.load(model_path)
print(classifier)

# Example strings
strings = ["Die Leben ist schön", ") Dr. Wahl (CPU)",
    "Dr. Wahl (CDU)", "Dr. Töpfer, Bundesminister für Umwelt, Naturschutz und Reaktorsicherheit"]

for string in strings:
    print("---")
    print(string)
    print(classifier.classify(string))
    print(classifier.classify_batch([string]))


def run_test(df):
    def run_loopy_list(df):
        batches = [df[i:i + 20] for i in range(0, df.shape[0], 20)]
        Cs, Ds = [], []
        for batch in batches:
            strings = batch["string"].tolist()
            c_batch, d_batch = classifier.classify_batch(strings)
            Cs.extend(c_batch)
            Ds.extend(d_batch)
        return pd.Series({'classifier_output': Cs, 'int': Ds})

    # Get Classification batch (single classification don't work well due to padding issues)
    df[['classifier_output', 'int']] = run_loopy_list(df)

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


# Open annotated data
with open(Path("../data/comparison_data_noise.pkl"), 'rb') as pickle_file:
    annotated_data = pickle.load(pickle_file)

# Random shuffle the data
annotated_data = annotated_data.sample(frac=1).reset_index(drop=True)

run_test(annotated_data)
