from pathlib import Path
from speech_labeler import SpeechLabeler
import torch
import pickle

import warnings
warnings.filterwarnings("ignore")

# oad models with and without attention
model_path = Path('models') / 'best-speech-tagger.pt'
classifier: SpeechLabeler = torch.load(model_path)
print(classifier)

# Example strings
strings = [
    "Die Leben ist schön",
    ") Dr. Wahl (CPU)",
    "Dr. Wahl (CDU)",
    "Dr. Töpfer, Bundesminister für Umwelt, Naturschutz und Reaktorsicherheit"
    ]

for string in strings:
    print("---")
    print(string)
    print(classifier.classify(string))

# Open annotated data
with open(Path("../data/comparison_data.pkl"), 'rb') as pickle_file:
    annotated_data = pickle.load(pickle_file)

# Classify training data
annotated_data["classifier_output"] = annotated_data.apply(lambda row: classifier.classify(row["string"]), axis=1)

mistakes = annotated_data[annotated_data["classifier_output"] != annotated_data["is_speech"]]
print(len(mistakes), " - ", round(len(mistakes) / len(annotated_data), 4))