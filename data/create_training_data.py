# Create training data from annotated XML files

import pandas as pd
import os
import re
from lxml import etree
import pickle

import warnings
warnings.filterwarnings("ignore")

# Define paths
xml_path = "annotated_protocols"
export = "data.pkl"
comparison_export = "comparison_data.pkl"

# Create dataframe of possible cases
dataset_columns = ["protocol_id", "nr", "id", "string", "is_speech"]
dataset = pd.DataFrame(columns=dataset_columns)

# Iterate over files
df_list = []

for file in os.listdir(xml_path):
    if file.endswith('.xml'):
        filename = os.path.join(xml_path, file)

        # Öffne die XML-Datei
        tree = etree.parse(filename)
        root = tree.getroot()
        sitzungsverlauf = root.find('sitzungsverlauf')

        # Extrahiere die Protokoll-Meta-Daten
        period = int(root.get("wahlperiode"))
        protocol_number = int(root.get("sitzung-nr"))
        protocol_id = int(str(period) + str(protocol_number))

        # Iteriere über Reden
        df_match_list = []

        counter = 0
        for speech in sitzungsverlauf:
            # Find speech match
            counter += 1
            speech_name_data_match = re.match(r"^.{5,150}?:", speech.text.strip(), re.DOTALL)
            match = speech_name_data_match[0] if bool(speech_name_data_match) else ""

            # Add speech match to dataframe
            match_df = pd.DataFrame([[protocol_id, counter, int(str(protocol_id) + str(counter)), match, True]],
                                    columns=dataset_columns)
            df_match_list.append(match_df)

            # Find matches within speech
            # Matches strings with a length between 5-130 Characters that start after a new line and end with a colon
            # In between, there can be any character, paragraphs too
            # The longest true speech match in the annotated data is 120 characters long
            regex = r"^.{5,135}:"
            matches = re.finditer(regex, speech.text.strip()[len(match):].strip(), re.MULTILINE | re.DOTALL)

            # Create dataframe per match
            for matchNum, match in enumerate(matches, start=1):
                counter += 1

                # Clean strings, cut off content ahead of double paragraphs
                match_string = match.group().split("\n\n")[-1]

                # Combine to dataframe
                match_df = pd.DataFrame([[protocol_id, counter, int(str(protocol_id) + str(counter)), match_string, False]],
                                        columns=dataset_columns)
                df_match_list.append(match_df)

        # Combine list of matches per file
        dataset_file = pd.concat(df_match_list, ignore_index=True)
        df_list.append(dataset_file)

# Concat list of file matches to dataframe
dataset = pd.concat(df_list, ignore_index=True)
dataset_list = dataset.drop(['protocol_id', 'nr', "id"], axis=1).values.tolist()

# Save
with open(export, "wb") as fp:
    pickle.dump(dataset_list, fp)

with open(comparison_export, "wb") as fp:
    pickle.dump(dataset, fp)


print(len(dataset))