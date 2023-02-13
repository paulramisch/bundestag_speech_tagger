import regex
from pathlib import Path
import pickle
import pandas as pd

# Clean function
# https://github.com/open-discourse/open-discourse/blob/03225c25c451b8331a3dcd25937accc70c44d9ad/python/src/od_lib/helper_functions/clean_text.py#L5
def clean(filetext, remove_pdf_header=True):
    # Replaces all the misrecognized characters
    filetext = filetext.replace(r"", "-")
    filetext = filetext.replace(r"", "-")
    filetext = filetext.replace("—", "-")
    filetext = filetext.replace("–", "-")
    filetext = filetext.replace("•", "")
    filetext = regex.sub(r"\t+", " ", filetext)
    filetext = regex.sub(r"  +", " ", filetext)

    # Remove pdf artifact
    if remove_pdf_header:
        filetext = regex.sub(
            r"(?:Deutscher\s?Bundestag\s?-(?:\s?\d{1,2}\s?[,.]\s?Wahlperiode\s?-)?)?\s?\d{1,3}\s?[,.]\s?Sitzung\s?[,.]\s?(?:(?:Bonn|Berlin)[,.])?\s?[^,.]+,\s?den\s?\d{1,2}\s?[,.]\s?[^\d]+\d{4}.*",  # noqa: E501
            r"\n",
            filetext,
        )
        filetext = regex.sub(r"\s*(\(A\)|\(B\)|\(C\)|\(D\))", "", filetext)

    # Remove delimeter
    filetext = regex.sub(r"-\n+(?![^(]*\))", "", filetext)

    # Deletes all the newlines in brackets
    bracket_text = regex.finditer(r"\(([^(\)]*(\(([^(\)]*)\))*[^(\)]*)\)", filetext)

    for bracket in bracket_text:
        filetext = filetext.replace(
            str(bracket.group()),
            regex.sub(
                r"\n+",
                " ",
                regex.sub(
                    r"(^((?<!Abg\.).)+|^.*\[.+)(-\n+)",
                    r"\1",
                    str(bracket.group()),
                    flags=regex.MULTILINE,
                ),
            ),
        )
    return filetext


def get_op_match(string, protocol_id):
    string = clean(string)

    president_pattern_str = r"(?P<position_raw>Präsident(?:in)?|Vizepräsident(?:in)?|Alterspräsident(?:in)?|Bundespräsident(?:in)?|Bundeskanzler(?:in)?)\s+(?P<name_raw>[A-ZÄÖÜß](?:[^:([}{\]\)\s]+\s?){1,5})\s?:\s?"
    faction_speaker_pattern_str = r"{3}(?P<name_raw>[A-ZÄÖÜß][^:([{{}}\]\)\n]+?)(\s*{0}(?P<constituency>[^:(){{}}[\]\n]+){1})*\s*{0}(?P<position_raw>{2}){1}(\s*{0}(?P<constituency>[^:(){{}}[\]\n]+){1})*\s?:\s?"
    minister_pattern_str = r"{0}(?P<name_raw>[A-ZÄÖÜß](?:[^:([{{}}\]\)\s]+\s?){{1,5}}?),\s?(?P<position_raw>(?P<short_position>Bundesminister(?:in)?|Staatsminister(?:in)?|(?:Parl\s?\.\s)?Staatssekretär(?:in)?|Präsident(?:in)?|Bundeskanzler(?:in)?|Schriftführer(?:in)?|Senator(?:in)?\s?(?:{1}(?P<constituency>[^:([{{}}\]\)\s]+){2})?|Berichterstatter(?:in)?)\s?([^:([\]{{}}\)\n]{{0,76}}?\n?){{1,2}})\s?:\s?"

    # List of parties
    parties = [
        r"(?:Gast|-)?(?:\s*C\s*[DSMU]\s*S?[DU]\s*(?:\s*[/,':!.-]?)*\s*(?:\s*C+\s*[DSs]?\s*[UÙ]?\s*)?)(?:-?Hosp\.|-Gast|1)?",
        r"\s*'?S(?:PD|DP)(?:\.|-Gast)?",
        r"\s*F\.?\s*[PDO][.']?[DP]\.?",
        r"(?:BÜNDNIS\s*(?:90)?/?(?:\s*D[1I]E)?|Bündnis\s*90/(?:\s*D[1I]E)?)?\s*[GC]R[UÜ].?\s*[ÑN]EN?(?:/Bündnis 90)?",
        r"DIE LINKE",
        r"(?:Gruppe\s*der\s*)?PDS(?:/(?:LL|Linke Liste))?",
        r"(fraktionslos|Parteilos)",
        r"(?:GB[/-]\s*)?BHE(?:-DG)?",
        "DP",
        "KPD",
        "Z",
        "BP",
        "FU",
        "WAV",
        r"DRP(\-Hosp\.)",
        "FVP",
        "SSW",
        "SRP",
        "DA",
        "Gast",
        "DBP",
        "NR",
    ]

    # Set brackets and prefix
    if int(protocol_id) in [120, 1229, 287, 2202, 318, 3103, 415, 4184, 5101, 5158,
                            617, 6185, 7118, 7140, 825, 880, 914, 9126, 1061, 10213]:
        open_brackets = r"[({\[]"
        close_brackets = r"[)}\]]"
        prefix = r"(?<=\n)"
    else:
        open_brackets = r"[(]"
        close_brackets = r"[)]"
        prefix = r"(?<=\n)"

    # Compile patters
    faction_speaker_pattern = regex.compile(
        faction_speaker_pattern_str.format(open_brackets, close_brackets, "|".join(parties), prefix)
    )
    president_pattern = regex.compile(president_pattern_str)
    minister_pattern = regex.compile(minister_pattern_str.format(prefix, open_brackets, close_brackets))

    patterns = [president_pattern, faction_speaker_pattern, minister_pattern]

    result = False
    for pattern in patterns:
        for match in regex.finditer(pattern, string):
            result = True


    return result


# Run test
def run_test(df):
    def run_loopy(df):
        results = []
        for _, row in df.iterrows():
            prefix = "\n" if row["is_speech"] else ""
            result = get_op_match(prefix + row["string"], row["protocol_id"])
            results.append(result)
        return pd.Series({'classifier_output': results})

    # Get Classification
    df[['classifier_output']] = run_loopy(df)

    # Create dataframe of mistakes
    mistakes = df[df["classifier_output"] != df["is_speech"]]

    # Get mistake count
    true_positive_prediction = len(df[(df["is_speech"] == True) & (df["classifier_output"] == df["is_speech"])])
    true_negative_prediction = len(df[(df["is_speech"] == False) & (df["classifier_output"] == df["is_speech"])])
    false_positive_prediction = len(df[(df["is_speech"] == False) & (df["classifier_output"] != df["is_speech"])])
    false_false_prediction = len(df[(df["is_speech"] == True) & (df["classifier_output"] != df["is_speech"])])

    # Calculate f1 score
    f1_score = (2 * true_positive_prediction / (
                2 * true_positive_prediction + false_positive_prediction + false_false_prediction))

    # Print results
    print(len(mistakes), " - ", round(len(mistakes) / len(df), 4))
    print(f"Final F1 score: {round(f1_score, 4)}"
          f"\ntp: {true_positive_prediction} tn: {true_negative_prediction} fp: {false_positive_prediction} fn: {false_false_prediction}")


# Open annotated data
with open(Path("../data/comparison_data.pkl"), 'rb') as pickle_file:
    annotated_data = pickle.load(pickle_file)

run_test(annotated_data)
