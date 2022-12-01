# bundestag_speech_tagger
A machine learning based classification system to find the beginning of speeches in Plenary protocols of the German Bundestag.  

The project [Open Discourse](https://opendiscourse.de/) offers a database of all the speeches that were held in the German Parliament, the Bundestag.
To accomplish this, they used the OCR- and/or PDF-extracted texts of the protocols and cut them into the single speeches by using Regex-based heuristics.
These complex heuristics cover the majority of the cases but around 3 % of the original speeches are missing.
Furthermore, there are speeches in Open Discourse corpora, that are in fact not speeches.

The heuristics are based on a few lines preceding the speech with the name, party and function of the speaker.
If these "meta information" about are found the system sees this as a start of new speech, and cuts of the preceding speech.
So a mistake here always also affects the previous speech as well.

It is important to note that this task is very complex as the structure of these lines and their information changed over 70 years,
there were also mistakes and OCR issues.

This project tries to get better results than a heuristic by using Natural Language Processing techniques.

# Data preparation
The first step is the data preparation: In order to train the model there is training data needed.
For this training data 36 out of the approximately 5000 protocols were annotated, containing over 7000 speeches.

From these annotated text files the lines that precede the speeches get extracted as they are what we want to classify. 
An possible alternative would obviously to use the whole text body and train it to insert tags where a new speech starts.
While this is a viable approach, the classification of single lines is more promising as we only need line(s) that end with a colon.
Some of those are the "meta information lines", some just speech content.
Further, the architectures we use here (LSTM & Transformers) in this case would need a sentence level encoding, instead of word level encoding. This would lead to big loss of informartion towards certain characters that help this classificiation.

Naturally the negatives are just speech content that includes a colon.
This also leads to a blind spot of this approach: If the colon is missing, either due to OCR or another mistake, 
the classification won't take place which would lead to a non-detected speech.
However, during the annotation process, there was not a single case of this found.

# LSTM architecture
final score: 0.9990

| #  | hidden | embedding | unk | lr   | batch | max epochs | used epoch | accuracy                 |
|----|--------|-----------|-----|------|-------|------------|------------|--------------------------|
| 1  | 64     | 50        | 1   | 0.05 | 10    | 5          | 3          | 0.9990 0.9981            |
| 2  | 100    | 50        | 1   | 0.05 | 50    | 5          | 3          | 0.9981                   |
| 3  | 100    | 50        | 1   | 0.05 | 10    | 5          | 0, 3       | 0.9976 0.9986            |
| 4  | 100    | 50        | 1   | 0.05 | 10    | 5          | 4          | 0.9981                   |
| 5  | 64     | 100       | 1   | 0.05 | 10    | 5, 10      | 3, 5       | 0.9990, 0.9981           |
| 6  | 64     | 200       | 1   | 0.05 | 10    | 7          | 4, 1, 5    | 1.0,  0.9952, 0.9981     |
| 7  | 64     | 400       | 1   | 0.05 | 10    | 7          | 0          | 0.9971                   |
| 8  | 64     | 300       | 1   | 0.05 | 10    | 7          | 3          | 0.9986                   |
| 9  | 64     | 200       | 3   | 0.05 | 10    | 7          | 3, 1, 2    | 0.9961,  0.9966,  0.9990 |
| 10 | 50     | 200       | 1   | 0.05 | 10    | 7          | 0, 0       | 0.9995, 1.0              |
# Transformer

# Model comparison