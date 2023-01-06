# bundestag_speech_tagger
A machine learning based classification system to find the beginning of speeches in Plenary protocols of the German Bundestag.

The project [Open Discourse](https://opendiscourse.de/) offers a database of all the speeches that were held in the German Parliament, the Bundestag. To accomplish this, the research team used OCR- and/or PDF-extracted texts of the protocols and cut them into the single speeches by using Regex-based heuristics.
These complex heuristics cover the majority of the cases but around 3 % of the original speeches are missing.
Furthermore, there are speeches in Open Discourse corpora, that are in fact not speeches.

The heuristics are based on a few lines preceding the speech with the name, party and function of the speaker.
If these "meta information" about are found the system sees this as a start of new speech, and cuts of the preceding speech. So a mistake here always also affects the previous speech as well.

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

## Accuracy
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
| 11 | 50     | 200       | 1   | 0.04 | 10    | 7          | 2          | 0.9995, 0.9966           |
| 12 | 50     | 200       | 10  | 0.048| 10    | 7          | 6          | 0.9061                   |

## F1-Score
| #   | ch_level | hidden | embedding | unk | lr   | batch | max epochs | used epoch | f1_score            |
|-----|----------|--------|-----------|-----|------|-------|------------|------------|---------------------|
| 12  |          | 50     | 200       | 10  | 0.048| 10    | 7          | 6          | 0.8157              |
| 13  |          | 50     | 500       | 10  | 0.05 | 10    | 10         | 2          | 0.8674              |
| 14  | X        | 50     | 500       | 1   | 0.05 | 10    | 10         | 3          | 0.9974              | 
| 15  | X        | 50     | 50        | 1   | 0.05 | 10    | 10         | 4          | 0.998               |
| 16  | X        | 50     | 50        | 1   | 0.05 | 10    | 5          | 4          | 0.9901              |
| 17  | X        | 50     | 20        | 1   | 0.05 | 10    | 5          | 4,3,4      | 1.0, 0.9954, 0.9987 |
| 18  | X        | 100    | 20        | 1   | 0.05 | 10    | 5          | 4          | 0.998               |
| 19  | X        | 20     | 20        | 1   | 0.05 | 10    | 5          | 1          | 0.9968              |
| 20  | X        | 40     | 15        | 1   | 0.05 | 10    | 5          | 2          | 0.9993              |
| 21  | X        | 50     | 15        | 1   | 0.05 | 10    | 5          | 2          | 0.9966              |
| 22  | X        | 50     | 25        | 1   | 0.05 | 10    | 5          | 4          | 0.9987              |
| 23  | X        | 50     | 20        | 1   | 0.05 | 10    | 5          | 2          | 0.9949              |
| 24  | X        | 50     | 20        | 1   | 0.05 | 10    | 8          | 7, 7       | 0.9990, 0.9993      |
| 25  | X        | 50     | 20        | 1   | 0.05 | 10    | 8          | 7          | 0.9975              |
| 26  | X        | 50     | 20        | 1   | 0.05 | 10    | 8          | 7          | 0.9987              |


Scores:
Model 12: 2092  -  0.1007
Model 10: 992  -  0.0477
Model 14: 83  -  0.0040
Model 15: 16  -  0.0008
Model 16: 57  -  0.0027
Model 17: 25  -  0.0012 / 123  -  0.0059 / 36  -  0.0017 / 20  -  0.001
Model 18: 42  -  0.0020
Model 19: 87  -  0.0042
Model 20: 28  -  0.0013
Model 21: 44  -  0.0021
Model 22: 43  -  0.0021
Model 23: 25  -  0.0012
Model 24: 13  -  0.0006
Model 25: 15  -  0.0007
Model 26: 15  -  0.0007


--
Model 17
20  -  0.001
Final F1 score: 0.9987
tp: 7534 tn: 13222 fp: 8 fn: 12

Model 24
Final F1 score: 0.9991
tp: 7541 tn: 13222 fp: 8 fn: 5

Final F1 score: 0.9985
tp: 7539 tn: 13215 fp: 15 fn: 7

Model 25
Final F1 score: 0.9990
tp: 7537 tn: 13224 fp: 6 fn: 9

Model 26
Final F1 score: 0.9988
tp: 7541 tn: 13217 fp: 13 fn: 5

Major Änderungen:
* Ab 12: Bewertung anhand des F1-Scores, statt Accuracy
* Ab 14: Character Level
* Ab 16: Vier fehlende Redebeiträge annotiert
* Ab 16: Random Masking von 25 % der Redebeiträge
* nur 23: Random Masking von 40 % der Redebeiträge
* Ab 24: Dropout 0.2, 2 layer added, 0.3 masking
* 25: Dropout 0.3, 0.2 masking
* 26: Dropout 0.1, 0.3 masking

## Isuess
With words, the system just learns the names of the politicians, the scores are high for those. The moment we increase the threshold for UNK words, the system fails to deliever good results.

By classifing not on the word, but character level the perfomance for these cases can significantly be incrased. Still the system has issues with functions, it does not know. 

We try to act on this by randomly masking the input.

# LSTM architecture with Fasttext embeddings
The previous LSTM model learns its embeddings just from the limited number of words represented in the given strings. Another possibility is feeding the LSTM with Word2Vec or Fasttext vectors. In this example we use a (Fasttext model pretrained on German Webcrawl and Wikipedia texts)[https://fasttext.cc/docs/en/crawl-vectors.html]. The embeddings have a size of 300.

The Fasttext model is automaticly downloaded, unpacked it takes up 7.2 gb of disk space.

The results 
| #   | hidden | lr   | batch | max epochs | used epoch | f1_score              |
|-----|--------|------|-------|------------|------------|-----------------------|
| 1   | 50     | 0.05 | 10    | 5          | 3          | 0.9955                |
| 2   | 100    | 0.05 | 10    | 5          | 4, 2, 1    | 0.9964, 0.987, 0.9974 |
| 3   | 150    | 0.04 | 10    | 5          | 0          | 0.9953                |
| 4   | 70     | 0.05 | 10    | 5          | 2, 3       | 0.9928, 0.994         |

Model 3:
1096  -  0.0528
Final F1 score: 0.9218
tp: 6458 tn: 13222 fp: 8 fn: 1088

Bad results because of the padding in the test data

 - using model from epoch 3 for final evaluation
accuracy: 0.9980741454020221
 - final score: 0.9974
 tp: 781 tn: 1292 fp: 4 fn: 0


# Pretrained Bert model

| #   | used epoch | f1_score  |
|-----|------------|-----------|
| 1   | 2          | 0.9968    |
| 2   | 4          | 0.9987    |
| 3   | 5          | 0.9987    |

2 Epoch: 5  -  0.0002
4 Epoch: 2  -  0.0001
5 Epoch: 2  -  0.0001

0.9987

14150: "Günther Friedrich Nolting (F.D.P.) (von Abgeordne-\nten der F.D.P. mit Beifall begrüßt):"
09126: "Häfele, zur Konzeption:"

# Model comparison
Im OP-Korpus sind in der betrachteten Stichprobe rund 233 von 7542 Redebeiträgen nicht erkannt worden, also 3,09 %. Weiterhin wurden 48 Redebeiträge erkannt, die eigentlich keine Redebeiträge sind.

How to compare 
Understanding Recall, Precision, F1-Score
https://medium.com/techspace-usict/measuring-just-accuracy-is-not-enough-in-machine-learning-a-better-technique-is-required-e7199ac36856


https://en.wikipedia.org/wiki/Sensitivity_and_specificity
Therefore F1-Score (2 * true_positive / (2 *  true_positive + false_positive + false_negative)):
2 * 7309 / (2* 7309 + 233 + 48) = 0.9811

Versus: 
2 * 746 / (2 * 746 + 1 + 1) = 0.9987

- ID-Tabelle