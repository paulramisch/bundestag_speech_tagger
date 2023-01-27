# bundestag_speech_tagger
A machine learning based classification system to find the beginning of speeches in Plenary protocols of the German Bundestag.

The project [Open Discourse](https://opendiscourse.de/) offers a database of all the speeches that were held in the German Parliament, the Bundestag. To accomplish this, the research team used OCR- and/or PDF-extracted texts of the protocols and cut them into the single speeches by using Regex-based heuristics.
These complex heuristics cover the majority of the cases but approximately 3 % of the original speeches are missing.
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

# Measure: Accuracy vs. F1-Score
Im OP-Korpus sind in der betrachteten Stichprobe rund 233 von 7542 Redebeiträgen nicht erkannt worden, also 3,09 %. Weiterhin wurden 48 Redebeiträge erkannt, die eigentlich keine Redebeiträge sind.

How to compare 
Understanding Recall, Precision, F1-Score
https://medium.com/techspace-usict/measuring-just-accuracy-is-not-enough-in-machine-learning-a-better-technique-is-required-e7199ac36856


https://en.wikipedia.org/wiki/Sensitivity_and_specificity
Therefore F1-Score (2 * true_positive / (2 *  true_positive + false_positive + false_negative)):
2 * 7309 / (2* 7309 + 233 + 48) = 0.9811


# Model comparison
## LSTM architecture

Changes:
* Accuracy measure
* character level
* missing speeches annotated (4)
* random masking of charachter 20 - 40%
* 2 Layer
* Dropout


### Isuess with LSTM
With words, the system just learns the names of the politicians, the scores are high for those. The moment we increase the threshold for UNK words, the system fails to deliever good results.

By classifing not on the word, but character level the perfomance for these cases can significantly be incrased. Still the system has issues with functions, it does not know. 

We try to act on this by randomly masking the input.

### LSTM architecture with Fasttext embeddings
The previous LSTM model learns its embeddings just from the limited number of words represented in the given strings. Another possibility is feeding the LSTM with Word2Vec or Fasttext vectors. In this example we use a (Fasttext model pretrained on German Webcrawl and Wikipedia texts)[https://fasttext.cc/docs/en/crawl-vectors.html]. The embeddings have a size of 300.

The Fasttext model is automaticly downloaded, unpacked it takes up 7.2 gb of disk space.

Bad results because of the padding in the test data


## Pretrained Bert model

| #   | used epoch | f1_score  |
|-----|------------|-----------|
| 2   | 2          | 0.9968    |
| 4   | 4          | 0.9987    |
| 5   | 5          | 0.9987    |

2 Epoch: 5  -  0.0002
4 Epoch: 2  -  0.0001
5 Epoch: 2  -  0.0001

14150: "Günther Friedrich Nolting (F.D.P.) (von Abgeordne-\nten der F.D.P. mit Beifall begrüßt):"
09126: "Häfele, zur Konzeption:"

# Model comparison
| model          | td_mistakes | td_accuracy | td_f1_score | fd_mistakes | fd_accuracy | fd_f1_score |
|----------------|-------------|-------------|-------------|-------------|-------------|-------------|
| lstm_10.2      | 0           | 1.0         | 1.0         | 992         | 0.9523      |             |
| lstm_17.1      | 0           | 1.0         | 1.0         | 25          | 0.9988      |             |
| lstm_20        | 1           | 0.9995      | 0.9993      | 28          | 0.9987      |             |
| lstm_24.1      | 2           | 0.9990      | 0.9987      | 13          | 0.9994      | 0.9991      |
| lstm_24.2      | 1           | 0.9995      | 0.9993      | 22          | 0.9989      | 0.9985      |
| fast_3         | 7           | 0.9966      | 0.9953      | 1096        | 0.9218      | 0.9218      |
| bert_2         | 5           | 0.9976      | 0.9968      | 5           | 0.9998      | 0.9998      |
| bert_4         | 2           | 0.9990      | 0.9987      | 2           | 0.9999      | 0.9999      |
| open_discourse |             |             |             | 281         |             | 0.9811      |



# Appendix
## Validation on whole dataset
| model     | mistakes | tp   | tn    | fp | fn   | accuracy | f1_score | comment     |
|-----------|----------|------|-------|----|------|----------|----------|-------------|
| lstm_10.2 | 992      |      |       |    |      | 0.9523   |          |             |
| lstm_12   | 2092     |      |       |    |      | 0.8993   |          |             |
| lstm_14   | 83       |      |       |    |      | 0.996    |          |             |
| lstm_15   | 16       |      |       |    |      | 0.9992   |          |             |
| lstm_16   | 57       |      |       |    |      | 0.9973   |          |             |
| lstm_17.1 | 25       |      |       |    |      | 0.9988   |          |             |
| lstm_17.2 | 123      |      |       |    |      | 0.9941   |          |             |
| lstm_17.3 | 36       |      |       |    |      | 0.9983   |          |             |
| lstm_17.4 | 20       | 7534 | 13222 | 8  | 12   | 0.999    | 0.9987   |             |
| lstm_18   | 42       |      |       |    |      | 0.998    |          |             |
| lstm_19   | 87       |      |       |    |      | 0.9958   |          |             |
| lstm_20   | 28       |      |       |    |      | 0.9987   |          |             |
| lstm_21   | 44       |      |       |    |      | 0.9979   |          |             |
| lstm_22   | 43       |      |       |    |      | 0.9979   |          |             |
| lstm_23   | 25       |      |       |    |      | 0.9988   |          |             |
| lstm_24.1 | 13       | 7541 | 13222 | 8  | 5    | 0.9994   | 0.9991   |             |
| lstm_24.2 | 22       | 7539 | 13215 | 15 | 7    | 0.9989   | 0.9985   |             |
| lstm_25   | 15       | 7537 | 13224 | 6  | 9    | 0.9993   | 0.9990   |             |
| lstm_26   | 18       | 7541 | 13217 | 13 | 5    | 0.9991   | 0.9988   |             |
| fast_3    | 1096     | 6458 | 13222 | 8  | 1088 | 0.9218   | 0.9218   | Batch issue |
| bert_2    | 5        |      |       |    |      | 0.9998   | 0.9998   |             |
| bert_4    | 2        | 7545 | 13229 | 1  | 1    | 0.9999   | 0.9999   |             |
| bert_5    | 2        | 7545 | 13299 | 1  | 1    | 0.9999   | 0.9999   |             |



## LSTM architecture
Major Changes:
* From model 12: Bewertung anhand des F1-Scores, statt Accuracy
* From model 14: Character Level
* From model 16: Four (mistakenly) missing speeches in the test data annotated
* From model 16: Random masking of 25 % of the characters in the string
* From model 23: Random masking of 40 % of the characters in the string
* From model 24: Dropout 0.2, 2 layer added, 30 % masking of the characters in the string
* From model 25: Dropout 0.3, 20 % masking of the characters in the string
* From model 26: Dropout 0.1, 30 % masking of the characters in the string

| #    | ch_level | hidden | embedding | unk | lr   | batch | max epochs | used epoch | accuracy | f1_score |
|------|----------|--------|-----------|-----|------|-------|------------|------------|----------|----------|
| 1.1  |          | 64     | 50        | 1   | 0.05 | 10    | 5          | 3          | 0.9990   |          |
| 1.2  |          | 64     | 50        | 1   | 0.05 | 10    | 5          | 3          | 0.9981   |          |
| 2    |          | 100    | 50        | 1   | 0.05 | 50    | 5          | 3          | 0.9981   |          |
| 3.1  |          | 100    | 50        | 1   | 0.05 | 10    | 5          | 0          | 0.9976   |          |
| 3.2  |          | 100    | 50        | 1   | 0.05 | 10    | 5          | 3          | 0.9986   |          |
| 4    |          | 100    | 50        | 1   | 0.05 | 10    | 5          | 4          | 0.9981   |          |
| 5.1  |          | 64     | 100       | 1   | 0.05 | 10    | 5          | 3          | 0.9990   |          |
| 5.2  |          | 64     | 100       | 1   | 0.05 | 10    | 10         | 5          | 0.9981   |          |
| 6.1  |          | 64     | 200       | 1   | 0.05 | 10    | 7          | 4          | 1.0      |          |
| 6.2  |          | 64     | 200       | 1   | 0.05 | 10    | 7          | 1          | 0.9952   |          |
| 6.3  |          | 64     | 200       | 1   | 0.05 | 10    | 7          | 5          | 0.9981   |          |
| 7    |          | 64     | 400       | 1   | 0.05 | 10    | 7          | 0          | 0.9971   |          |
| 8    |          | 64     | 300       | 1   | 0.05 | 10    | 7          | 3          | 0.9986   |          |
| 9.1  |          | 64     | 200       | 3   | 0.05 | 10    | 7          | 3          | 0.9961   |          |
| 9.2  |          | 64     | 200       | 3   | 0.05 | 10    | 7          | 1          | 0.9966   |          |
| 9.3  |          | 64     | 200       | 3   | 0.05 | 10    | 7          | 2          | 0.9990   |          |
| 10.1 |          | 50     | 200       | 1   | 0.05 | 10    | 7          | 0          | 0.9995   |          |
| 10.2 |          | 50     | 200       | 1   | 0.05 | 10    | 7          | 0          | 1.0      | 1.0      |
| 11.1 |          | 50     | 200       | 1   | 0.04 | 10    | 7          | 2          | 0.9995   |          |
| 11.2 |          | 50     | 200       | 1   | 0.04 | 10    | 7          | 2          | 0.9966   |          |
| 12   |          | 50     | 200       | 10  | 0.048| 10    | 7          | 6          | 0.9061   | 0.8157   |
| 13   |          | 50     | 500       | 10  | 0.05 | 10    | 10         | 2          |          | 0.8674   |
| 14   | X        | 50     | 500       | 1   | 0.05 | 10    | 10         | 3          |          | 0.9974   | 
| 15   | X        | 50     | 50        | 1   | 0.05 | 10    | 10         | 4          |          | 0.998    |
| 16   | X        | 50     | 50        | 1   | 0.05 | 10    | 5          | 4          |          | 0.9901   |
| 17.1 | X        | 50     | 20        | 1   | 0.05 | 10    | 5          | 4          | 1.0      | 1.0      |
| 17.2 | X        | 50     | 20        | 1   | 0.05 | 10    | 5          | 3          |          | 0.9954   |
| 17.3 | X        | 50     | 20        | 1   | 0.05 | 10    | 5          | 4          |          | 0.9987   |
| 18   | X        | 100    | 20        | 1   | 0.05 | 10    | 5          | 4          |          | 0.998    |
| 19   | X        | 20     | 20        | 1   | 0.05 | 10    | 5          | 1          |          | 0.9968   |
| 20   | X        | 40     | 15        | 1   | 0.05 | 10    | 5          | 2          | 0.9995   | 0.9993   |
| 21   | X        | 50     | 15        | 1   | 0.05 | 10    | 5          | 2          |          | 0.9966   |
| 22   | X        | 50     | 25        | 1   | 0.05 | 10    | 5          | 4          |          | 0.9987   |
| 23   | X        | 50     | 20        | 1   | 0.05 | 10    | 5          | 2          |          | 0.9949   |
| 24.1 | X        | 50     | 20        | 1   | 0.05 | 10    | 8          | 7          | 0.9990   | 0.9987   |
| 24.2 | X        | 50     | 20        | 1   | 0.05 | 10    | 8          | 7          | 0.9995   | 0.9993   |
| 25   | X        | 50     | 20        | 1   | 0.05 | 10    | 8          | 7          |          | 0.9975   |
| 26   | X        | 50     | 20        | 1   | 0.05 | 10    | 8          | 7          |          | 0.9987   |


## LSTM with FastText
| #   | hidden | lr   | batch | max epochs | used epoch | f1_score |
|-----|--------|------|-------|------------|------------|----------|
| 1   | 50     | 0.05 | 10    | 5          | 3          | 0.9955   |
| 2.1 | 100    | 0.05 | 10    | 5          | 4          | 0.9964   |
| 2.2 | 100    | 0.05 | 10    | 5          | 2          | 0.987    |
| 2.3 | 100    | 0.05 | 10    | 5          | 1          | 0.9974   |
| 3   | 150    | 0.04 | 10    | 5          | 0          | 0.9953   |
| 4.1 | 70     | 0.05 | 10    | 5          | 2          | 0.9928   |
| 4.2 | 70     | 0.05 | 10    | 5          | 3          | 0.994    |