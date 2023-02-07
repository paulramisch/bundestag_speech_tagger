# bundestag_speech_tagger
The Bundestag Speech Tagger is a group of machine learning (ML) models trained to classify the beginning of speeches in Plenary protocols of the German Bundestag. In the following the need, development process and evualation are documented.

# 1. Introduction
The project [Open Discourse](https://opendiscourse.de/) (OP) offers a database of all the speeches that were held in the German Parliament, the Bundestag.[[1]](#ref_1) To accomplish this, the research team used OCR- and/or PDF-extracted texts of the protocols and cut them into the single speeches by using Regex-based heuristics. These complex heuristics cover the majority of the cases but approximately 3 % of the original speeches are missing. Furthermore, there are speeches in Open Discourse corpora, that are in fact not speeches.[[2]](#ref_2)

The heuristics are based on a few lines preceding the speech with the name, party and function of the speaker, three complex Regular Expression (RegEx) patterns where written for this.[[3]](#ref_3) While the RegEx pattern are public, there is no documentation of their development. The approach of using RegEx to structure unstructered (plenary) documents is very common, and was also used for the similar GermaParl corpus.[[4]](#ref_4)

If these "meta information" about a speech are found, the system sees this as a start of a new speech, and cuts of the preceding speech. So a mistake here always also affects the previous speech as well. In the recent preprint paper for the corpus the team states two known issues for the electoral terms 1, 2, 19 and 20.[[5]](#ref_5) The issues are unfortuneatly wider spread and cover all the electoral terms in the OP corpus.[[6]](#ref_6)

It is important to note that this task is very complex as the structure of these lines and their information changed a lot over the 70 years the corpus mirrors. Adding to that there are also typing and OCR errors and RegEx patterns are very sensible and only match exactly the exact pattern.[[7]](#ref_7)

This work tries to get better results than a heuristic by using Natural Language Processing techniques for the classification of possible speech starts. Many machine learning task where exactly developed for this use and the use of ML models to structure XML files isn't a common resarch trope. This work is insofar special at it aims not for general use but for near-perfect results on a specific dataset and to overtake heuristic approaches that where developed with great care. Both to reduce the effort needed for this task while increasing the quality of the classifications.

# 2. Approach & preparation
The approach for contains two steps: A very basic heuristic that identifies and extracts (multi)lines that end or contain a colon, as a the start of speech contains a colon. These extracted strings then are feed for classification to a machine learning model. The machine learning models are based on common architectures in Natural Language Processing (NLP): LSTM, LSTM with FastText Embeddings, BERT.

In this approach we only classificate short strings, a possible alternative would be, to use the whole text body and train it to insert tags where a new speech starts. While this is a viable approach, the classification of single lines is a lot less ressource intensive for the training as the content is prefiltered and concentrated to less than 5% of the original character count. 

More importantly the LSTM architecture would need a sentence level encoding, instead of word or character level encoding. First tests showed, that files of up to 1 or 2mb and hundreds of thousands tokens interfere with the architectures capabilities. LSTMs can keep track of over 1.000 timesteps which represent tokens/words here.[[8]](#ref_8) This is not enough to gain an advantage of the taken prefilter approach but the added token act as noise and have a huge negative impact on the systems performance, however this might not be true for transformer networks - this hasn't been tested in this work. Sentence level encoding on the other hand leads to huge loss of informartion towards certain token that are needed for this classificiation such as commas, dots and colons.

## 2.1 Training data preparation
The first step is the data preparation: In order to train the model there is training data needed. For this 36 out of the approximately 5000 protocols were annotated, containing over 7000 speeches. Departing from the modeling approach in the modern XML files that are published by the German parliamentary since 2017, every time a new person speaks, it is annoated as a new speech, this mirrors the modelling of the OP data. In the Bundestag XML files, the speeches of the speaker are modelled as of the speech of the politicans. The data 36 where worked through twice and automaticly checked for plausibility.[[9]](#ref_9)

The training data is created by extracting all strings of up 150 characters (all possible characters, including whitespace, line breaks and numbers) that with a colon with this RegEx pattern: `^.{5,150}?:`. If such a string is directly following the annotated speech tag, it is considered a positive, that is a speech beginning, if not it is tagged as negative, which means not a speech start but just speech content.

Naturally the negatives are just speech content that include a colon. This also leads to a blind spot of this approach: If the colon is missing, either due to OCR erors or another mistake, the classification won't take place which would lead to a non-detected speech. However, during the annotation process, there was not a single case of this found.

In the iterating development process of models in the LSTM architecture, I realised by manually checking the list of mistakes of the models, that there where four missing speeches in the test data. These where annotated and became the foundation for the further training of the LSTM models 16 to 26, all LSTM FastText and BERT models.

## 2.2 Measure: Accuracy vs. F1-Score
The are different possible metrics to measure the perfomance for this kind of task. The easiest one to compute is accuracy, that show that considers the proportion of correct predictions out of all predictions made. For the fest few iterations of the the LSTM model, when the development process was very exploritory this metric was used to determine the best performing model epoch in training and to validate the models itself.

The very good results of this process showed the need for a better measure of performance, for two reasons the first being practical: As the Open Discourse Corpus works completely different, the accuracy can't be computed as the input data ist not prefiltered as this one and is the complete raw test. The "negatives" in the training data don't exist for the OP-System, which is why a comparable accuracy cannot be calculated:
```
accuracy = true_predictions / (true_predictions + false_predictions)
```

The other reason is that the accuracy gives little weight to false_negatives (i.e. instances of speeches that were not identified by the model). The issue lies with unbalanced data, where one kind of outcome is more common than the other, in the data we have 7000 strings that are the beginning pf speeches but over 14.000 that are not. A common example to illustrate this, is a hypothetical cancer test, were 300 people get tested, 270 do not have cancer and get the right negative (true negative) result - "no cancer", but 30 people with cancer the wrong negative result (false negative). However this not working test, that always predicts "no cancer", has an accuracy of 90 %.[[10](#ref_10)]

The F1-Score is a better metric here, as it takes into account both precision and recall. Precision is the proportion of true positive predictions (i.e. correctly identifying a speech string) out of all positive predictions made by the model. Recall is the proportion of true positive predictions out of all actual instances.[[11](#ref_11)]

Implemented these look therefore like this:
```python
f1_score = (2 * true_positive / (2 * true_positive + false_positive + false_negative))
accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive)
```

The F1-Score for the Open Discourse Corpus is therefore 0.9811:
2 * 7309 / (2* 7309 + 233 + 48) = 0.9811

## 2.3 Comparison
The usual comparision of Machine Learning models is based on the training and test or the training, test and validation split. For the LSTM it is the later: 80 % are used for the training, 10 % for the test - which of the epochs model is the best, and 10 % for the validation. So the accuracy or f1 score are tested on the validation data, 10 % of the whole dataset of 20775 enties. In each training the data gets randomly mixed.

Under normal circumstances this gives a great measure to compare different architectures, however here we need to anticipate some of the results from the latter chapters: The models are too good and to close for these measures to give a clear result. 10 % of the whole datset means 2078 data points, however the best models ranges between no to three mistakes on the test data. With the random split these results are therefore although depended wether difficult cases become part of the training or test data.

While this could be adressed by not having a random split but seeding the randomness and therefore preventing this issuee. However while this makes them better comparable, it although manifest that specific split. Certain harder edge cases are just represented in that specific split.

The way this is dealt with, is therefore a combination of comparing the performance on the test data and although on the whole dataset. This is normally not recommended, as it distorts the results towards models, that memorize the data. In this case this issue already exits insofar as all the examples are derived from 36 documents. Often the speakers have multiple small speeches, for example during a question session, then the systems already have seen the data.

The good quantitative results therefore lead to a careful consideration between the validation data results and the results on the whole dataset - data of which the models have seen 80 % during training.

# 3. Machine Learning Models
Here three different kinds of architecture approaches are being compared: LSTM, LSTM + Fasttext embeddings and BERT.
While the first are both based on the Long short-term memory (LSTM) architecture, the second used pretrained embeddings to encode the text. BERT is short for Bidirectional Encoder Representations is a transformer-based machine learning technique and today widely used for NLP tasks.

The evolutionary step between LSTM and BERT are LSTM models with the Attention mechanism but is omitted here due to limited scope of this study.

## 3.1 LSTM architecture
LSTM (Long Short-Term Memory) is a type of Recurrent Neural Network (RNN) architecture was is specifically developed to overcome the vanishing gradient problem in traditional RNNs by allowing the network to selectively retain information from previous time steps. In the context of NLP, the vanishing gradient problem can pose a significant challenge for traditional Recurrent Neural Networks (RNNs) in processing long sequences of text data.[[12]](#ref_12) As the gradients become very small during the error backpropagation process, it becomes difficult for the network to effectively learn from the long-term dependencies in the text data. LSTMs are comprised of memory cells, gates, and activation functions that help regulate the flow of information into and out of the cells.[[13]](#ref_13)

The input strings from the speeches go through an embedding layer that is used to convert the input text data into word representations in the form of vector embeddings, but it is not pre-trained on external data. Instead, the embeddings are learned from scratch during training, using the input text data to update the model parameters.

During the iterative development of the 26 different combinations of parameters where tested and 37 models trained. There where a few major changes that had a strong impact on the performance, an overview over the models can be found in [Appendix 5.1](#appendix_1). 

The measure was exchanged from accuracy to F1 score for model 12 because of the considerations from chapter 2.2. Furthermore in the evaluation of model 15, the four manually  annotated speeches that were wrong, were corrected, as noted in chapter 2.1. This also means, that the results of the models 1-15 need to be cosidered carefully.

Another major shift was from token/word level to character level. While the results seemed really good on the test data a validation on the whole dataset - so although data it had seen during training, revealed, that the system just memorized the speaker names, but did not learn the structure. So it performs poorly on data is hasn't seen. This problem has become much smaller on character level.

To further facilitate this, a random masking of characters was added: The masked characters are exchanged against an `<mask>`-Token. The best result was yieled by masking 30 % of the characters. 

A dropout was added in the 24 model: Dropout refers to a regularization technique used to prevent overfitting. It works by randomly "dropping out" or ignoring a certain percentage of neurons during each iteration of training, meaning their activations and gradients will not be updated. This helps to prevent complex co-adaptations on training data, leading to a more general model that can perform well on unseen data.[[14]](#ref_14)

Up until the 24th model the LSTM was single layered, then a second layer was added, that is feed with learned representations from the first layer. 

The combination of a dropout of 20% and the second LSTM layer led to the best results on the whole dataset and the second best (comparable) results on the dataset itself. The much eassier model 17 without a second layer and dropout performend perfect on the test data (no mistakes), instead of 2 or 1 mistakes as the two model 24 iterations, however it performs worse on the full dataset with 25 mistakes compared to 13 and 22 for model 24_1 and 24_2.


## 3.2 LSTM architecture with Fasttext embeddings
The next architecture is the same, however the Embeddings layer is different, instead of the limited representations learned from the input data it uses a pretrained Embedding layer, with Facebooks FastText.[[15]](#ref_15) While FastText is often used for the embedding through a bag of words approach, here just the word embeddings are used. These embeddings capture the semantic meaning of the words and are fed as input to the LSTM network. The use of pre-trained FastText embeddings helps the network overcome the challenge of learning meaningful representations from scratch, especially when dealing with large amounts of text data. 

There are other pretrained embeddings, e.g. based on Googles Word2Vec algorithm. However the Fasttext model is trained on a broader dataset than the most common German Word2Vec set.[[16]](#ref_16)

However as shown in the after iterating over the parameters for [four models and seven training iterations](#appendix_2) the results are objectivly very good, with F1 scores of up to 0.9974 on the validation data but still not as good as the LSTM performance.

Running the Fasttext LSTM with the test data showed very poor performance, however this is due to batching issues: As the data is trained in batches, using it on single instances produces wrong results because of impact of the padding of short sequences. There are a number of ways to fix this, however due to the more promosing results of the LSTM this was not followed through. Another reason for that is that the Fasttext model takes up 7.2 gb of disk space.

Thus the character level LSTM does not only peform better, but although is much more efficient in terms of memory and power usage.

## 3.3 Pretrained Bert model

https://arxiv.org/abs/1810.04805


14150: "Günther Friedrich Nolting (F.D.P.) (von Abgeordne-\nten der F.D.P. mit Beifall begrüßt):"
09126: "Häfele, zur Konzeption:"

## 3.4 Model comparison
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


# 4. Conclusion
The models 


# References
<a name="ref_1"></a>[1] Open Discourse, https://opendiscourse.de/ (checked: 27.12.2023).

<a name="ref_2"></a>[2] Paul Ramisch, Open Discourse - eine Quellenkritik, 2022, https://paulramisch.de/opendiscourse/6_analyse%2Bevaluation.html (checked: 02.02.2023).

<a name="ref_3"></a>[3] 01_extract_speeches.py, Github, https://github.com/open-discourse/open-discourse/blob/03225c25c451b8331a3dcd25937accc70c44d9ad/python/src/od_lib/04_speech_content/01_extract_speeches.py#L16 (abgerufen: 02.02.23), Zeile 16-20.

<a name="ref_4"></a>[4] Andreas Blätte, Julia Rakers, Christoph Leonhardt, How GermaParl Evolves: Improving Data Quality by Reproducible, in: Corpus Preparation and User Involvement In Proceedings of the Workshop ParlaCLARIN III within the 13th Language Resources and Evaluation Conference 2022, https://aclanthology.org/2022.parlaclarin-1.2.pdf (checked: 02.02.2023), p. 11.

<a name="ref_5"></a>[5] Florian Richter, et al., Open Discourse: Towards the First Fully Comprehensive and Annotated Corpus of the Parliamentary Protocols of the German Bundestag, SocArXiv Preprint 2023, DOI: https://doi.org/10.31235/osf.io/dx87u (checked 02.02.23), p. 10.

<a name="ref_6"></a>[6] Ramisch, Open Discourse.

<a name="ref_7"></a>[7] Richter, Open Discourse, p. 10.

<a name="ref_8"></a>[8] Ralf C. Staudemeyer, Eric Rothstein Morris, Understanding LSTM - a tutorial into Long Short-Term Memory Recurrent Neural Networks, in CoRR 2019, https://arxiv.org/abs/1909.09586 (checked: 07.02.23), p. 2.

<a name="ref_9"></a>[9] The whole process is fully documented in my digital source critique of Open Discourse:
Ramisch, Open Discourse, chapter 4.2 Tagging der Redebeiträge, https://paulramisch.de/opendiscourse/4_vergleichskorpus.html.

<a name="ref_10"></a>[10] Aakash Bindal, Measuring just Accuracy is not enough in machine learning, A better technique is required.., Techspace 2019, 
https://medium.com/techspace-usict/measuring-just-accuracy-is-not-enough-in-machine-learning-a-better-technique-is-required-e7199ac36856 (checked: 05.12.2022).

<a name="ref_11"></a>[11] Sensitivity and specificity, Wikipedia, https://en.wikipedia.org/wiki/Sensitivity_and_specificity (checked: 05.12.2022).

<a name="ref_12"></a>[12] Staudemeyer, Rothstein Morris, Understanding LSTM, p. 2. 

<a name="ref_13"></a>[13] Staudemeyer, Rothstein Morris, Understanding LSTM, p. 19-20. 

<a name="ref_14"></a>[14] Geoffrey E. Hinton, et al., Improving neural networks by preventing co-adaptation of feature detectors, in: CoRR 2012, https://arxiv.org/abs/1207.0580 (checked: 07.02.23), p. 1.

<a name="ref_15"></a>[15] Armand Joulin, et al., Bag of Tricks for Efficient Text Classification, in: CoRR 2016, https://arxiv.org/abs/1607.01759 (checked: 07.02.23), p. 1-2.

<a name="ref_16"></a>[16] Word vectors for 157 languages, Fasttext 2018, https://fasttext.cc/docs/en/crawl-vectors.html (checked: 07.02.23).

Andreas Müller, GermanWordEmbeddings, GitHub 2022, https://github.com/devmount/GermanWordEmbeddings (checked: 07.02.23).

# 5. Appendix
<a name="appendix_1"></a>[13] ## 5.1 LSTM architecture
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


<a name="appendix_2"></a>## 5.2 LSTM with FastText
| #   | hidden | lr   | batch | max epochs | used epoch | f1_score |
|-----|--------|------|-------|------------|------------|----------|
| 1   | 50     | 0.05 | 10    | 5          | 3          | 0.9955   |
| 2.1 | 100    | 0.05 | 10    | 5          | 4          | 0.9964   |
| 2.2 | 100    | 0.05 | 10    | 5          | 2          | 0.987    |
| 2.3 | 100    | 0.05 | 10    | 5          | 1          | 0.9974   |
| 3   | 150    | 0.04 | 10    | 5          | 0          | 0.9953   |
| 4.1 | 70     | 0.05 | 10    | 5          | 2          | 0.9928   |
| 4.2 | 70     | 0.05 | 10    | 5          | 3          | 0.994    |

<a name="appendix_3"></a>## 5.3 BERT
| #   | used epoch | f1_score  |
|-----|------------|-----------|
| 2   | 2          | 0.9968    |
| 4   | 4          | 0.9987    |
| 5   | 5          | 0.9987    |

## 5.3 Validation on whole dataset
| model     | mistakes | tp   | tn    | fp | fn   | accuracy | f1_score | comment     |
|-----------|----------|------|-------|----|------|----------|----------|-------------|
| lstm_10.2 | 992      |      |       |    |      | 0.9523   |          | Batch issue |
| lstm_12   | 2092     |      |       |    |      | 0.8993   |          | Batch issue |
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
