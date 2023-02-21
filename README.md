# Bundestag Speech Tager: Comparison of Different Machine Learning Architectures to Structure Plenary Minutes

This repository contains an evaluation of different machine learning (ML) architectures and models trained to find speeches in the unstructured plenary minutes files of the German Bundestag, and compares their performance with the regex based approach used for the [Open Discourse Corpus](https://opendiscourse.de/).

The code for training the models can be found in the respective folders `lstm_model`, `lstm_model_fasttext`, `bert_model`, they each contain a trained, ready-to-use model, except for the BERT architecture due to the size limitations of the GitHub hosting. The `open_discourse` folder contains the Open Discourse regex heuristics adapted to work with the training data used for the ML training for comparison. The `data` folder contains the annotated transcripts in XML format, a script to generate training data, and the training data in CSV and Pickle.

---

# 1. Introduction

The project [Open Discourse](https://opendiscourse.de/) (OP) offers a database of all speeches held in the German parliament, the Bundestag.[^1] To achieve this, the research team used OCR- and/or PDF-extracted texts of the transcripts and cut them into individual speeches using regex-based heuristics. These complex heuristics cover the majority of cases, but about 3 % of the original speeches are missing. Furthermore, there are speeches in the Open Discourse corpora that are not really speeches.[^2]

The heuristics are based on a short introduction before the speech with the speaker's name, party and function, for which three complex regular expression (regex) patterns were written.[^3] Although the regex patterns are public, there is no documentation of their development. The approach of using regex to structure unstructured (plenary) documents is very common, and was also used for the similar GermaParl corpus.[^4]

If this introduction is found, the system sees it as the beginning of a new speech and cuts the previous speech. So an error here always affects the previous speech too, because it means that the previous speech now contains the content of the missed speech. In the recent preprint paper for the corpus, the team identifies two known problems for election terms 1, 2, 19 and 20.[^5] Unfortunately, the problems are more widespread, affecting all election terms in the OP corpus.[^6]

It is important to note that this task is very complex, as the structure of these lines and the information they contain has changed slightly over the 70 years that the corpus reflects. There are also typing and OCR errors, to which regex patterns are very sensitive as they only match the exact pattern.[^7]

This work tries to find out if it is possible to get better results than a heuristic by using natural language processing techniques to structure plenary minutes by classifying possible speech beginnings. Many machine learning tasks have been developed for precisely this purpose, and the use of ML models to structure XML files is not an uncommon research trope.[^8] This work is unique in that it's not intended for general use, but to achieve near perfect results on a specific dataset, outperforming heuristic approaches that have been developed with great care. Both to reduce the effort required for this task and to increase the quality of the classifications.

# 2. Approach & Preparation

The approach consists of two steps: A very simple regex heuristic that identifies and extracts (multi)lines that end with or contain a colon, since the beginning of speech contains a colon. These extracted strings are then fed into a machine learning model for classification. The machine learning models are based on common natural language processing (NLP) machine learning architectures: BiLSTM, BiLSTM with FastText Embeddings and BERT.

In this approach we only classify short strings, a possible alternative would be to use the whole text body and train it to insert tags where a new utterance starts. While this is a viable approach, the classification of single lines is much less resource intensive for training, as the content is pre-filtered and concentrated to less than 5% of the original character count.

More importantly, the LSTM architecture would require sentence-level encoding rather than word- or character-level encoding. Initial tests showed that files of up to 1 or 2 MB and hundreds of thousands of tokens would overwhelm the architecture's capabilities: LSTMs can keep track of over 1,000 tokens, most BERT models can take up to 512 tokens as input.[^9] While BERT doesn't work at all with sentence-level encoding, for LSTMs early tests showed that the sentence-level abstraction lost too much information about certain characters such as commas, dots and colons, and performance was very poor.

## 2.1 Minutes & Open Discourse

The proceedings of the German Bundestag are open to the public; only by a two-thirds majority can the public be excluded.[^10] Plenary minutes are taken for each session, and are prepared by the Bundestag's own stenographic service.[^11] The basic structure has not changed since 1949 and is in the direct tradition of the minutes of the Weimar Republic:

- Table of contents: List of agenda items
- Main body
- Annexes to the stenographic report: The first appendix is the attendance list or the list of excused delegates; other appendices are, for example, written replies to questions.

On its [Open Data Portal](https://www.bundestag.de/services/opendata), the German Bundestag makes available all the Minutes since 1949 in PDF and XML format, but the files until 2017 they are completely unstructured and contain only some meta information. This completely unstructured text does not contain any delimitation of the various components, such as the table of contents, the main body or the individual agenda items and speeches. Until 1997, the documents are also based on scans that have been OCR'ed, after which they are based on born-digital PDF documents.

Open Discourse is a research project by Berlin-based data science company Limebit GmbH, and aims to remove the biggest limitation of existing Bundestag debates: The inability to filter speeches by politician, party or position. The dataset, which was released in December 2020, appears to be a highly valuable resource for historical, political and linguistic research on the Federal Republic of Germany. The code has been released under the MIT licence for free reuse, and the data has been released under CC0 1.0, i.e. without any restrictions.[^12]

The process of creating the database is complex and involves a large number of steps, such as structuring the transcripts into a table of contents, a body and a conclusion, breaking down the body into individual speeches and assigning the speeches to the politicians. In between, various cleaning steps are necessary, for example to filter out layout artefacts. Most of the errors occur in the step of splitting the main part into the different speeches.

This step is mainly based on three regex patterns:[^13]
```python
president_pattern_str = r"(?P<position_raw>Präsident(?:in)?|Vizepräsident(?:in)?|Alterspräsident(?:in)?|Bundespräsident(?:in)?|Bundeskanzler(?:in)?)\s+(?P<name_raw>[A-ZÄÖÜß](?:[^:([}{\]\)\s]+\s?){1,5})\s?:\s?"

faction_speaker_pattern_str = r"{3}(?P<name_raw>[A-ZÄÖÜß][^:([{{}}\]\)\n]+?)(\s*{0}(?P<constituency>[^:(){{}}[\]\n]+){1})*\s*{0}(?P<position_raw>{2}){1}(\s*{0}(?P<constituency>[^:(){{}}[\]\n]+){1})*\s?:\s?"

minister_pattern_str = r"{0}(?P<name_raw>[A-ZÄÖÜß](?:[^:([{{}}\]\)\s]+\s?){{1,5}}?),\s?(?P<position_raw>(?P<short_position>Bundesminister(?:in)?|Staatsminister(?:in)?|(?:Parl\s?\.\s)?Staatssekretär(?:in)?|Präsident(?:in)?|Bundeskanzler(?:in)?|Schriftführer(?:in)?|Senator(?:in)?\s?(?:{1}(?P<constituency>[^:([{{}}\]\)\s]+){2})?|Berichterstatter(?:in)?)\s?([^:([\]{{}}\)\n]{{0,76}}?\n?){{1,2}})\s?:\s?"
```

In a sample of 36 out of approximately 5,000 transcripts, 233 of the 7542 transcripts were not recognised and 48 speeches were recognised that were not speeches because these patterns do not catch every case. As the data has been structured since 2017 and the OP corpus has not been based on regex patterns since then, this study focuses on transcripts from 1949-2017.[^14]

## 2.2 Preparing the training data

The first step is data preparation: Training data are needed to train the model. For this purpose, 36 of the approximately 5,000 transcripts were annotated, resulting in more than 7,000 annotated speeches, two for each of the Bundestag periods 1-18. Unlike the modelling approach in the modern XML files published by the Bundestag since 2017, every time a new person speaks, it is annotated as a new speech, reflecting the modelling of the OP data. In the Bundestag XML files, the speeches of the President of the Bundestag, who acts as speaker, are modelled as part of the politicians' speeches. The data was processed twice and automatically checked for plausibility.[^15]

The training data is created by extracting all strings of up to 150 characters (all possible characters, including spaces, line breaks and numbers) that end with a colon with the following regex pattern: `^.{5,150}?:`. If such a string comes directly after the annotated speech tag, it will be considered positive, i.e. a beginning of an speech; if not, it will be considered negative, i.e. no beginning of an speech, just content of an speech. A few examples look like this:

> Dr. Wilhelmi (CDU/CSU) :

> Präsident Dr. Norbert Lammert:

> Genscher, Bundesminister des Innern:

> Heide Simonis, Ministerpräsidentin (Schleswig-Hol-
> stein) (von der SPD mit Beifall begrüßt):

Of course, the negatives are just speech content that includes a colon. This also leads to a blind spot in this approach: If the colon is missing, either due to OCR errors or some other mistake, no classification takes place, which would result in an unrecognised speech. During the annotation process, however, not a single case of this was found. Here are some of these non-speech strings:

> Unterhaltungsmaßnahmen. Dabei verkennen wir nicht
> die Zuständigkeiten der Länder für den Hochwasser-
> schutz. Ich sage aber auch:

> 18. April des letzten Jahres – das ist noch kein Jahr her –
> wird Herr Scharping mit folgender Aussage zitiert:

> (Dr. Peter Ramsauer [CDU/CSU]:

> (Georg Pfannenstein [SPD]:

The last two are examples of interjections in speeches, which look similar to a speech, but use different brackets and have a round bracket at the beginning, although the bracket may be on a different line.

During the iterative development process of the BiLSTM models, by manually checking the error list of the models, I realised that there were four missing speeches in the test data. These were annotated and became the basis for further training of LSTM models 16 to 26, all LSTM FastText and BERT models.

For validation purposes, a second set was created from this data, in which noise was added: 10% of the a-z/A-Z characters in each string were randomly replaced with a different a-z/A-Z character.

## 2.3 Measure: Accuracy vs. F1 Score

There are several possible measures of performance for this type of task. The easiest to calculate is accuracy, which takes into account the proportion of correct predictions out of all predictions made. For the first few iterations of the LSTM model, when the development process was very exploratory, this metric was used to determine the best performing model epoch in training and to validate the models themselves.

The very good results of this process showed the need for a better measure of performance, for two reasons, the first of which is practical: Since the Open Discourse Corpus works in a completely different way, the accuracy can't be calculated, because the input data is not pre-filtered and consists of the complete raw data. The "negatives" in the training data don't exist for the OP system, so no comparable accuracy can be calculated:
```
accuracy = true_predictions / (true_predictions + false_predictions)
```

The other reason is that accuracy gives little weight to false_negatives (i.e. instances of speeches not identified by the model). The problem is with unbalanced data, where one type of outcome is more common than the other; in the data we have 7,000 strings that are the beginning of speeches, but over 14,000 that are not. A common example to illustrate this is a hypothetical cancer test where 300 people are tested, 270 do not have cancer and get the right negative (true negative) result - "no cancer", but 30 people with cancer get the wrong negative (false negative) result. However, this dysfunctional test, which always predicts "no cancer", has an accuracy of 90%.[^16]

The F1 score is a better metric because it takes into account both precision and recall. Precision is the proportion of true positive predictions (i.e. correctly identifying a string of speech) out of all positive predictions made by the model. Recall is the proportion of true positive predictions out of all actual instances[^17].

Implemented, this looks like this:
```python
f1_score = (2 * true_positive / (2 * true_positive + false_positive + false_negative))
accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive)
```

The F1 score for the Open Discourse Corpus is therefore 0.9811:  
`2 * 7309 / (2* 7309 + 233 + 48) = 0.9811`

## 2.4 Comparison Methodology

When training and comparing machine learning models, the data is usually split into either training and test data, or training, test and validation data. For LSTM it is the latter: 80% is used for training, 10% for testing - which of the epochs is the best model - and 10% for validation. So the accuracy or F1 score is tested on the validation data, 10% of the whole dataset of 20,775 entries. At the beginning of the training, the data is randomly mixed.

Normally this would be a good measure to compare different architectures, but here we need to anticipate some of the results from the later chapters: The models are too good and too close for these measures to give a clear result. 10% of the total dataset means 2078 data points, but the best models range from zero to three errors on the test data. With the random split, these results are therefore dependent on whether difficult cases become part of the training or test data.

This could be addressed by not having a random split, but by seeding the randomness and thus avoiding this problem. However, while this makes them more comparable, it does manifest this particular split. However, the different models and architectures will have different edge case problems, and with such a small error rate it becomes a coin flip as to which model performs best, depending on whether their specific edge case is part of the training or validation data.

The way this is dealt with is therefore a combination of comparing performance on the test data and on the whole dataset. This is normally not recommended as it biases the results towards models that remember the data. In this case, this problem is already present to some extent, as all the examples are taken from the same 36 documents. Often speakers have several short speeches, for example during a Q&A session, by which time the systems have already seen the data.

The good quantitative results therefore lead to the need for careful consideration of the results from the validation data and the results from the full dataset - data of which the models have seen 80% during training. In addition, a comparison of performance on the noisy dataset will be part of this consideration.

# 3. Machine Learning Architectures

Three different types of architectures are compared here: BiLSTM, BiLSTM + Fasttext embeddings and BERT.
While the first two are both based on the Long Short-Term Memory (LSTM) architecture, the second uses pre-trained embeddings to encode the text. BERT, which stands for Bidirectional Encoder Representations, is a transformer-based machine learning technique that is now widely used for NLP tasks, the most famous being OpenAI's ChatGPT. These architectures can be used for a variety of different applications, here we use it for a sequence classification task. Given a sequence of inputs in the form of tokens or characters, the result is a binary classification as true or false, answering whether the sequence represents speech according to the model.

The evolutionary step between LSTM and BERT are LSTM models with the Attention mechanism, but are omitted here due to the limited scope of this study.

## 3.1 BiLSTM

LSTM (Long Short-Term Memory) is a type of Recurrent Neural Network (RNN) architecture was is specifically designed to overcome the vanishing gradient problem in traditional RNNs by allowing the network to selectively retain information from previous time steps. In the context of NLP, the vanishing gradient problem can be a significant challenge for traditional Recurrent Neural Networks (RNNs) when processing long sequences of text data.[^18] As the gradients become very small during the error backpropagation process, it becomes difficult for the network to learn effectively from the long-term dependencies in the text data. LSTMs are comprised of memory cells, gates, and activation functions that help regulate the flow of information into and out of the cells.[^19]

A special type of LSTM is used here: A BiLSTM (Bidirectional Long Short-Term Memory networks), a variant of LSTMs that are bidirectional, meaning they process the input sequence in both forward and backward directions, allowing them to capture both past and future context in the hidden state representation.[^20]

The input strings from the speeches go through an embedding layer that is used to convert the input text data into word representations in the form of vector embeddings, but it is not pre-trained on external data. Instead, the embeddings are learned from scratch during training, using the input text data to update the model parameters.

During the iterative development, 26 different parameter combinations were tested and 37 models were trained. There were a few major changes that had a strong impact on performance, an overview of the models can be found in [Appendix 5.1](#appendix_1).

The measure was changed from accuracy to F1 score for model 12 because of the considerations in chapter 2.2. In addition, the evaluation of model 15 corrected the four manually annotated speeches that were incorrect, as noted in chapter 2.1. This also means that the results of models 1-15 need to be considered carefully.

Another major shift was from the token/word level to the character level. The results of the previous model 10 looked impressive with an F1 score of 1.0 on the test data and an F1 score of 0.9999 on the full data set. However, after going through the list of errors it made on the full data set, it became clear that the system had only learnt the names of the speakers, but not the structure. So it performs poorly on data it hasn't seen. This problem is much smaller at character level.

To further facilitate this, a random masking of characters was added: The masked characters are exchanged for a `<mask>' token. The best result was achieved by masking 30% of the characters.

A dropout was added in the 24th model: Dropout refers to a regularisation technique used to prevent overfitting. It works by randomly 'dropping out' or ignoring a certain percentage of neurons during each training iteration, meaning that their activations and gradients are not updated. This helps to prevent complex co-adaptations to training data, leading to a more general model that can perform well on unseen data.[^21]

Up to the 24th model the LSTM was single layer, then a second layer was added, i.e. fed with learned representations from the first layer. 

The combination of a 20% dropout and the second LSTM layer produced the best results on the whole dataset and the second best (comparable) results on the dataset itself. The much simpler model 17 with no second layer and no dropout performed perfectly on the test data (no errors) instead of 2 and 1 errors as the two iterations of model 24, but performed worse on the full dataset with 25 errors compared to 13 and 22 for models 24_1 and 24_2.

## 3.2 BiLSTM & FastText

The next architecture is the same, however the embedding layer is different, instead of the limited representations learned from the input data it uses a pre-trained embedding layer,  Facebook's FastText.[^22] While FastText is often used for embedding sentences through a bag of words approach, here only the word embeddings are used. These embeddings capture the semantic meaning of the words and are fed as input to the LSTM network. The use of pre-trained FastText embeddings helps the network to overcome the challenge of learning meaningful representations from scratch, especially when dealing with large amounts of text data. 

There are other pre-trained embeddings, for example based on Google's Word2Vec algorithm. However, the Fasttext model is trained on a broader dataset than the most common German Word2Vec set.[^23]

However, as shown in the after-iteration over parameters for [four models and seven training iterations](#appendix_2), the results are objectively very good, with F1 scores up to 0.9974 on the validation data, but still not as good as the LSTM performance. On the whole dataset the best model makes 27 errors and gets an F1 score of 0.9982.

Thus, not only does the character level BiLSTM model perform better, but it is also much more efficient in terms of memory and power consumption, as the FastText embedding model required to use the trained model takes up 7.2 GB of disk space.

## 3.3 BERT

Transformers are a type of neural network architecture originally introduced in the 2017 paper 'Attention is All You Need'.[^24] Transformers are so called because they use self-attention mechanisms to transform and consider the relationships between all input elements at once. This is in contrast to recurrent neural networks such as LSTM, which process sequences one element at a time. Transformers are highly parallelizable and can be trained on large amounts of data, making them well suited to NLP tasks that require an understanding of long-range dependencies in language. In particular, BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained transformer-based language model that has been trained on a large corpus of text data, allowing it to be fine-tuned for specific NLP tasks with relatively small amounts of task-specific training data.[^25]

The implementation used is from the Python package [simple transformers](https://simpletransformers.ai/), it provides ready-made transformer architectures for a variety of applications, here we use it for binary classification.[^26] The development of this model was by far the easiest, as simple transformers does all the heavy code lifting. There are pre-trained [BERT models for German](https://huggingface.co/bert-base-german-cased), but at the time of testing the model was not compatible with the Mac M1 processor initially used for development. The first iteration used the [BERT model from the paper](https://huggingface.co/bert-base-cased) trained on English data, the second and third iterations used a German model trained on German data.[^27]

As for the two LSTM architectures, 80% of the data was shown in training, but there was no test after each epoch; for the first iteration the validation split was 10%, so 10% of the data remained unused.

The results are impressive: The model from the first iteration, pre-trained on the English data, made only two mistakes on both the test and the whole dataset. The model pre-trained on the German dataset made no mistakes on the unseen 20% of the data and a single mistake on the whole dataset. The third iteration, although showing good results, does not reach the same level as the previous two iterations, while still having all the same parameters as the second, showing the non-determinism of the learning systems.

The first model incorrectly labelled the string `Günther Friedrich Nolting (F.D.P.) (von Abgeordne- ten der F.D.P. mit Beifall begrüßt):`(protocoll 14-150) as a speech and incorrectly labelled the string `Häfele, zur Konzeption:`(protocoll 09-126) as a speech. The second German trained model incorrectly labelled the string `CDU/CSU:` as a speech. The English trained model seems to have learned the different structures and interpunctuation of real speech beginnings, which led it to reject the Nolting speech, which has a very unusual format, and to accept the Häfele string, which looks very much like the beginning of a speech when the vocabulary is unknown - as we will see in the next chapter, this is not untrue. The German model seems to work with the known tokens/words, as there are no mistakes like in the first one, but the CDU/CSU string is a big indicator for the beginning of a speech, but obviously not for the structure.

## 3.4 Model & Architecture Comparison

| model          | td_mistakes | td_accuracy | td_f1_score | fd_mistakes | fd_accuracy | fd_f1_score |
|----------------|-------------|-------------|-------------|-------------|-------------|-------------|
| lstm_10.2*     | 0           | 1.0         | 1.0         | 13          | 0.9994      | 0.9991      |
| lstm_17.1      | 0           | 1.0         | 1.0         | 25          | 0.9988      | -           |
| lstm_20        | 1           | 0.9995      | 0.9993      | 28          | 0.9987      | -           |
| lstm_24.1      | 2           | 0.9990      | 0.9987      | 13          | 0.9994      | 0.9991      |
| lstm_24.2      | 1           | 0.9995      | 0.9993      | 22          | 0.9989      | 0.9985      |
| fast_2.4       | 7           | 0.9966      | 0.9955      | 27          | 0.9987      | 0.9982      |
| bert_1_e2      | -           | -           | -           | 5           | 0.9998      | 0.9998      |
| bert_1_e5      | 2           | 0.9990      | 0.9987      | 2           | 0.9999      | 0.9999      |
| bert_2_e5      | 0           | 1.0         | 1.0         | 1           | 0.9999      | 0.9999      |
| bert_3_e5      | 0           | 1.0         | 1.0         | 1           | 0.9999      | 0.9999      |
| open_discourse | -           | -           | -           | 281         | -           | 0.9811      |

Table 1: Overview of best results for different ML architectures and Open Discourse; td is short for test_data, fd is short for full_data; *Tested but not trained on updated data

The table shows nine of the best performing models and three different architectures compared to the results of the Open Discourse Corpus Code. Every single model in this list outperforms the Open Discourse Heuristics.

While the first two LSTM models work at word level and only remember speaker names, and therefore will not maintain their test accuracy scores on the different data, the other models learn the structure of the data, but are most likely aided by the names they saw during training.

The performance of the BERT models is excellent: The two best performing models achieve an F1 score of 0.9999 on the full dataset, while both bert_2_e5 and bert_3_e5 achieve an accuracy of 1.0 with no errors on the validation data.
The performance of the bert_1_e2 and bert_2_e5 models, pre-trained on English, shows the great adaptability that makes BERT such a good architecture.

However, if we use a noisy dataset, where 10% of the a-Z characters of each string have been randomly replaced with another a-Z character, we learn a lot about the robustness of the models to errors and different data, as Table 2 shows:

| model           | td_f1_score | noisy_mistakes | noisy_f1_score | notes                                             |
|-----------------|-------------|----------------|----------------|---------------------------------------------------|
| lstm_10.2       | 1.0         | 2585           | 0.7939         | Word level embeddings                             |
| lstm_24.1       | 1.0         | 113            | 0.9925         | Character level embeddings, noise & dropout added |
| fast_2.4        | 0.9955      | 951            | 0.9331         | Word level FastText embeddings                    |
| bert_2_e5       | 1.0         | 837            | 0.9413         | English pre-trained model bert-base-cased         |
| bert_3_e5       | 1.0         | 69             | 0.9954         | German pre-trained model bert-base-german-cased   |
| open_discourse* | -           | 4733           | 0.5460         | Approximation[^28]                                |

Table 2: Performance of different models on the noisy full dataset, where 10% of the a-Z characters in each string are randomly swapped. The Open Discourse data is an approximation.

The model lstm_10.2 performs very poorly with an F1 score of 0.7939, it has its vocabulary at word level from the data and in addition it has overfitted by learning names and titles from the data, as these are now noisy this has a big impact on performance. Similarly, the noisy data has a big impact on the pre-trained FastText embeddings. Unfortunately, bert_2_e5, which is based on a pre-trained English BERT model, seems to have similar problems to a lesser extent. The approximation[^28] of the Open Discourse performance on the noisy data shows the sensitivity of regex-based heuristics to small errors in words, for example due to poor OCR or typing errors.

The best results come from the model bert_3_e5 with 69 errors and an F1 of 0.9954, the model lstm_24.1 doesn't perform too badly with 113 errors and an F1 of 0.9925. It is not too surprising that lstm_24.1 performs quite well here, as during training 30% of the strings were randomly masked, it is likely that the BERT models could further improve their performance on noisy data by adding noise during training. 

This also highlights a similarity between the lstm_24.1 model and the BERT architecture in general: They don't use whole-word embeddings, which don't work well on noisy data; this particular LSTM model works at the character level, the BERT architecture is based on flexible sub-word levels.

In general, the OCR quality of the Bundestag Corpus XML files is very good and there are a very limited number of typing errors, so the results of this test should not be taken as an absolute measure. However, it does show three things: regex heuristics are not really suited to noisy data, a good LSTM architecture can perform very well without any prior training, and thirdly, although BERT can produce incredible results, it is still important to stress test the performance of the model to find out about its weaknesses.

# 4. Conclusion

The models show that developing an ML model that outperforms complex heuristic approaches for classifying speeches from parliamentary transcripts is not only possible, but yields great results. An F1 score of 1.0 on the test data and 0.9999 on the whole dataset is far better than the Open Discourse F1 score of 0.9811.

The development of regex heuristics for this application must include a large database of annotated examples to test their performance prior to their use in production. This work shows that such annotated examples are better used directly as training data for an ML model. Computationally, regex heuristics can be much more efficient, both because they don't need training and during execution time, and the LSTM models are more efficient than the BERT models and also show great performance. However, when it comes to effectiveness, the BERT model is winnung. In the use case as a one-off annotation tool for a large dataset, these efficiency considerations pale in comparison to the great results.

For further research and development, the training and test data should come from different files, ideally by annotating a different file for each period. Another less laborious way would be to randomly swap the names and parties in the training data with a database of surnames and random parties.

Nevertheless, the results show a much better performing model than the original Open Discourse implementation. With a slight improvement of the training data and proper quality control measures, there is nothing to stop it being used in production.

With the incredible speed at which NLP systems are being developed at the moment, it is very likely that XML structuring methods will get a huge boost in the coming years. However, this should not obscure the fact that detailed quality control of such systems is necessary, as almost right is just not right. New methods of quality control will need to be developed, especially in the area of historical scholarship.

# Bibliography

# Appendix

<a name="appendix_1"></a>

## Appendix 1: LSTM architecture
Major Changes:
* From model 12: Evaluation change from accuracy to F1 scores
* From model 14: Change to character level (previouss word level)
* From model 16: Four (mistakenly) missing speeches in the test data annotated
* From model 16: Random masking of 25 % of the characters in the string
* From model 23: Random masking of 40 % of the characters in the string
* From model 24: Dropout 0.2, 2. layer added, 30 % masking of the characters in the string
* From model 25: Dropout 0.3, 20 % masking of the characters in the string
* From model 26: Dropout 0.1, 30 % masking of the characters in the string

| #    | ch_level | hidden | embedding | unk | lr   | batch | max epochs | used epoch | accuracy | f1_score |
|------|----------|--------|-----------|-----|------|-------|------------|------------|----------|----------|
| 1.1  | ❌       | 64     | 50        | 1   | 0.05 | 10    | 5          | 3          | 0.9990   | -        |
| 1.2  | ❌       | 64     | 50        | 1   | 0.05 | 10    | 5          | 3          | 0.9981   | -        |
| 2    | ❌       | 100    | 50        | 1   | 0.05 | 50    | 5          | 3          | 0.9981   | -        |
| 3.1  | ❌       | 100    | 50        | 1   | 0.05 | 10    | 5          | 0          | 0.9976   | -        |
| 3.2  | ❌       | 100    | 50        | 1   | 0.05 | 10    | 5          | 3          | 0.9986   | -        |
| 4    | ❌       | 100    | 50        | 1   | 0.05 | 10    | 5          | 4          | 0.9981   | -        |
| 5.1  | ❌       | 64     | 100       | 1   | 0.05 | 10    | 5          | 3          | 0.9990   | -        |
| 5.2  | ❌       | 64     | 100       | 1   | 0.05 | 10    | 10         | 5          | 0.9981   | -        |
| 6.1  | ❌       | 64     | 200       | 1   | 0.05 | 10    | 7          | 4          | 1.0      | -        |
| 6.2  | ❌       | 64     | 200       | 1   | 0.05 | 10    | 7          | 1          | 0.9952   | -        |
| 6.3  | ❌       | 64     | 200       | 1   | 0.05 | 10    | 7          | 5          | 0.9981   | -        |
| 7    | ❌       | 64     | 400       | 1   | 0.05 | 10    | 7          | 0          | 0.9971   | -        |
| 8    | ❌       | 64     | 300       | 1   | 0.05 | 10    | 7          | 3          | 0.9986   | -        |
| 9.1  | ❌       | 64     | 200       | 3   | 0.05 | 10    | 7          | 3          | 0.9961   | -        |
| 9.2  | ❌       | 64     | 200       | 3   | 0.05 | 10    | 7          | 1          | 0.9966   | -        |
| 9.3  | ❌       | 64     | 200       | 3   | 0.05 | 10    | 7          | 2          | 0.9990   | -        |
| 10.1 | ❌       | 50     | 200       | 1   | 0.05 | 10    | 7          | 0          | 0.9995   | -        |
| 10.2 | ❌       | 50     | 200       | 1   | 0.05 | 10    | 7          | 0          | 1.0      | 1.0      |
| 11.1 | ❌       | 50     | 200       | 1   | 0.04 | 10    | 7          | 2          | 0.9995   | -        |
| 11.2 | ❌       | 50     | 200       | 1   | 0.04 | 10    | 7          | 2          | 0.9966   | -        |
| 12   | ❌       | 50     | 200       | 10  | 0.048| 10    | 7          | 6          | 0.9061   | 0.8157   |
| 13   | ❌       | 50     | 500       | 10  | 0.05 | 10    | 10         | 2          | -        | 0.8674   |
| 14   | ✅       | 50     | 500       | 1   | 0.05 | 10    | 10         | 3          | -        | 0.9974   | 
| 15   | ✅       | 50     | 50        | 1   | 0.05 | 10    | 10         | 4          | -        | 0.998    |
| 16   | ✅       | 50     | 50        | 1   | 0.05 | 10    | 5          | 4          | -        | 0.9901   |
| 17.1 | ✅       | 50     | 20        | 1   | 0.05 | 10    | 5          | 4          | 1.0      | 1.0      |
| 17.2 | ✅       | 50     | 20        | 1   | 0.05 | 10    | 5          | 3          | -        | 0.9954   |
| 17.3 | ✅       | 50     | 20        | 1   | 0.05 | 10    | 5          | 4          | -        | 0.9987   |
| 18   | ✅       | 100    | 20        | 1   | 0.05 | 10    | 5          | 4          | -        | 0.998    |
| 19   | ✅       | 20     | 20        | 1   | 0.05 | 10    | 5          | 1          | -        | 0.9968   |
| 20   | ✅       | 40     | 15        | 1   | 0.05 | 10    | 5          | 2          | 0.9995   | 0.9993   |
| 21   | ✅       | 50     | 15        | 1   | 0.05 | 10    | 5          | 2          | -        | 0.9966   |
| 22   | ✅       | 50     | 25        | 1   | 0.05 | 10    | 5          | 4          | -        | 0.9987   |
| 23   | ✅       | 50     | 20        | 1   | 0.05 | 10    | 5          | 2          | -        | 0.9949   |
| 24.1 | ✅       | 50     | 20        | 1   | 0.05 | 10    | 8          | 7          | 0.9990   | 0.9987   |
| 24.2 | ✅       | 50     | 20        | 1   | 0.05 | 10    | 8          | 7          | 0.9995   | 0.9993   |
| 25   | ✅       | 50     | 20        | 1   | 0.05 | 10    | 8          | 7          | -        | 0.9975   |
| 26   | ✅       | 50     | 20        | 1   | 0.05 | 10    | 8          | 7          | -        | 0.9987   |

<a name="appendix_2"></a>

## Appendix 2: LSTM & FastText
| #   | hidden | lr   | batch | max epochs | used epoch | f1_score |
|-----|--------|------|-------|------------|------------|----------|
| 1   | 50     | 0.05 | 10    | 5          | 3          | 0.9955   |
| 2.1 | 100    | 0.05 | 10    | 5          | 4          | 0.9964   |
| 2.2 | 100    | 0.05 | 10    | 5          | 2          | 0.987    |
| 2.3 | 100    | 0.05 | 10    | 5          | 1          | 0.9974   |
| 2.4 | 100    | 0.05 | 10    | 5          | 1          | 0.9955   |
| 3   | 150    | 0.04 | 10    | 5          | 0          | 0.9953   |
| 4.1 | 70     | 0.05 | 10    | 5          | 2          | 0.9928   |
| 4.2 | 70     | 0.05 | 10    | 5          | 3          | 0.994    |

<a name="appendix_3"></a>

## Appendix 3: BERT
| #      | pretrained model       | used epoch | f1_score  |
|--------|------------------------|------------|-----------|
| 1_e2   | bert-base-cased        | 2          | -         |
| 1_e4   | bert-base-cased        | 4          | -         |
| 1_e5   | bert-base-cased        | 5          | 0.9987    |
| 2_e5   | bert-base-cased        | 5          | 1.0       |
| 3_e5   | bert-base-german-cased | 5          | 1.0       |
| 4_e10  | bert-base-german-cased | 10         | 0.9998    |

<a name="appendix_4"></a>

## Appendix 4: Validation on whole dataset
| model      | mistakes | tp   | tn    | fp | fn   | accuracy | f1_score |
|------------|----------|------|-------|----|------|----------|----------|
| lstm_10.2  | 13       | 7537 | 13226 | 4  | 9    | 0.9994   | 0.9991   |
| lstm_14    | 83       | -    | -     | -  | -    | 0.9960   | -        |
| lstm_15    | 16       | -    | -     | -  | -    | 0.9992   | -        |
| lstm_16    | 57       | -    | -     | -  | -    | 0.9973   | -        |
| lstm_17.1  | 25       | -    | -     | -  | -    | 0.9988   | -        |
| lstm_17.2  | 123      | -    | -     | -  | -    | 0.9941   | -        |
| lstm_17.3  | 36       | -    | -     | -  | -    | 0.9983   | -        |
| lstm_17.4  | 20       | 7534 | 13222 | 8  | 12   | 0.9990   | 0.9987   |
| lstm_18    | 42       | -    | -     | -  | -    | 0.9980   | -        |
| lstm_19    | 87       | -    | -     | -  | -    | 0.9958   | -        |
| lstm_20    | 28       | -    | -     | -  | -    | 0.9987   | -        |
| lstm_21    | 44       | -    | -     | -  | -    | 0.9979   | -        |
| lstm_22    | 43       | -    | -     | -  | -    | 0.9979   | -        |
| lstm_23    | 25       | -    | -     | -  | -    | 0.9988   | -        |
| lstm_24.1  | 13       | 7541 | 13222 | 8  | 5    | 0.9994   | 0.9991   |
| lstm_24.2  | 22       | 7539 | 13215 | 15 | 7    | 0.9989   | 0.9985   |
| lstm_25    | 15       | 7537 | 13224 | 6  | 9    | 0.9993   | 0.9990   |
| lstm_26    | 18       | 7541 | 13217 | 13 | 5    | 0.9991   | 0.9988   |
| fast_2.4   | 27       | 7539 | 13210 | 20 | 7    | 0.9987   | 0.9982   | 
| bert_1_e2  | 5        | -    | -     | -  | -    | 0.9998   | 0.9998   |
| bert_1_e4  | 2        | 7545 | 13229 | 1  | 1    | 0.9999   | 0.9999   |
| bert_1_e5  | 2        | 7545 | 13299 | 1  | 1    | 0.9999   | 0.9999   |
| bert_2_e5  | 1        | 7545 | 13230 | 0  | 1    | 0.9999   | 0.9999   |
| bert_3_e4  | 3        | 7544 | 13229 | 1  | 2    | 0.9996   | 0.9998   |
| bert_3_e5  | 1        | 7546 | 13229 | 1  | 0    | 0.9999   | 0.9999   |
| bert_4_e5  | 12       | 7545 | 13219 | 11 | 1    | 0.9995   | 0.9992   |
| bert_4_e10 | 3        | 7544 | 13229 | 1  | 2    | 0.9996   | 0.9998   |


[^1]: Open Discourse, https://opendiscourse.de/ (checked: 27.12.2023).

[^2]: Paul Ramisch, Open Discourse - eine Quellenkritik, 2022, https://paulramisch.de/opendiscourse/6_analyse%2Bevaluation.html (checked: 02.02.2023).

[^3]: 01_extract_speeches.py, Github, https://github.com/open-discourse/open-discourse/blob/03225c25c451b8331a3dcd25937accc70c44d9ad/python/src/od_lib/04_speech_content/01_extract_speeches.py#L16 (abgerufen: 02.02.23), Zeile 16-20.

[^4]: Andreas Blätte, Julia Rakers, Christoph Leonhardt, How GermaParl Evolves: Improving Data Quality by Reproducible, in: Corpus Preparation and User Involvement In Proceedings of the Workshop ParlaCLARIN III within the 13th Language Resources and Evaluation Conference 2022, https://aclanthology.org/2022.parlaclarin-1.2.pdf (checked: 02.02.2023), p. 11.

[^5]: Florian Richter, et al., Open Discourse: Towards the First Fully Comprehensive and Annotated Corpus of the Parliamentary Protocols of the German Bundestag, SocArXiv Preprint 2023, DOI: https://doi.org/10.31235/osf.io/dx87u (checked 02.02.23), p. 10.

[^6]: Ramisch, Open Discourse.

[^7]: Richter, Open Discourse, p. 10.

[^8]: Todo: ML Machine Learning for XML structuring reference

[^9]: Ralf C. Staudemeyer, Eric Rothstein Morris, Understanding LSTM - a tutorial into Long Short-Term Memory Recurrent Neural Networks, in CoRR 2019, https://arxiv.org/abs/1909.09586 (checked: 07.02.23), p. 2.  
Jacob Devlin, et. al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, CoRR 2018, https://arxiv.org/abs/1810.04805 (checked: 07.02.23), p. 13-14.

[^10]: Grundgesetz für die Bundesrepublik Deutschland, Artikel 42 Abs. 1.

[^11]: Geschäftsordnung des Deutschen Bundestages § 116 Abs. 1, Deutscher Bundestag, https://www.bundestag.de/parlament/aufgaben/rechtsgrundlagen/go_btg/go11-245172 (checked: 14.02.23).

[^12]: Open Discourse, Harvard Dataverse, https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/FIKIBO (checked 08.07.22).

[^13]: 01_extract_speeches.py, Github, https://github.com/open-discourse/open-discourse/blob/03225c25c451b8331a3dcd25937accc70c44d9ad/python/src/od_lib/04_speech_content/01_extract_speeches.py#L16 (abgerufen: 02.02.23), Zeile 16-20.

[^14]: Paul Ramisch, Open Discourse - eine Quellenkritik, 2022, https://paulramisch.de/opendiscourse/6_analyse%2Bevaluation.html (checked: 02.02.2023).

[^15]: The whole process is fully documented in my digital source critique of Open Discourse:
Ramisch, Open Discourse, chapter 4.2 Tagging der Redebeiträge, https://paulramisch.de/opendiscourse/4_vergleichskorpus.html.

[^16]: Aakash Bindal, Measuring just Accuracy is not enough in machine learning, A better technique is required.., Techspace 2019, 
https://medium.com/techspace-usict/measuring-just-accuracy-is-not-enough-in-machine-learning-a-better-technique-is-required-e7199ac36856 (checked: 05.12.2022).

[^17]: Sensitivity and specificity, Wikipedia, https://en.wikipedia.org/wiki/Sensitivity_and_specificity (checked: 05.12.2022).

[^18]: Staudemeyer, Rothstein Morris, Understanding LSTM, p. 2. 

[^19]: Staudemeyer, Rothstein Morris, Understanding LSTM, p. 19-20. 

[^20]:  Staudemeyer, Rothstein Morris, Understanding LSTM, p. 29. 

[^21]: Geoffrey E. Hinton, et al., Improving neural networks by preventing co-adaptation of feature detectors, in: CoRR 2012, https://arxiv.org/abs/1207.0580 (checked: 07.02.23), p. 1.

[^22]: Armand Joulin, et al., Bag of Tricks for Efficient Text Classification, in: CoRR 2016, https://arxiv.org/abs/1607.01759 (checked: 07.02.23), p. 1-2.

[^23]: Word vectors for 157 languages, Fasttext 2018, https://fasttext.cc/docs/en/crawl-vectors.html (checked: 07.02.23).  
Andreas Müller, GermanWordEmbeddings, GitHub 2022, https://github.com/devmount/GermanWordEmbeddings (checked: 07.02.23).

[^24]: Ashish Vaswani, et. al., Attention Is All You Need, CoRR 2017, https://arxiv.org/abs/1706.03762 (checked: 07.02.23).

[^25]: Jacob Devlin, BERT, p. 1-2. 

[^26]: Classification Models, Simple Transformers 2020, https://simpletransformers.ai/docs/classification-models/ (checked: 07.02.23).

[^27]: Jacob Devlin, bert-base-cased, Hugging Face 2018, https://huggingface.co/bert-base-cased (checked: 08.02.23).  
Branden Chan, et. al., bert-base-german-cased, Hugging Face 2019, https://huggingface.co/bert-base-german-cased (checked: 08.02.23).

[^28]: The test of the Open Discourse Heuristic labeler in the file `open_discourse\test_labeler.py`, yields slightly better results with the dataset that was created in **2.1 Training data preparation** than on the XML files because the simple prefilter heurstic already filters out certain false matches and due because not all preprocessing step is implementet there. Therefore it gets an F1 score of 0.9818 on the dataset, while in reality the performance was only 0.9811. Therefore is likely that the model here performs slightly better on the noisy data than its real performance would look like.  
original performance: tp: 7309 tn: unknown fp: 48 fn: 233  
dataset performance: tp: 7332 tn: 13172 fp: 58 fn: 214  