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
loc("mps_matmul"("(mpsFileLoc): /AppleInternal/Library/BuildRoots/a0876c02-1788-11ed-b9c4-96898e02b808/Library/Caches/com.apple.xbs/Sources/MetalPerformanceShadersGraph/mpsgraph/MetalPerformanceShadersGraph/Core/Files/MPSGraphUtilities.mm":28:0)): error: inner dimensions differ 14 & 128

loc("mps_matmul"("(mpsFileLoc): /AppleInternal/Library/BuildRoots/a0876c02-1788-11ed-b9c4-96898e02b808/Library/Caches/com.apple.xbs/Sources/MetalPerformanceShadersGraph/mpsgraph/MetalPerformanceShadersGraph/Core/Files/MPSGraphUtilities.mm":28:0)): error: invalid shape

Issue: probably can't handle smaller batch size
 error: inner dimensions differ 14 & 128
LLVM ERROR: Failed to infer result type(s).
torch.Size([64, 128])
torch.Size([45, 128])

# Transformer

# Model comparison