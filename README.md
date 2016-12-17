# Neural Network Lipreading 
by Max Wolff, Bora Erden, Sam Wood 

For our final project in CS 221, we set out to build a lip-reader, which takes audio-free videos of people speaking and reconstruct their spoken sentences. We were drawn to this problem because of its complexity and variety of applications. Most compelling is assisting people with hearing impairments. If this algorithm could be improved to run in real-time, it could be a boon to the deaf, given that even the most accurate human lipreaders are far from perfect.

We implemented a phoneme classifier, which takes in a short clip of a subject speaking one phoneme (phonemes are the fundamental units of sound for a spoken language) and produces an array containing the estimated probabilities that that clip corresponds to any particular phoneme. We also built an NLP model, so that the phoneme classifier could be used to move from audio-free video to a full reconstruction of a spoken sentence.
 

We used the VidTIMIT audio-video dataset, found [here](https://catalog.ldc.upenn.edu/docs/LDC93S1/). It contains high-contrast, close-up videos of 43 people speaking into the camera, along with phonemic transcriptions of their sentences. Implemented using Python, and Keras, and borrowing ideas for spatio-temporal convolutional neural network from [Lipnet](https://openreview.net/pdf?id=BkjLkSqxg). 




## Summary of Important Files 

data.zip
	This folder contains all of the pre-processed frames for the first two speakers in our dataset.

###Baseline:
nearest_neighbor.py
	This simple script creates a "standard" image for each phoneme by averaging
	50 frames of that phoneme distributed among the various speakers.
	When given an input phoneme clip, this algorithm classifies the clip as the phoneme
	whose standard image is closest to the frames of the clip.
	When we tested this algorithm on a set of 3,000 phoneme clips, it achieved an accuracy
	of 2%, just above chance.


###Data pre-processing:
save_mouth_phoneme.py
	This script loads the phoneme timing for each speaker and video first from the allphonetime.txt.
	It then retrieves all the frames for that phoneme, and saves the first, middle and last ones' face cropped grayscaled versions.

###Our model:
keras3d.py
	This script loads the data and labels from a folder containing the training set, partitions the data into randomly shuffled batches, and trains our seven-layer STCNN
	using stochastic gradient descent optimized against categorical cross-entropy loss.

keras3d_test.py
	This script loads a model saved by keras3d.py, and evaluates the model on data extracted
	from a folder containing the test set. The script reports the accuracy, the top-3 accuracy, and the top-5 accuracy of the trained model on the test set; the predicted class for each example in the test set; and the confusion matrix for the phoneme classes.

keras3d_1speaker.py
keras3d_test.py
	These scripts function just like the previous two, but they are designed to run on
	a training set comprised only of data from one speaker. For this reason, they do not
	partition the data into batches.

keras2d_train_batches.py
keras2d_train.py
	We used these scripts to improve the architecture of our neural net and to develop our batching code while we were still unable to get our 3-D CNN to run due to memory overflow issues.

###NLP reconstruction:

makeDics.py
	Parses and transcribes data from VidTIMIT, put them into maps usable in the search problem 

segProblem.py
	Formats sentence construction as a search problem for the UCS algorithm in util.py. Calls UCS on sentences and tests accuracy of the algorithm. 

Miscellaneous helpful scripts:
extract_phonemes.py
	This script reports the phonemes contained in folder containing a dataset. We used this
	whenever we trained on a set that did not contain all 59 phonemes (e.g., a dataset from just one speaker).

find_bad_data.py
	This script ensures that there are exactly 3 frames for each phoneme in a dataset.


make_test_train_set.py
	This script produces a dataset of the specified size containing at least one example
	of each of the 59 phonemes (so long as the specified size is greater than 59). We used this script to produce training sets, validation sets, and test sets.
