from shutil import copyfile
import os, re, glob, random
from pdb import set_trace as t

source = '/Users/boraerden/Desktop/downsize_vidTIMIT/'
jpegDirs = os.listdir(source)

dst = '/Users/boraerden/Desktop/'
os.chdir(dst)

n_test_phonemes = 0
n_train_phonemes = 500

test_folder_name = dst + 'downsize_vidTIMIT_test_' + str(n_test_phonemes) + '/'
train_folder_name =  dst + 'downsize_vidTIMIT_train_' + str(n_train_phonemes) + '/'


frames = glob.glob(source + '*01.jpg')

copied_frames_test = []
copied_phonemes = []
# TEST
if n_test_phonemes > 0:
	if not os.path.exists(test_folder_name):
		os.makedirs(test_folder_name)

	
	


	while len(copied_phonemes) < 59:
		rand_index = random.randint(1,len(frames)-1)
		frame = frames[rand_index].split('/')[-1]
		phoneme = frame.split('_')[-2]
		if phoneme in copied_phonemes:
			continue

		copied_frames_test.append(frame)
		copied_phonemes.append(phoneme)

		for v in range(1,11):
			source_filename = source + frame[:-6] + '%02d.jpg' % v
			destination_filename = test_folder_name + frame[:-6] + '%02d.jpg' % v
			copyfile(source_filename, destination_filename)

	while len(copied_frames_test) < n_test_phonemes:
		rand_index = random.randint(1,len(frames)-1)
		frame = frames[rand_index].split('/')[-1]
		phoneme = frame.split('_')[-2]
		if frame in copied_frames_test:
			continue

		copied_frames_test.append(frame)

		for v in range(1,11):
			source_filename = source + frame[:-6] + '%02d.jpg' % v
			destination_filename = test_folder_name + frame[:-6] + '%02d.jpg' % v
			copyfile(source_filename, destination_filename)


# TRAIN
if n_train_phonemes > 0:
	if not os.path.exists(train_folder_name):
		os.makedirs(train_folder_name)

	copied_phonemes = []
	copied_frames_train = []
	while len(copied_phonemes) < 59:
		rand_index = random.randint(1,len(frames)-1)
		frame = frames[rand_index].split('/')[-1]
		phoneme = frame.split('_')[-2]
		if phoneme in copied_phonemes:
			continue

		copied_frames_train.append(frame)
		copied_phonemes.append(phoneme)

		for v in range(1,11):
			source_filename = source + frame[:-6] + '%02d.jpg' % v
			destination_filename = train_folder_name + frame[:-6] + '%02d.jpg' % v
			copyfile(source_filename, destination_filename)

	while len(copied_frames_train) < n_train_phonemes:
		rand_index = random.randint(1,len(frames)-1)
		frame = frames[rand_index].split('/')[-1]
		phoneme = frame.split('_')[-2]
		if frame in copied_frames_test or frame in copied_frames_train:
			continue

		copied_frames_train.append(frame)

		for v in range(1,11):
			source_filename = source + frame[:-6] + '%02d.jpg' % v
			destination_filename = train_folder_name + frame[:-6] + '%02d.jpg' % v
			copyfile(source_filename, destination_filename)


