import numpy as np
import cv2
from pdb import set_trace as t
import os
import random

def time_frames(frames_per_phoneme, phoneme_frames):
	
	if len(phoneme_frames) == frames_per_phoneme:
		return phoneme_frames

	#if not enough frames
	if len(phoneme_frames) < frames_per_phoneme:
		timed_frames = phoneme_frames
		n_missing_frames = frames_per_phoneme - len(phoneme_frames)
		frames_to_add = random.sample(range(frames_per_phoneme+1), n_missing_frames)
		for frame_to_add in frames_to_add:
			a=1
		# 	timed_frames[frames_to_add] = interpolate(frame before, frame after)
		return timed_frames

	#if too many frames
	if len(phoneme_frames) > frames_per_phoneme:
		timed_frames = phoneme_frames
		n_extra_frames = len(phoneme_frames) - frames_per_phoneme
		frames_to_remove = random.sample(range(len(phoneme_frames)+1), n_extra_frames)

		#in reverse order to not mess up the list as it goes
		for frame_to_remove in sorted(frames_to_remove, reverse=True):
			timed_frames.pop(frame_to_remove)
		return phoneme_frames




frames_per_phoneme = 10

mouth_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.13.1/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml')

vidTIMIT_folder = '/Users/boraerden/Google Drive/classes/4 Senior/Seni 1/cs221/cs_221_project/vidTIMIT/'
phoneme_mouth_frames_folder = vidTIMIT_folder + 'phoneme_mouth_frames/'
data_folder = vidTIMIT_folder + 'data/'

#get phoneme timings in dict
phoneme_timings = {}
sentence_lines = [line.rstrip('\n') for line in open(vidTIMIT_folder + 'allphonetime.txt')]
for sentence_line in sentence_lines:
	stripped_line = sentence_line.split()
	train_test, dialect, speaker, sentence = stripped_line[0].split('/')
	if speaker not in phoneme_timings.keys():
		phoneme_timings[speaker] = {}
	phoneme_timings[speaker][sentence] = stripped_line[1:]


os.chdir(phoneme_mouth_frames_folder)

speakers = os.listdir(data_folder)
for speaker_index, speaker in enumerate(speakers):
	print 'speaker ' + str(speaker_index) + '/' + str(len(speakers)) + ': ' + speaker
	#if not formatted data file (starting with f for female or m for male), skip 
	if speaker[0] != 'f' and speaker[0] != 'm':
		print 'skip'
		continue
	
	video_folder = data_folder + speaker + '/video/'
	videos = os.listdir(video_folder)
	for video_index, video in enumerate(videos):
		print '    video ' +  str(video_index) +'/' +str(len(videos)) + ': ' + video
		#if not a sentence video folder, skip
		if video[0] != 's':
			print '    skip'
			continue

		#cop them frames
		frame_folder = video_folder + video + '/'
		frames_filenames = os.listdir(frame_folder)
		frames = [None] * len(frames_filenames)
		for frame_index, frame_filename in enumerate(frames_filenames):
			#print '        frame '+ str(frame_index)+ '/' + str(len(frames_filenames)) + ': ' + frame_filename
			if frame_filename == 'Icon\r':
				#print '        skip'
				continue
			img = cv2.imread(frame_folder + '/' + frame_filename)
			frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

		phonemes = phoneme_timings[speaker][video]
		timings_end = float(phonemes[-1])
		n_frames = len(frames)


		for phoneme_index, phoneme in enumerate(phonemes):
			#only stop every 3 since list is in format = phoneme, start_timing, end_timing
			if phoneme.isdigit() or phoneme == 'h#':
				continue
			start_timing = float(phonemes[phoneme_index + 1])
			end_timing = float(phonemes[phoneme_index + 2])

			start_frame = int(round(start_timing*n_frames/timings_end) + 1)
			end_frame = int(round(end_timing*n_frames/timings_end) + 1)
			phoneme_frames = []

			#extra +1 because range is end exclusive
			for frame_index in range(start_frame, end_frame + 1):
				phoneme_frames.append(frames[frame_index])

			timed_frames = time_frames(frames_per_phoneme, phoneme_frames)



		#save as .jpg, 00 pad
		#same number of frames
		#same width and height



################################################################################

# #code to figure out longest phoneme video
# max_frames_1 = 0
# max_frames_2 = 0
# max_frames_3 = 0
# max_frames_4 = 0
# max_frames_5 = 0
# 			#in for phoneme_index
# 			vid_length = end_frame - start_frame
# 			if vid_length > max_frames_1:
# 				max_frames_1 = vid_length
# 			elif vid_length > max_frames_2:
# 				max_frames_2 = vid_length
# 			elif vid_length > max_frames_3:
# 				max_frames_3 = vid_length
# 			elif vid_length > max_frames_4:
# 				max_frames_4 = vid_length
# 			elif vid_length > max_frames_5:
# 				max_frames_5 = vid_length

# print 'top 5 is :' + str(max_frames_1) + ', ' + str(max_frames_2) + ', ' + str(max_frames_3) + ', ' + str(max_frames_4) + ', ' + str(max_frames_5)
	
