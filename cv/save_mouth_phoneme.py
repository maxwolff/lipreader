import numpy as np
import cv2
from pdb import set_trace as t
import os
import random, math

def interpolate(prev_frame, next_frame):
	# to do: figure out interpolation
	return prev_frame


def time_frames(frames_per_phoneme, phoneme_frames):
	n_exis_frames = len(phoneme_frames)

	if n_exis_frames == frames_per_phoneme:
		return phoneme_frames

	#if not enough frames
	if n_exis_frames < frames_per_phoneme:
		timed_frames = phoneme_frames
		n_missing_frames = frames_per_phoneme - n_exis_frames
		for f in range(n_missing_frames):
			if len(timed_frames) == 1:
				frame_to_add_index = 0
				prev_frame = timed_frames[frame_to_add_index]
				next_frame =  timed_frames[frame_to_add_index]
			else:
				frame_to_add_index = random.choice(range(1, len(timed_frames)))
				prev_frame = timed_frames[frame_to_add_index-1]
				next_frame =  timed_frames[frame_to_add_index]

		 	timed_frames.insert(frame_to_add_index, interpolate(prev_frame, next_frame))

		return timed_frames

	#if too many frames
	if len(phoneme_frames) > frames_per_phoneme:
		timed_frames = phoneme_frames
		n_extra_frames = len(phoneme_frames) - frames_per_phoneme
		frames_to_remove = random.sample(range(n_exis_frames), n_extra_frames)

		#in reverse order to not mess up the list as it goes
		for frame_to_remove in sorted(frames_to_remove, reverse=True):
			timed_frames.pop(frame_to_remove)

		return timed_frames

frames_per_phoneme = 10

mouth_width = 85
mouth_height = 50

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
already_done = ['fadg0', 'faks0','fcft0','fcmh0', 'fcmr0', 'fcrh0', 'fdac1', 'fdms0', 'fdrd1', 'fedw0', 'felc0', 'fgjd0', 'fjas0', 'fjem0', 'fjre0', 'fjwb0', 'fkms0', 'fpkt0', 'fram1', 'mabw0', 'mbdg0', 'mbjk0', 'mccs0', 'mcem0', 'mdab0', 'mdbb0', 'mdld0', 'mgwt0', 'mjar0', 'mjsw0', 'mmdb1', 'mmdm2', 'mpdf0']

for speaker_index, speaker in enumerate(speakers):
	print 'speaker ' + str(speaker_index) + '/' + str(len(speakers)) + ': ' + speaker
	#if not formatted data file (starting with f for female or m for male), skip 
	if speaker[0] != 'f' and speaker[0] != 'm':
		print 'skip'
		continue

	if speaker in already_done:
		print 'already done'
		continue
	
	video_folder = data_folder + speaker + '/video/'
	videos = os.listdir(video_folder)
	for video_index, video in enumerate(videos):
		print '    video ' +  str(video_index) +'/' +str(len(videos)) + ': ' + video
		#if not a sentence video folder, skip
		if video[0] != 's':
			print '    skip'
			continue

		if video not in phoneme_timings[speaker].keys():
			print '    video not in phoneme timing .txt'
			continue

		#cop them frames
		frame_folder = video_folder + video + '/'
		frames_filenames = os.listdir(frame_folder)
		frames = []
		for frame_index, frame_filename in enumerate(frames_filenames):
			#print '        frame '+ str(frame_index)+ '/' + str(len(frames_filenames)) + ': ' + frame_filename
			if frame_filename == 'Icon\r':
				#print '        skip'
				continue
			img = cv2.imread(frame_folder + '/' + frame_filename)
			frames.append(img)

		phonemes = phoneme_timings[speaker][video]
		timings_end = float(phonemes[-1])
		n_frames = len(frames)


		for phoneme_index, phoneme in enumerate(phonemes):
			#only stop every 3 since list is in format = phoneme, start_timing, end_timing
			if phoneme.isdigit() or phoneme == 'h#':
				continue
			start_timing = float(phonemes[phoneme_index + 1])
			end_timing = float(phonemes[phoneme_index + 2])

			start_frame = int(round(start_timing*n_frames/timings_end))
			end_frame = int(math.floor(end_timing*n_frames/timings_end))
			phoneme_frames = []

			
			#extra +1 because range is end exclusive
			for frame_index in range(start_frame, end_frame + 1):
				phoneme_frames.append(frames[frame_index])

			if len(phoneme_frames) == 0:
				continue

			timed_frames = time_frames(frames_per_phoneme, phoneme_frames)

			assert len(timed_frames) == 10

			for frame_index, frame in enumerate(timed_frames):
				# mouths = mouth_cascade.detectMultiScale(frame, 1.3, 5)
				# biggestMouthArea = 0
				# biggestMouth = 0,0,0,0
				# for (x,y,w,h) in mouths:
				# 	if w*h > biggestMouthArea:
				# 		biggestMouthArea = w*h
				# 		biggestMouth = x,y,w,h
				# x,y,w,h = biggestMouth
				
				# cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
				# cv2.imshow('img',frame)
				# cv2.waitKey(0)

				# start_x = (x+w-mouth_width)/2
				# end_x = (x+w+mouth_width)/2
				# start_y = (y+h-mouth_height)/2
				# end_y = (y+h+mouth_height)/2
				# cropped_sized_mouth = frame[start_y:end_y, start_x:end_x]
				# cv2.imwrite("%s_%s_%02d.jpg" % (speaker, phoneme, frame_index+1), cropped_sized_mouth)
				
				#cv2.CreateImage( mouth_height, mouth_width, frame.depth, frame.nChannels)

				cv2.imwrite("%s_%s_%s_%02d.jpg" % (speaker, video, phoneme, frame_index+1), frame)

		#save as .jpg, 00 pad
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

################################################################################


# #code to figure out average mouth width and height
# mouth_widths = []
# mouth_heights = []

# 				mouth_widths.append(w)
# 				mouth_heights.append(h)

# print 'avg mouth width:  ' + str(np.mean(np.array(mouth_widths)))
# print 'std mouth width:  ' + str(np.std(np.array(mouth_widths)))
# print 'avg mouth height: ' + str(np.mean(np.array(mouth_heights)))
# print 'std mouth height: ' + str(np.std(np.array(mouth_heights)))


