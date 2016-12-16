import os, sys
import re

data_folder = ''

jpegDir = sorted(os.listdir(data_folder))
frame_index = 0
for jpegDir in jpegDirs:
		#speaker_sentence_phoneme_frameNumber
		if jpegDir[-3:] != 'jpg': continue
		specs = re.split('\_', jpegDir)
		if specs[0] != curSpecs[0] or specs[1] != curSpecs[1] or specs[2] != curSpecs[2]:
			frame_index = 0
		frame_index += 1
		if frame_index >= 10:
			print(jpegDir)