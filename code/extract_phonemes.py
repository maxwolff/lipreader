import os, sys
import re

phonemes = []
jpegDirs = os.listdir('/Users/wamsood/Desktop/speaker1/')
for jpegDir in jpegDirs:
	if not 'jpg' in jpegDir: continue
	specs = re.split('\_', jpegDir)
	if specs[2] not in phonemes:
		phonemes.append(specs[2])
phonemes = sorted(phonemes)
print phonemes
print len(phonemes)
print phonemes[41]