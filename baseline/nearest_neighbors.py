import os, sys
import re
import cv2
from pdb import set_trace as t


phonemes = []
jpegDirs = os.listdir('./vidTIMIT/')
for jpegDir in jpegDirs:
	if jpegDir[-3:] != 'jpg': continue
	specs = re.split('\_', jpegDir)
	if specs[2] not in phonemes: phonemes.append(specs[2])
print(phonemes)


