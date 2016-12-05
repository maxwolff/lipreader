from shutil import copyfile
import os
import re

source = './vidTIMIT/'

dst = './input2/'


jpegDirs = os.listdir(source)
for i in range(1,101):
	if not os.path.exists(dst + str(i)):
		os.makedirs(dst + str(i))
	j = i*500
	while(True):
		if len(re.split('\_', jpegDirs[j])) < 4: continue
		if re.split('\_', jpegDirs[j])[3] == '01.jpg':
			print 'ok'
			break
		j += 1
	for k in range(0,10):
		if not os.path.exists(dst + str(i) + '/' + jpegDirs[j+k]):
			copyfile(source + jpegDirs[j + k], dst + str(i) + '/' + jpegDirs[j+k])