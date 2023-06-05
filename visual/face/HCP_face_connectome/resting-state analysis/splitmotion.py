#!/usr/bin/env/python
#this script takes the text file with 12 columns of movement (Movement_regressors.txt) and copies the first 6 motion parameters into a new file (motionfile_6columns.txt).

import sys, os

inmotionf = sys.argv[1]
outmotionf = sys.argv[2]

motionfile=open(inmotionf, 'r')
motionfile_6columns = open(outmotionf, 'w')
for row in motionfile:
	columns = row.strip().split()
	motionfile_6columns.write("\t".join(columns[:6])+"\n")
motionfile.close()
motionfile_6columns.close()
