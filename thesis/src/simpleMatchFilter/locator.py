#!/usr/bin/env python
import numpy as np
import sys, getopt
import os
import scipy.stats as st  

def result(original_file_name ,filtered_file_name):
	#print("original:"+original_file_name,"filterd:"+filtered_file_name)
	original_data = np.loadtxt(original_file_name,delimiter=',')
	filtered_data = np.loadtxt(filtered_file_name,delimiter=',')
	sizeA , null = np.shape(original_data)
	sizeB , null = np.shape(filtered_data)
	dataSize = sizeB
	if sizeA > sizeB:
		dataSize = sizeB

	# print("size of sample:",dataSize)
	y1 = np.resize(original_data[:,1],dataSize)
	y2 = np.resize(filtered_data[:,1],dataSize)	
	
	score = np.corrcoef(y1,y2)
	return score[0,1]
	# score = st.spearmanr(y1, y2)
	# return score[1]

if len(sys.argv) < 3:
	print("not enouge argumetns")
	sys.exit()	

original_dir_name=sys.argv[1]
filtered_dir_name=sys.argv[2]

for on in os.listdir(original_dir_name):
	match=0
	match_name = ""
	for fn in os.listdir(filtered_dir_name):
	        #print (result(original_dir_name+on,filtered_dir_name+fn))
	        p_match = result(original_dir_name+on,filtered_dir_name+fn)
	        #if abs(p_match) > abs(match):
	        if p_match > match:
	        	match = p_match
	        	match_name = fn
	print(on,match_name)        	
