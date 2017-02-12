import os
from pydub import AudioSegment
import numpy as np
import sys
import math

"""
0 = -ax^2 + bx + y
a=1
b = sqrt(n)*2
c = -y
x = (-b+-sqrt(delta))/2a
"""


def generate_features(folder, n):
	r = '/home/git/cut_audio_classifier/corpora/{}'.format(folder)
	n = int(n)
	features_file = os.path.join(r, 'features.tsv'.format(n))
	out_file = open(features_file, 'w+')
	out_data = []
	header = '\t'.join(["avg_{0}\tmax_{0}".format(i) for i in range(n)] + ['class'])
	out_file.write(header+'\n')
	for root, dirs, files in os.walk(r):
		for file in files:		
			if 'tsv' in file:
				continue
			if 'left' in root:
				cl = 'cut'
			if 'right' in root:
				cl = 'cut'
			if 'whole' in root:
				cl = 'ok'

			in_file = AudioSegment.from_file(os.path.join(root, file), format='flac')
			in_file_max = in_file.max
			l = len(in_file)

			splits = [0]*(n+1)
			for i in xrange(1, (n/2)+1):
				splits[i] = get_block_size(n/2+1, i, l)[0]
				splits[n-i+1] = get_block_size(n/2+1, i, l)[1]
			splits[n] = l
			
			splitted = [np.array(in_file[splits[i]:splits[i+1]].get_array_of_samples(), dtype=np.float)/in_file_max for i in xrange(0, n-1)]
			
			out_data.append('\t'.join(["{0}\t{1}".format(get_features(s)[0], get_features(s)[1]) for s in splitted] + [cl]))

	out_file.write('\n'.join(out_data))
	out_file.close()		

def get_block_size(n2, y, length):
	b = math.sqrt(n2)*2
	delta = (b*b) - (4*-1*-y)
	x = (-b + math.sqrt(delta))/-2
	return x*length/b, length - (x*length/b)

def get_features(segment):
	return [np.mean(np.abs(segment)), np.abs(segment).max()]

if __name__ == '__main__':
	generate_features(sys.argv[1], sys.argv[2])