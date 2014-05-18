#!/usr/bin/python
import os
import sys

''' Creates records with encoded values for MR'''

if __name__ == '__main__':
	jobid = sys.argv[1]
	file = open(sys.argv[2])
	text = file.read() 
	file.close()

	text = text.encode("string-escape")
	
	MAX_VALUE_SIZE = int(1e7)

	for i in xrange(int(len(text) / MAX_VALUE_SIZE) + 1):
		begin = i * MAX_VALUE_SIZE
		end = (i + 1) * MAX_VALUE_SIZE
		sys.stdout.write('%s\t\t%03d%s\n' % (jobid, i, text[begin:end]))
