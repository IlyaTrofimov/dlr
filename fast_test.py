#!/usr/bin/env python
import os
import sys

def execute(cmd):
	''' Simple wrapper for bash scripts '''
	print cmd
	os.system(cmd)

if __name__ == '__main__':
	execute('./dlr -d small_train_ro_0.1.ii2 --iterations 10 --lambda-1 16 -f tmp_model --last-iter-sum --linear-search 1 -l small_train_ro_0.1.label')

	file1 = open('tmp_model', 'r')
	file2 = open('ethalon_model', 'r')

	line1 = file1.readline()
	line2 = file2.readline()

	is_weights = False

	while (line1 and line2):

		tolerance = 1.0e-4

		if is_weights:
			if abs(float(line1) - float(line2)) > tolerance:
				print 'ERROR', line1, line2
				execute('vimdiff ethalon_model tmp_model')
				sys.exit()

		if line1 == 'w':
			is_weights = True					

		line1 = file1.readline().strip()
		line2 = file2.readline().strip()

	print 'OK!'

	
