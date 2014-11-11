#!/usr/bin/env python
import os
import sys

def execute(cmd):
	''' Simple wrapper for bash scripts '''
	print cmd
	os.system(cmd)

if __name__ == '__main__':
	execute('./dlr -d small_train_ro_0.1.ii2 --iterations 100 --termination 0.0 --lambda-1 16 -f tmp_model --last-iter-sum --linear-search 1 -l small_train_ro_0.1.label --sparse-model 1 --async-cycle')
	#execute('./dlr -d small_train_ro_0.1.ii2 --iterations 10 --termination 0.0 --lambda-1 16 -f tmp_model --last-iter-sum --linear-search 1 -l small_train_ro_0.1.label --sparse-model 1')
	file1 = open('ethalon_model_sparse', 'r')
	file2 = open('tmp_model', 'r')

	for k in xrange(2):
		line1 = file1.readline()
		line2 = file2.readline()

	while (line1 and line2):

		tolerance = 1.0e-4

		idx1, value1 = line1.split(":")
		idx2, value2 = line2.split(":")

		idx1 = int(idx1)
		idx2 = int(idx2)

		if idx1 != idx2 or (abs(float(value1) - float(value2)) > tolerance):
			print 'ERROR', line1, line2
			execute('vimdiff ethalon_model_sparse tmp_model')
			sys.exit()

		line1 = file1.readline().strip()
		line2 = file2.readline().strip()

	print 'OK!'

	
