#!/usr/bin/env python
import os
import sys
import random

def execute(cmd):
	''' Simple wrapper for bash scripts '''
	print cmd
	os.system(cmd)

def fast_test(cmd, ethalon_filename, tolerance = 1.0e-4, head = 0, model = 'tmp_model'):

	execute('rm tmp_model')
	execute(cmd)
 	
	if isinstance(model, list):
		merge_models(model, 'tmp_model', style = 'vw')
		model = 'tmp_model'

	if len(ethalon_filename) == 0:
 		return 

	file1 = open(ethalon_filename, 'r')
	file2 = open('tmp_model', 'r')

	for k in xrange(2):
		line1 = file1.readline()
		line2 = file2.readline()

	while (line1 and line2):

		idx1, value1 = line1.split(":")
		idx2, value2 = line2.split(":")

		idx1 = int(idx1)
		idx2 = int(idx2)

		if idx1 != idx2 or (abs(float(value1) - float(value2)) > tolerance):
			print 'ERROR', line1, line2
			execute('vimdiff %s tmp_model' % ethalon_filename)
			sys.exit()

		if head > 0 and (idx1 > head or idx2 > head):
			break

		line1 = file1.readline().strip()
		line2 = file2.readline().strip()

	print 'OK!'

if __name__ == '__main__':

	execute('gunzip small_train_ro_0.1.ii2.gz -c > small_train_ro_0.1.ii2')

	print 'Testing d-GLMNET, async'
	fast_test('./dlr -d small_train_ro_0.1.ii2 --iterations 10 --termination 0.0 --lambda-1 16 -f tmp_model --last-iter-sum --linear-search 1 -l small_train_ro_0.1.label --sparse-model 1 --async-cycle 1',
			'ethalon_model_sparse', tolerance = 1.0e-2, head = 50)

	print 'Testing d-GLMNET, sync'
	fast_test('./dlr -d small_train_ro_0.1.ii2 --iterations 10 --termination 0.0 --lambda-1 16 -f tmp_model --last-iter-sum --linear-search 1 -l small_train_ro_0.1.label --sparse-model 1 --async-cycle 0',
			'ethalon_model_sparse')
	
	print 'Testing ADMM'
	fast_test('./dlr -d small_train_ro_0.1.ii2 --iterations 10 --termination 0.0 --lambda-1 16 -f tmp_model --admm --rho 1 --loss 1 -l small_train_ro_0.1.label --sparse-model 1', 
			'ethalon_model_admm')
