#!/usr/bin/env python
import os
import sys
import socket
import datetime
from random import randint
from optparse import OptionParser
from copy import deepcopy

from train_dlr_hadoop_lib import process_task, create_reports

#from __future__ import with

if __name__ == '__main__':

	get_lambda_list = lambda x: ' --lambda-1 '.join(map(str, x))

	tasks = []

#----------------------------------------------------
#	"small_test" dataset
#----------------------------------------------------

	test_task = {
		'train_tables':		'/user/trofim/dlr/test/small_train_ro_0.1.ii2',
		'label_table':		'/user/trofim/dlr/test/small_train_ro_0.1.label',
		'test_file':		'./small_test_ro_0.1.svm',
		'jobcount':		3,
		'params': {
			'iterations':		32,
			'lambda-1':		64.0,
			'termination':		0.0,
			'combine-type':		0,
			'save-per-iter':	1,
			'beta-max':		1.0e6,
			'linear-search':	1,
			},
		'etalon_loss': 105958.76,
		}

	tasks.append(deepcopy(test_task))

	test_task['params']['admm'] = 1
	test_task['params']['rho'] = 1
	test_task['params']['loss'] = 1

	tasks.append(deepcopy(test_task))

	for task in tasks:
		process_task(task, should_create_reports = True)
