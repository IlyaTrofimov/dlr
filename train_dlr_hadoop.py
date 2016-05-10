#!/usr/bin/env python
import os
import sys
import socket
import datetime
from random import randint
from optparse import OptionParser
from copy import deepcopy

from test_dlr_dist import local_dist_run
from test_dlr import get_models_metrics
from train_dlr_hadoop_lib import process_task, create_reports

#from __future__ import with

if __name__ == '__main__':

	get_lambda_list = lambda x: ' --lambda-1 '.join(map(str, x))

	tasks = []
#
#----------------------------------------------------
#	"small test" dataset
#----------------------------------------------------

	base_task = {
		'local-dist-test':	True,
		'local-dataset':	True,
		'train_tables':		'small_train_ro_0.1.ii2',
		'label_table':		'small_train_ro_0.1.label',
		'test_file':		'small_test_ro_0.1.svm',
		'jobcount':		4,
		'params': {
			'iterations':		100,
			'lambda-1':		16.0,
			'termination':		1.0e-4,
			'combine-type':		0,
			'save-per-iter':	1,
			'beta-max':		10,
			},
		'etalon_loss': 4774.1,
		'save_dataset': False,
		}

	task = deepcopy(base_task)
	task['params']['combine-type'] = 0
	task['params']['find-bias'] = 1
	task['params']['linear-search'] = 1
	task['params']['last-iter-sum'] = 1.0

#	tasks.append(task)

#	task = deepcopy(base_task)
#	task['params']['combine-type'] = 6;
#	task['params']['find-bias'] = 1;
#	task['params']['initial-shrinkage'] = 1;
#	task['params']['increase-shrinkage'] = 1;
#	task['params']['decrease-shrinkage'] = 1;
#	tasks.append(task)

#
#----------------------------------------------------
#	"yandex_ad2_scad" dataset
#----------------------------------------------------

	base_task = {
		'train_tables':		'dlr/data/yandex_ad2.train.ii2',
		'label_table':		'dlr/data/yandex_ad2.train.label',
		'test_file':		'/home/trofim/dlr_data/yandex_ad2.test.svm',
		'jobcount':		16,
		'params': {
			'iterations':		20,
			'lambda-1':		get_lambda_list([8,4,2,1,0.5,0.25,0.125]),
			'scad-a':		3,
			'termination':		1.0e-2,
			'combine-type':		0,
			'beta-max':		10,
			},
		'etalon_loss': 16986380.197,
		}

	yandex_ad_scad_task = deepcopy(base_task)
	yandex_ad_scad_task['params']['combine-type'] = 0;
	yandex_ad_scad_task['params']['find-bias'] = 1;
	yandex_ad_scad_task['params']['last-iter-sum'] = 1;
	yandex_ad_scad_task['params']['ada-alpha'] = 1;
	yandex_ad_scad_task['params']['async-cycle'] = 1

#
#----------------------------------------------------
#	"yandex_ad2" dataset
#----------------------------------------------------

	base_task = {
		'train_tables':		'dlr/data/yandex_ad2.train.ii2',
		'label_table':		'dlr/data/yandex_ad2.train.label',
		'test_file':		'/home/trofim/dlr_data/yandex_ad2.test.svm',
		'jobcount':		64,
		'params': {
			'iterations':		20,
#			'lambda-1':		2,
			'lambda-1':		0,
			'lambda-2':		16,
			'termination':		0.0,
			'combine-type':		0,
			'save-per-iter':	1,
			'beta-max':		10,
			},
		'etalon_loss': 16986380.197,
#		'report_only':			'./dump-20150213-1',
		}

	yandex_ad_task = deepcopy(base_task)
	yandex_ad_task['params']['combine-type'] = 0;
	yandex_ad_task['params']['find-bias'] = 1;
	yandex_ad_task['params']['linear-search'] = 1;
	yandex_ad_task['params']['last-iter-sum'] = 1;
	yandex_ad_task['params']['ada-alpha'] = 1;
	yandex_ad_task['params']['async-cycle'] = 1
#	task['params']['termination'] = 1.0e-3;
#	task['params']['lambda-path'] = 10
#	task['params']['lambda-1'] = get_lambda_list([22229.60547, 11114.80273, 5557.401367, 2778.700684, 1389.350342, 694.6751709, 347.3375855, 173.6687927, 86.83439636, 43.41719818])
#	task['params']['lambda-1'] = 5557.401367
#	task['initial-regressor'] = '/mnt/raid/home/trofim/dlr/dump-20140531-0/model.001.004'
#	task['params']['iterations'] = 10;
#	task['params']['lambda-1'] = get_lambda_list([pow(2, x) for x in xrange(10, -1, -1)])
#	tasks.append(task)

#
#----------------------------------------------------
#	"yandex_ad2_l2" dataset
#----------------------------------------------------

	base_task = {
		'train_tables':		'dlr/data/yandex_ad2.train.ii2',
		'label_table':		'dlr/data/yandex_ad2.train.label',
		'test_file':		'/home/trofim/dlr_data/yandex_ad2.test2.svm',
		'jobcount':		64,
		'params': {
			'iterations':		20,
			'lambda-1':		0,
			'lambda-2':		16,
			'termination':		0.0,
			'combine-type':		0,
			'save-per-iter':	1,
			'beta-max':		10,
			},
		'etalon_loss': 16986380.197,
	#	'report_only':			'./dump-20150409-1',
		}

	yandex_ad_l2_task = deepcopy(base_task)
	yandex_ad_l2_task['params']['find-bias'] = 1;
	yandex_ad_l2_task['params']['linear-search'] = 1;
	yandex_ad_l2_task['params']['ada-alpha'] = 1;
	yandex_ad_l2_task['params']['async-cycle'] = 1

#
#----------------------------------------------------
#	"fram" dataset
#----------------------------------------------------

	base_task = {
		'train_tables':		'dlr/data/fram.train.ii',
		'label_table':		'dlr/data/fram.train.label',
		'test_file':		'/home/trofim/dlr_data/yandex_ad2.test2.svm',
		'jobcount':		64,
		'params': {
			'iterations':		20,
			'lambda-1':		0,
			'lambda-2':		40,
			'termination':		0.0,
			'combine-type':		0,
			'save-per-iter':	1,
			'beta-max':		10,
			},
		'etalon_loss': 16986380.197,
		}

	fram_task = deepcopy(base_task)
	fram_task['params']['find-bias'] = 1;
	fram_task['params']['linear-search'] = 1;
	fram_task['params']['ada-alpha'] = 1;
	fram_task['params']['async-cycle'] = 1


#	task = deepcopy(base_task)
#	task['params']['combine-type'] = 1;
#	task['params']['find-bias'] = 1;
#	tasks.append(task)

#	for shrinkage in [128.0, 64.0, 32.0, 16.0]:
#		task = deepcopy(base_task)
#		task['params']['combine-type'] = 6;
#		task['params']['initial-shrinkage'] = shrinkage;
#		task['params']['find-bias'] = 1;
#		tasks.append(task)

#	task = deepcopy(base_task)
#	task['params']['combine-type'] = 6;
#	task['params']['initial-shrinkage'] = 64.0;
#	task['params']['increase-shrinkage'] = 0;
#	tasks.append(task)



#
#----------------------------------------------------
#	"epsilon" dataset
#----------------------------------------------------

	base_task = {
		'train_tables':		'dlr/data/epsilon.train.ii2',
		'label_table':		'dlr/data/epsilon.train.label',
		'test_file':		'/home/trofim/dlr_data/epsilon_normalized.t',
		'jobcount':		16,
		'params': {
			'iterations':		150,
			'lambda-1':		2.0,
			'termination':		0.0e-6,
			'combine-type':		0,
			'save-per-iter':	1,
			'beta-max':		100,
			},
		'etalon_loss': 105958.76,
		}

	epsilon_task = deepcopy(base_task)
	epsilon_task['params']['linear-search'] = 1;
	epsilon_task['params']['last-iter-sum'] = 1;
	epsilon_task['params']['async-cycle'] = 1

#
#----------------------------------------------------
#	"epsilon" dataset, reg path
#----------------------------------------------------

	base_task = {
		'train_tables':		'dlr/data/epsilon.train.ii2',
		'label_table':		'dlr/data/epsilon.train.label',
		'test_file':		'/home/trofim/dlr_data/epsilon_normalized.t',
		'jobcount':		16,
		'params': {
			'iterations':		100,
			'termination':		1.0e-3,
			'combine-type':		0,
			'beta-max':		100,
			'lambda-path':		20,
			'scad-a':		3,
			},
		'etalon_loss': 105958.76,
		}

	epsilon_path_task = deepcopy(base_task)
#	epsilon_path_task['params']['linear-search'] = 1;
	epsilon_path_task['params']['last-iter-sum'] = 1;

#----------------------------------------------------
#	L2: "epsilon" dataset
#----------------------------------------------------

	base_task = {
		'train_tables':		'dlr/data/epsilon.train.ii2',
		'label_table':		'dlr/data/epsilon.train.label',
		'test_file':		'/home/trofim/dlr_data/epsilon.test2.svm',
		'jobcount':		16,
		'params': {
			'iterations':		100,
			'lambda-1':		0.0,
			'lambda-2':		0.25,
			'termination':		0.0e-6,
			'combine-type':		0,
			'save-per-iter':	1,
			'beta-max':		100,
			},
		'etalon_loss': 105958.76,
		}

	epsilon_l2_task = deepcopy(base_task)
	epsilon_l2_task['params']['linear-search'] = 1;
	epsilon_l2_task['params']['async-cycle'] = 1


#----------------------------------------------------
#	"webspam" dataset
#----------------------------------------------------
#
	base_task = {
                'train_tables':         'dlr/data/webspam.train.ii2', 
                'label_table':          'dlr/data/webspam.train.label',
                'test_file':            '/home/trofim/dlr_data/webspam_wc_normalized_trigram.test',
		'jobcount':		16,
		'params': {
			'iterations':		100,
			'lambda-1':		1 / 64.0,
			'termination':		0.0,
			'combine-type':		0,
			'save-per-iter':	1,
			'beta-max':		1.0e6,
			},
		'etalon_loss': 2005.81766840625
		}

	webspam_task = deepcopy(base_task)
#	webspam_task['params']['termination'] = 1.0e-3
	webspam_task['params']['linear-search'] = 1;
	webspam_task['params']['last-iter-sum'] = 1;
#	webspam_task['params']['lambda-1'] = get_lambda_list([pow(2, -x) for x in xrange(-2, 12)])
#	webspam_task['params']['lambda-path'] = 20

#----------------------------------------------------
#	"webspam" dataset L2
#----------------------------------------------------
#
	base_task = {
                'train_tables':         'dlr/data/webspam.train.ii2', 
                'label_table':          'dlr/data/webspam.train.label',
                'test_file':            '/home/trofim/dlr_data/webspam_wc_normalized_trigram.test2.svm',
		'jobcount':		16,
		'params': {
			'iterations':		100,
			'lambda-1':		0,
			'lambda-2':		1 / 4.0,
			'termination':		0.0,
			'combine-type':		0,
			'save-per-iter':	1,
			'beta-max':		1.0e6,
			},
		'etalon_loss': 2005.81766840625
		}

	webspam_l2_task = deepcopy(base_task)
	webspam_l2_task['params']['async-cycle'] = 1
	webspam_l2_task['params']['linear-search'] = 1;

#	tasks.append(task)

#	create_reports('./dump-20140519-0', task)
#	sys.exit()

#----------------------------------------------------
#	"dna" dataset
#----------------------------------------------------

	base_task = {
		'train_tables':		'users/trofim/genkin/dna.train.ii2',
		'label_table':		'users/trofim/genkin/dna.train.label',
		'test_file':		'/mnt/raid/home/trofim/genkin/dna/dna.test.svm',
		'jobcount':		16,
		'params': {
			'iterations':		32,
			'lambda-1':		64.0,
			'termination':		1.0e-3,
			'combine-type':		0,
			'save-per-iter':	1,
			'beta-max':		1.0e6,
			'linear-search':	1,
			},
		'etalon_loss': 105958.76,
#		'report_only': './dump-20140327-0',
		}

	task = deepcopy(base_task)
	task['params']['termination'] = 1.0e-3;
	task['params']['last-iter-sum'] = 1;
	task['params']['lambda-1'] = get_lambda_list([5461.335938, 4754.362269, 4138.906824, 3603.122507, 3136.695836, 2730.667969])
#	task['params']['lambda-path'] = 20

#	tasks.append(task)

#	parent_dir = '/mnt/raid/home/trofim/dlr/meta-dump-20140926/'
#	dumps = ['dump-20140925-0', 'dump-20140925-1', 'dump-20140925-2', 'dump-20140925-3', 'dump-20140925-4', 'dump-20140926-0', 'dump-20140926-1', 'dump-20140926-2']
#
#	for dump in dumps:
#		create_reports(a_dump_dir = parent_dir + dump, task = None, calc_metrics = False)

#----------------------------------------------------
#	"small_test" dataset
#----------------------------------------------------

	base_task = {
		'train_tables':		'/user/trofim/dlr/test/small_train_ro_0.1.ii2',
		'label_table':		'/user/trofim/dlr/test/small_train_ro_0.1.label',
		'test_file':		'./small_test_ro_0.1.svm',
		'jobcount':		3,
		'params': {
			'iterations':		32,
			'lambda-1':		16.0,
			'termination':		1.0e-6,
			'combine-type':		0,
			'save-per-iter':	1,
			'beta-max':		1.0e6,
			'linear-search':	1,
			},
		'etalon_loss': 105958.76,
#		'report_only': './dump-20140327-0',
		}

	test_task = deepcopy(base_task)
	test_task['params']['async-cycle'] = 1
#	test_task['params']['admm'] = 1
#	test_task['params']['rho'] = 1
#	test_task['params']['loss'] = 1

#	tasks.append(test_task)

#	parent_dir = '/mnt/raid/home/trofim/dlr/meta-dump-20140926/'
#	dumps = ['dump-20140925-0', 'dump-20140925-1', 'dump-20140925-2', 'dump-20140925-3', 'dump-20140925-4', 'dump-20140926-0', 'dump-20140926-1', 'dump-20140926-2']
#
#	for dump in dumps:
#		create_reports(a_dump_dir = parent_dir + dump, task = None, calc_metrics = False)

#	for i in [2, 4]:
#		epsilon_task['params']['rho'] = 2 ** i;
#		tasks.append(deepcopy(epsilon_task))

#	for i in range(10):
		#epsilon_task['params']['async-cycle'] = i % 2;
#		epsilon_task['params']['admm'] = 1
#		epsilon_task['params']['rho'] = 1
#		epsilon_task['params']['loss'] = 1

#		tasks.append(deepcopy(epsilon_task))

#	for async_count in [1, 2, 4, 8]:
#		epsilon_task['params']['async-cycle'] = 1
#		epsilon_task['params']['async-count'] = async_count
#		epsilon_task['params']['iterations'] = 100 * async_count
#		epsilon_task['params']['ada-alpha'] = 1
#		tasks.append(deepcopy(epsilon_task))

#	for i in xrange(9):
#		epsilon_task['params']['admm'] = 1
#		epsilon_task['params']['rho'] = pow(4, -1)
#		epsilon_task['params']['loss'] = 1
#		tasks.append(deepcopy(epsilon_task))
#
#		del epsilon_task['params']['admm']
#		epsilon_task['params']['async-cycle'] = 1
#		tasks.append(deepcopy(epsilon_task))
#
#
#	for i in xrange(9):
#		epsilon_task['params']['admm'] = 1
#		epsilon_task['params']['rho'] = pow(4, -1)
#		epsilon_task['params']['loss'] = 1
#		tasks.append(deepcopy(epsilon_task))

#	epsilon_task['params']['admm'] = 1
#	epsilon_task['params']['rho'] = 1
#	epsilon_task['params']['loss'] = 1
#	tasks.append(deepcopy(epsilon_task))

#	yandex_ad_task['params']['async-cycle'] = 0
#	tasks.append(deepcopy(yandex_ad_task))
	
#	yandex_ad_task['params']['async-cycle'] = 1
#	tasks.append(deepcopy(yandex_ad_task))

#	for i in xrange(4):
#		yandex_ad_task['params']['admm'] = 1
#		yandex_ad_task['params']['rho'] = 1
#		yandex_ad_task['params']['loss'] = 1
#		tasks.append(deepcopy(yandex_ad_task))

#		yandex_ad_task['params']['async-cycle'] = 0
#		tasks.append(deepcopy(yandex_ad_task))
	
#		yandex_ad_task['params']['async-cycle'] = 1
#		tasks.append(deepcopy(yandex_ad_task))

#	for i in xrange(4):
#		for jobcount in [64, 32, 16, 8, 4, 2, 1]:
#			yandex_ad_task['jobcount'] = jobcount
#			yandex_ad_task['params']['iterations'] =  jobcount / 2 + 5
#			tasks.append(deepcopy(yandex_ad_task))

#	tasks.append(deepcopy(yandex_ad_task))
#	for i in xrange(1):
#		tasks.append(deepcopy(yandex_ad_l2_task))

#	tasks.append(yandex_ad_scad_task)
#	epsilon_task['params']['admm'] = 1
#	epsilon_task['params']['rho'] = pow(4, -1)
#	epsilon_task['params']['loss'] = 1
#1
	for jobcount in [64, 32, 16, 8, 4, 2, 1]:
		yandex_ad_l2_task['jobcount'] = jobcount
		yandex_ad_l2_task['params']['iterations'] = jobcount / 3 + 5
		tasks.append(deepcopy(yandex_ad_l2_task))

#	tasks.append(webspam_l2_task)

	for task in tasks:
		process_task(task, should_create_reports = True)
