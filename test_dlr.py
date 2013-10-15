#!/usr/bin/env python
import os
import sys
from marina.base_metrics import get_metrics
from multiprocessing import Process, Pipe

def execute(cmd):
	sys.stdout.write(cmd + '\n')
	os.system(cmd)

def read_file(file_name):
	file = open(file_name)
	text = file.read()
	file.close()
	return text

def get_lines_count(file_name):
	execute('wc -l %s > tmp.wordcount' % file_name)
	return int(read_file('tmp.wordcount').split()[0])

def get_clicks(file_name):
    clicks = []

    file = open(file_name)
    for line in file:
        click1 = int(line.split(' ')[0])
        clicks.append((click1 + 1) / 2)
    file.close()

    return clicks        

def get_ctr(file_name):
    ctrs = []

    file = open(file_name)
    file.readline()

    for line in file:
        ctr = float(line.split(' ')[1])
        ctrs.append(ctr)
    file.close()

    return ctrs      

def get_metrics_liblinear(test_file, model_file, pred_file, metrics):
	execute('~/predict -b 1 %s %s %s' % (test_file, model_file, pred_file))
	clicks = get_clicks(test_file)
	CTR = get_ctr(pred_file)

	return get_metrics(clicks, CTR, metrics)

def get_features_count(tmp_liblinear_model):
	count = 0

	file = open(tmp_liblinear_model)
	for i in xrange(6):
		file.readline()

	for line in file:
		if float(line) != 0.0:
			count += 1

	file.close()

	return count

class AsyncFunc(Process):
	''' Class is dedicated for asynchonous running mapreduce process with bash-script'''
	
	__metrics = None

	def __init__(self, func, child_con):
		self.__func = func
		Process.__init__(self)
		self.__child_con = child_con

	def run(self):
		exec(self.__func)		
		self.__child_con.send((metrics, features_count))

def get_models_metrics(dir_models, max_model, test_file):

	metrics = ['MSE', 'NLL', 'LinCorr', 'auPRC', 'Alpha']

	data = 'id test_LinCorr test_auPRC test_NLL test_Alpha features\n'
	funcs = [None] * (max_model + 1)
	parent_cons = [None] * (max_model + 1)
	block_size = 10

	all_test_metrics = []

	for j in xrange(0, max_model / block_size + 1):
		begin = j * block_size + 1
		end = (j + 1) * block_size + 1	

		if end > max_model + 1:
			end = max_model + 1

		for i in xrange(begin, end):
			model_file = '%s/model%d' % (dir_models, i)
			pred_file = '%s/tmp_pred%d' % (dir_models, i)

			parent_cons[i], con = Pipe()
			cmd = 'metrics = get_metrics_liblinear("%s", "%s", "%s", %s); features_count = get_features_count("%s")' % (test_file, model_file, pred_file, str(metrics), model_file)
			funcs[i] = AsyncFunc(cmd, con)
			funcs[i].start()		

		for i in xrange(begin, end):
			funcs[i].join()

		for i in xrange(begin, end):
			test_metrics, features_count = parent_cons[i].recv()

			data += '%d %f %f %f %f %d\n' % \
				(i, test_metrics['LinCorr'], test_metrics['auPRC'], test_metrics['NLL'], test_metrics['Alpha'], features_count)

			all_test_metrics += [[i, test_metrics]]

	return data, all_test_metrics
