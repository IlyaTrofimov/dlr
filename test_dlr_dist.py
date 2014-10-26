#!/usr/bin/env python
import os
import sys

from marina.base_metrics import get_metrics

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
	execute('./predict -b 1 %s %s %s' % (test_file, model_file, pred_file))
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

def split_dataset(file_name, parts):
	in_file = open(file_name)
	out_files = [None] * parts

	for i in xrange(parts):
		out_files[i] = open('%s.part%d_%d' % (file_name, i, parts), 'w')

	for line in in_file:
		feature_id = int(line.split(' ')[0])
		idx = feature_id % parts
		out_files[idx].write(line)

	in_file.close()

	for i in xrange(parts):
		out_files[i].close()

def local_dist_run(train_file, label_file, test_file, jobcount, features, dump_dir):

	server = '178.154.194.56'

	metrics = ['MSE', 'NLL', 'LinCorr', 'auPRC', 'Alpha']

	learn_id = 1
	data = 'id L1 test_LinCorr test_auPRC test_NLL test_Alpha eval_LinCorr eval_auPRC eval_NLL eval_Alpha features\n'

	for nodes in [jobcount]:
		execute('rm *.cache')
		execute('rm *.part*')

		if nodes != 1:
			split_dataset(train_file, nodes)
			execute('killall dlr')
			execute('./spanning_tree || true')
			execute('sleep 5')

		l1 = 16.0

#		model_file = '%s/rfeatures_%d' % (dump_dir, nodes)
		pred_file = 'tmp_pred'
		cmd = ''

		for i in xrange(nodes):
			if nodes == 1:
				cmd += 'cat %s | ./dlr -d /dev/stdin -f %s --lambda-1 %.1f --termination 1.0e-6 > %d_%d.log' % (train_file, model_file, l1, i, nodes)
			else:
				train_file_part = '%s.part%d_%d' % (train_file, i, nodes)

				#opt_args = '--lambda-1 %.1f --termination 0.0 --iterations 1 --beta-max 10 --combine-type 0' % (l1)
				
				if i == 0:	
					model_file = '-f %s/model' % dump_dir
				else:
					model_file = ''

				cmd += '(sleep % 2d; ./dlr -d %s -l %s %s %s --server %s --node %d --total %d --unique-id 10 1> %s/%d_%d.log 2>&1)' \
						% (i + 1, train_file_part, label_file, model_file, features, server, i, nodes, dump_dir, i, nodes)


				if i < nodes - 1:
					cmd += ' & \n'
		
		execute(cmd)

#		test_metrics = get_metrics_liblinear(test_file, model_file, pred_file, metrics)

#		features_count = get_features_count(model_file)

#		data += '%d %f %f %f %f %f %d\n' % \
#			(learn_id, l1, \
#			test_metrics['LinCorr'], test_metrics['auPRC'], test_metrics['NLL'], test_metrics['Alpha'], features_count)

		learn_id += 1

		if learn_id > 100000:
			print data
			sys.exit()

#	print data

if __name__ == '__main__':

	local_dist_run('small_train_ro_0.1.ii2', 'small_train_ro_0.1.label', 'small_test_ro_0.1.svm', 3, '--lambda-1 64.0 --save-per-iter 1 --linear-search 1 --combine-type 0', 'admm_dump')
