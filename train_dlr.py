#!/usr/bin/env python
import os
import sys
import socket
import datetime
from time import sleep
from random import randint
from optparse import OptionParser
from socket import gethostname
from multiprocessing import Process, Pipe
from math import sqrt
import re
from copy import deepcopy
from tempfile import NamedTemporaryFile

from test_dlr import get_models_metrics

'''
------------------------------------------------------------------

A wrapper for VW training on a MapReduce cluster.

----------------------------------------------------------------------
'''

TEST = False

def execute(cmd):
	''' Simple wrapper for bash scripts '''
	print cmd
	os.system(cmd)

def create_reducer(jobcount, port, features, save_dataset = False):
	''' Creates a text of mapper with given parameters '''

	reducer = '''
more > train;
jobid=`id_client.py HOST PORT start`;
#head -n 1000 train > head_train;
./dlr -d train -f model FEATURES SERVER_PARAMS 1> log 2>&1;
node=`uname -n`;
tar cfz models.tar.gz model*;
echo 0;
./encode_record.py $node log;
echo 1;
if [ $jobid == 0 ]; then
	./encode_record.py $node models.tar.gz;
fi;
'''

	if save_dataset:
		reducer += '''
tar cfz train.tar.gz train;
echo 2;
./encode_record.py $node train.tar.gz;
'''

	reducer += '''
id_client.py HOST PORT finish;
'''

	SERVER_PARAMS = '--server HOST --total JOBCOUNT --unique-id UNIQUE_ID --node $jobid'

	reducer = reducer.replace('SERVER_PARAMS', SERVER_PARAMS)
	reducer = reducer.replace('HOST', '141.8.172.125')
	reducer = reducer.replace('PORT', str(port))
	reducer = reducer.replace('JOBCOUNT', str(jobcount))
	reducer = reducer.replace('FEATURES', features)
	reducer = reducer.replace('UNIQUE_ID', str(randint(0, 1000000)))

	return reducer

def check_models(file_name, jobcount):
	''' Checks that models, downloaded from MR are correct: equal and their number equals jobcount'''

	models = {}
	new_model = ''
	ok = True

	file = open(file_name)
	for line in file:
		new_model = line.rstrip('\n').split('\t')[1]
		part = int(new_model[0:3])

		if (part in models):
			print 'Error: found duplicate part'
			ok = False
			break
		else:
			models[part] = new_model

	file.close()

	if len(models) == 0:
		print 'Error: model is empty'
		ok = False

	return ok

class VWTrainer(Process):
	''' Class is dedicated for asynchonous running mapreduce process with bash-script'''

	def __init__(self, script):
		self.__script = script
		Process.__init__(self)

	def run(self):
		execute(self.__script)

class IDServer(Process):
	''' Class is dedicated for asynchonous running ID server for controlling mappers'''

	def __init__(self, jobcount, child_con):
		self.__jobcount = jobcount
		self.__child_con = child_con
		Process.__init__(self)
	
	def run(self):
		print 'ID SERVER STARTED'

		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		s.bind(('', 0))
		print 'HOST = %s PORT = %d' % (s.getsockname()[0], s.getsockname()[1])
		print

		port = s.getsockname()[1]
		self.__child_con.send(port)

		s.listen(self.__jobcount)
		status = False

		start_count = 0
		finish_count = 0

		while start_count < 1000:
			conn, addr = s.accept()
			message = conn.recv(1024)

			if message == 'start':
				print 'Connected by %s, jobid %d, %s' % (addr, start_count, message)
				conn.send(str(start_count))
				start_count += 1
			else:
				finish_count += 1
				conn.send('0')
				print 'Connected by %s, finish_count %d' % (addr, finish_count)
			
			conn.close()

			if start_count > self.__jobcount:
				print 'Error: start_count > jobcount'
				status = False
				break

			if finish_count == self.__jobcount:
				status = True
				break
			
		s.close()
		self.__child_con.send(status)

def table_exists(table):
	''' Checks is table exists'''

	tmp_file = '/tmp/%d' % randint(0, 1000000)
	execute('mapreduce -read %s | head -n 1 > %s' % (table, tmp_file))

	file = open(tmp_file)
	text = file.read()
	file.close()

	execute('rm %s' % tmp_file)

	return len(text) > 0

def py_execute(cmd):
	''' Execute bash script and returns stdout '''

	tmp_file = '/tmp/%d' % randint(0, 1000000)
	execute('%s > %s' % (cmd, tmp_file))
	file = open(tmp_file)
	res = file.read()
	file.close()
	execute('rm %s' % tmp_file)

	return res

def kill_children(pid):
	''' Kill recursively all child processes of a process '''

	res = py_execute('ps --ppid %d' % pid)
	children_lines = res.split('\n')[1:]

	for line in children_lines:
		if len(line) > 0:
			child_pid = int(line.strip(' ').split(' ')[0])
			kill_children(child_pid)
			execute('kill %d' % child_pid)


def train_dlr(src_tables, jobcount, features, model, debug = 'debug', mapper_args = '', dump_dir = '.', save_dataset = False):

	any_table_exists = False

	for table in src_tables.split(','):
		if table_exists(table):
			any_table_exists = True
			break

	if not any_table_exists:
		print 'ERROR: NO INPUT DATA!'
		return False

	id_server_con, con = Pipe()

	id_server = IDServer(jobcount, con)
	id_server.start()
	
	port = id_server_con.recv()

	reducer = create_reducer(jobcount, port, features, save_dataset)

	reducer_file = '%d.sh' % randint(0, 1000000)

	file = open('/tmp/%s' % reducer_file, 'w')
	file.write(reducer)
	file.close()

	print 'CREATED REDUCER: /tmp/%s' % reducer_file
	print
	print reducer
	print

	script = '''\
if [ "$(killall -0 spanning_tree 2>&1)" != "" ]; then
	spanning_tree;
fi;
 
mapreduce -reduce ./REDUCER -subkey -file ./dlr -file ./encode_record.py -file /tmp/REDUCER SRC_TABLES -dst DEBUG_TABLE -dst MODEL_TABLE -opt "OPTIONS" -jobcount JOBCOUNT EXT_ARGS;

rm *.log;
rm *.model.tar.gz;
mapreduce -read DEBUG_TABLE | ./decode_record_value.py DUMP_DIR log;
mapreduce -read MODEL_TABLE | ./decode_record_value.py DUMP_DIR model.tar.gz;

rm /tmp/REDUCER;'''

	model_tmp = '/tmp/%d' % randint(0, 1000000)

	r = randint(0, 1000000)
	model_table = 'tmp/dlr/%d-model' % r
	debug_table = 'tmp/dlr/%d-debug' % r

	src_tables = src_tables.split(',')

	if save_dataset:
		mapper_args = mapper_args + ' -dst users/trofim/genkin/dlr-train'

	script = script.replace('REDUCER', reducer_file)
	script = script.replace('JOBCOUNT', str(jobcount))
	script = script.replace('SRC_TABLES', '-src ' + ' -src '.join(src_tables))
	script = script.replace('MODEL_TABLE', model_table)
	script = script.replace('DEBUG_TABLE', debug_table)
	script = script.replace('OPTIONS', 'box=developers,tier=2,threadcount=1')
	script = script.replace('MODEL_TMP', model_tmp)
	script = script.replace('MODEL_FILE', model)
	script = script.replace('EXT_ARGS', mapper_args)
	script = script.replace('DUMP_DIR', dump_dir)

	print 'RUNNING REDUCERS:'
	print 

	vw_trainer = VWTrainer(script)
	vw_trainer.start()

	res = id_server_con.recv()

	# mappers didn't restart
	if res: 
		vw_trainer.join()

		print
		print 'DLR DEBUG:'
		print

		if True or check_models(model_tmp, jobcount):
			print '--------'
			print 'MODEL OK'
			print '--------'
			return True
		else:
			print '-----------'
			print 'MODEL ERROR'
			print '-----------'
			return False

		execute('rm %s;' % model_tmp)

	# some mapper restarted
	else:
		kill_children(vw_trainer.pid)
		vw_trainer.terminate()
		id_server.terminate()

		print '---------'
		print 'JOB ERROR'
		print '---------'
		return False

def get_iter_stat(dump_dir):

	items = os.listdir(dump_dir)

	for item in items:
		if item.endswith('.log'):
			with open(os.path.join(dump_dir, item)) as log_file:
				text = log_file.read()	
				match = re.match('[\s\S]*(Iter\s*Loss[\s\S]*)', text)
				return match.group(1)

def sigma(v):
	avg = 0.0
	for i in xrange(len(v)):
		avg += v[i]

	avg /= len(v)

	sigma = 0.0
	for i in xrange(len(v)):
		sigma += (avg - v[i]) ** 2

	sigma /= len(v)
	sigma = sqrt(sigma)

	return sigma

def median(v):
	return sorted(v)[len(v) / 2]

def get_time_summary(dump_dir, iterations):

	files = os.listdir(dump_dir)
	times = {}
	times_full = {}
	count = 0

	for a_file in files:
		if a_file.endswith('.log'):
			count += 1
			with open(os.path.join(dump_dir, a_file)) as log_file:
				text = log_file.read()	
				match = re.match('[\s\S]*Time summary:([\s\S]*)Iter[\s\S]*', text)
				text = match.group(1)

			for line in text.split('\n'):
				match = re.match('\s*([\s\S]*):\s*(\d+)s[\s\S]*', line)
				if match:
					name = match.group(1).strip()
					time = float(match.group(2))
	
					if name in times:
						times[name] = times[name] + time
						times_full[name].append(time)
					else:
						times[name] = time
						times_full[name] = [time]

	full_time = 0.0	

	s = ''
	s += '# Operation\tAvgNode\tRelStdNode\tRelMax\tAvgIteration\n'
	for name in sorted(times.iterkeys()):
		avg_node_time = times[name] / count
		avg_iter_time = times[name] / count / iterations
		sigma_node_time = sigma(times_full[name])

		if avg_node_time:
			rel_std = (sigma_node_time / avg_node_time)
		else:
			rel_std = 0.0
	
		a_median = median(times_full[name])

		if a_median:
			rel_max = (max(times_full[name]) - a_median) / a_median
		else:
			rel_max = 0.0

		s += '\t'.join(map(str, [name, avg_node_time, '%.2f' % rel_std, '%.2f' % rel_max, avg_iter_time])) + '\n'
		full_time += times[name]

	s += '\n'
	s += 'found log files %s\n' % count	
	s += 'full time %f \n' % full_time
	s += 'avg iteration %f \n' % (full_time / count / iterations)
	
	return s

def get_test_metrics(dump_dir):

	with open(os.path.join(dump_dir, 'test_metrics')) as log_file:
		text = log_file.read()	

	return text

def join_metrics(iter_stat, test_metrics):
	iter_stat_lines = iter_stat.split('\n')
	test_metrics_lines = test_metrics.split('\n')
	joined_lines = []

	joined_lines.append(iter_stat_lines[0].strip(' ') + ' ' + test_metrics_lines[0].strip(' '))
	joined_lines.append(iter_stat_lines[1].strip(' '))

	for i in xrange(1, len(test_metrics_lines) - 1):
		joined_lines.append(iter_stat_lines[i + 1].strip(' ') + ' ' + test_metrics_lines[i].strip(' '))

	for j in xrange(len(joined_lines)):
		joined_lines[j] = joined_lines[j].replace(' ', '\t')

	return '\n'.join(joined_lines)

def write_task_info(a_dump_dir, task, test_file):
	with open('%s/task' % a_dump_dir, 'w') as f:
		print >>f, str(task)

def create_reports(a_dump_dir, task, test_file):

	iterations = int(task['params']['iterations'])
	jobcount = int(task['jobcount'])
	distributed = bool(jobcount > 1)

	if distributed:
		str_all_metrics, all_metrics = get_models_metrics(a_dump_dir, iterations, test_file)
	else:
		str_all_metrics, all_metrics = get_models_metrics(a_dump_dir, iterations, test_file)

	with open('%s/test_metrics' % a_dump_dir, 'w') as f:
		print >>f, str_all_metrics

	with open('%s/joined_stat' % a_dump_dir, 'w') as f:
		iter_stat = get_iter_stat(a_dump_dir)
		test_metrics = get_test_metrics(a_dump_dir)
		print >>f, join_metrics(iter_stat, test_metrics)

	etalon_loss = task['etalon_loss']

	with NamedTemporaryFile() as tmpfile:
		execute('cp %s/joined_stat %s' % (a_dump_dir, tmpfile.name))
		execute('cat %s | tabmap \'$RelLoss = ($Loss - %f) / %f; $CoeffsNormAlpha = $CoeffsNorm * $Alpha\' > %s/joined_stat' % (tmpfile.name, etalon_loss, etalon_loss, a_dump_dir))

	with open('%s/time_summary' % a_dump_dir, 'w') as f:
		print >>f, get_time_summary(a_dump_dir, int(iterations))

	print 'Created dump ' + a_dump_dir

def process_task(task):

	train_tables = task['train_tables']
	test_file = task['test_file']
	iterations = int(task['params']['iterations'])
	jobcount = int(task['jobcount'])

	features = ''
	for feature, value in task['params'].iteritems():
		features += '--%s %s ' % (feature, str(value))

	distributed = bool(jobcount > 1)
	a_dump_dir = None

	for i in xrange(100):	
		a_dump_dir = './dump-%s-%d' % (datetime.datetime.now().strftime('%Y%m%d'), i)

		if not os.path.isdir(a_dump_dir):
			os.makedirs(a_dump_dir)
			break

	if a_dump_dir is None:
		sys.exit()

	write_task_info(a_dump_dir, task, test_file)

	if distributed:
		train_dlr(train_tables, jobcount, features, 'model', dump_dir = a_dump_dir, save_dataset = task.get('save_dataset', False))
		execute('cd %s; tar xfvz *.model.tar.gz' % a_dump_dir)
	else:
		execute('rm *.cache');
		execute('mapreduce -subkey -read %s | ./dlr -d /dev/stdin %s -f %s/model 1> %s/local.log 2>&1' % (train_tables, features, a_dump_dir, a_dump_dir))

	create_reports(a_dump_dir, task, test_file)

if __name__ == '__main__':

	tasks = []
#
#----------------------------------------------------
#	"yandex_ad2" dataset
#----------------------------------------------------

	base_task = {
		'train_tables':		'users/trofim/genkin/yandex_ad2.train.ii',
		'test_file':		'/mnt/raid/home/trofim/genkin/yandex_ad2.test.svm',
		'jobcount':		16,
		'params': {
			'iterations':		20,
			'lambda-1':		128.0,
			'termination':		0.0,
			'combine-type':		1,
			'save-per-iter':	1,
			'beta-max':		100,
			},
		'etalon_loss': 105958.76
		}

	for a in map(lambda x : pow(2, x), range(0, 9)):
		task = deepcopy(base_task)
		task['params']['lambda-1'] = a	
		tasks.append(task)

#
#----------------------------------------------------
#	"epsilon" dataset
#----------------------------------------------------

#	base_task = {
#		'train_tables':		'users/trofim/genkin/epsilon-train.ii',
#		'test_file':		'/mnt/raid/home/trofim/genkin/epsilon_normalized.t',
#		'jobcount':		16,
#		'params': {
#			'iterations':		50,
#			'lambda-1':		2.0,
#			'termination':		0.0,
#			'combine-type':		1,
#			'save-per-iter':	1,
#			'beta-max':		100,
#			'random-count':		2
#			},
#		'etalon_loss': 105958.76,
#		}
#
#	task = deepcopy(base_task)
#	task['jobcount'] = 1
#	tasks.append(task)

#
#----------------------------------------------------
#	"webspam" dataset
#----------------------------------------------------
#
#	base_task = {
#		'train_tables':		'users/trofim/genkin/webspam.train.ii',
#		'test_file':		'/mnt/raid/home/trofim/dlr/webspam_wc_normalized_trigram.test',
#		'jobcount':		16,
#		'params': {
#			'iterations':		200,
#			'lambda-1':		1 / 64.0,
#			'termination':		0.0,
#			'combine-type':		0,
#			'save-per-iter':	1,
#			'beta-max':		100
#			},
#		'etalon_loss': 2005.81766840625
#		}

#	for combine_type in [0, 1, 3]:
#		task = deepcopy(base_task)
#		task['params']['combine-type'] = combine_type
#		tasks.append(task)

#	task = deepcopy(base_task)
#	task['jobcount'] = 1  	# local
#	task['params']['iterations'] = 50 
#	tasks.append(task)

#	'train_tables':		'users/trofim/genkin/train_set.ii',
#	'test_file':		'/mnt/raid/home/trofim/dlr/test_set1.normed.svm',

	for task in tasks:
		process_task(task)
