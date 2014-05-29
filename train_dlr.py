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

from test_dlr_dist import local_dist_run
from test_dlr import get_models_metrics

'''
------------------------------------------------------------------

A wrapper for VW training on a MapReduce cluster.

----------------------------------------------------------------------
'''

TEST = False
SPANNING_TREE_SERVER = '141.8.134.1' 

def execute(cmd):
	''' Simple wrapper for bash scripts '''
	print cmd
	os.system(cmd)

def create_reducer(jobcount, port, features, save_dataset = False, head = False, initial = None):
	''' Creates a text of mapper with given parameters '''

	if head:
		reducer = '''
more | head -n 1000 > train;
'''
	else:
		reducer = '''
more > train;
'''
	reducer += '''
gunzip -f label.tmp.gz;
jobid=`id_client.py HOST PORT start`;
./dlr -d train -l label.tmp -f model FEATURES SERVER_PARAMS INITIAL 1> log;
node=`uname -n`;
touch model_empty;
tar cfz models.tar.gz *model*;
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
	reducer = reducer.replace('HOST', SPANNING_TREE_SERVER)
	reducer = reducer.replace('PORT', str(port))
	reducer = reducer.replace('JOBCOUNT', str(jobcount))
	reducer = reducer.replace('FEATURES', features)
	reducer = reducer.replace('UNIQUE_ID', str(randint(0, 1000000)))
	if initial:
		reducer = reducer.replace('INITIAL', '-i %s' % os.path.split(initial)[1])

	return reducer

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


def train_dlr(src_tables, label_table, jobcount, features, model, debug = 'debug', mapper_args = '', dump_dir = '.', save_dataset = False, head = False, initial = None):

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

	reducer = create_reducer(jobcount, port, features, save_dataset, head, initial)

	reducer_file = '%d.sh' % randint(0, 1000000)

	file = open('/tmp/%s' % reducer_file, 'w')
	file.write(reducer)
	file.close()

	print 'CREATED REDUCER: /tmp/%s' % reducer_file
	print
	print reducer
	print

	script = '''
if [ "$(killall -0 spanning_tree 2>&1)" != "" ]; then
	spanning_tree;
fi;

mapreduce -subkey -read LABEL_TABLE > label.tmp;
gzip -f label.tmp;
mapreduce -reduce ./REDUCER -subkey -file ./dlr -file ./encode_record.py -file /tmp/REDUCER SRC_TABLES -file label.tmp.gz -dst DEBUG_TABLE -dst MODEL_TABLE -opt "OPTIONS" -jobcount JOBCOUNT EXT_ARGS;

rm *.log;
rm *.model.tar.gz;
mapreduce -read DEBUG_TABLE | ./decode_record_value.py DUMP_DIR log;
mapreduce -read MODEL_TABLE | ./decode_record_value.py DUMP_DIR model.tar.gz;'''

	if save_dataset:
		script += '''
mapreduce -read TRAIN_TABLE | ./decode_record_value.py DUMP_DIR train.tar.gz;'''

	script += '''
rm /tmp/REDUCER;'''

	model_tmp = '/tmp/%d' % randint(0, 1000000)

	r = randint(0, 1000000)
	model_table = 'tmp/dlr/%d-model' % r
	debug_table = 'tmp/dlr/%d-debug' % r

	src_tables = src_tables.split(',')

	if save_dataset:
		mapper_args = mapper_args + ' -dst users/trofim/genkin/dlr-train'

	if initial:
		mapper_args += ' -file %s' % initial

	script = script.replace('REDUCER', reducer_file)
	script = script.replace('JOBCOUNT', str(jobcount))
	script = script.replace('SRC_TABLES', '-src ' + ' -src '.join(src_tables))
	script = script.replace('MODEL_TABLE', model_table)
	script = script.replace('DEBUG_TABLE', debug_table)
	script = script.replace('TRAIN_TABLE', debug_table)
	script = script.replace('LABEL_TABLE', label_table)
	script = script.replace('OPTIONS', 'threadcount=1')
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

		print '--------'
		print 'MODEL OK'
		print '--------'
		return True

	# some reducers restarted
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
				match = re.match('[\s\S]*(Iter\s*Loss[\s\w]*\n[\s*[\d.+-e]*\n]*)', text)
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
		if i + 1 < len(iter_stat_lines):
			joined_lines.append(iter_stat_lines[i + 1].strip(' ') + ' ' + test_metrics_lines[i].strip(' '))

	for j in xrange(len(joined_lines)):
		joined_lines[j] = joined_lines[j].replace(' ', '\t')

	return '\n'.join(joined_lines)

def write_task_description(a_dump_dir, task):
	with open('%s/task' % a_dump_dir, 'w') as f:
		print >>f, str(task)

def create_reports(a_dump_dir, task):

	iterations = int(task['params']['iterations'])
	jobcount = int(task['jobcount'])
	distributed = bool(jobcount > 1)
	test_file = task['test_file']

	execute('cd %s; tar xfvz *.model.tar.gz' % a_dump_dir)
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

	if 'report_only' in task:
		create_reports(task['report_only'], task)
		sys.exit()

	train_tables = task['train_tables']
	label_table = task['label_table']
	test_file = task['test_file']
	iterations = int(task['params']['iterations'])
	jobcount = int(task['jobcount'])

	features = ''
	for feature, value in task['params'].iteritems():
		features += '--%s %s ' % (feature, str(value))

	if 'dump_dir' in task:
		a_dump_dir = task['dump_dir']
	else:
		a_dump_dir = None

		for i in xrange(100):	
			a_dump_dir = './dump-%s-%d' % (datetime.datetime.now().strftime('%Y%m%d'), i)

			if not os.path.isdir(a_dump_dir):
				os.makedirs(a_dump_dir)
				break

	if a_dump_dir is None:
		sys.exit()

	write_task_description(a_dump_dir, task)
	train_ok = False

	if 'local-dist-test' in task:
		local_dist_run(train_tables, label_table, test_file, jobcount, features, a_dump_dir)
		train_ok = True

	elif (jobcount > 1):   # distributed
		for count in xrange(5):
			print 'Attempt ', count
			if train_dlr(train_tables, label_table, jobcount, features, 'model', dump_dir = a_dump_dir, save_dataset = task.get('save_dataset', False), \
					 mapper_args = '-memlimit 1', head = 'head' in task, initial = task.get('initial-regressor', None)):
				train_ok = True
				break
	else:
		execute('rm *.cache');

		if 'local-dataset' in task:
			execute('./dlr -d %s -l %s %s -f %s/model 1> %s/local.log 2>&1' % (train_tables, label_table, features, a_dump_dir, a_dump_dir))
		else:
			head = '| head -n 1000' if 'head' in task else ''
				
			execute('mapreduce -subkey -read %s > label.tmp' % (label_table))
			execute('mapreduce -subkey -read %s %s | ./dlr -d /dev/stdin -l label.tmp %s -f %s/model 1> %s/local.log 2>&1' % (train_tables, head, features, a_dump_dir, a_dump_dir))
			
		train_ok = True

	if train_ok:
		create_reports(a_dump_dir, task)

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
#	"yandex_ad2" dataset
#----------------------------------------------------

	base_task = {
		'train_tables':		'users/trofim/genkin/yandex_ad2.train.ii2',
		'label_table':		'users/trofim/genkin/yandex_ad2.train.label',
		'test_file':		'/mnt/raid/home/trofim/genkin/yandex_ad2.test.svm',
		'jobcount':		16,
		'params': {
			'iterations':		32,
			'lambda-1':		16.0,
			'termination':		0.0,
			'combine-type':		0,
			'save-per-iter':	1,
#			'beta-max':		10,
			},
		'etalon_loss': 7.10e7,
		'save_dataset': False,
		}

	task = deepcopy(base_task)
	task['params']['combine-type'] = 0;
	task['params']['find-bias'] = 1;
	task['params']['linear-search'] = 1;
	task['params']['last-iter-sum'] = 1;
	task['params']['termination'] = 1.0e-3;
#	task['params']['lambda-path'] = 10
	task['params']['lambda-1'] = get_lambda_list([22229.60547, 11114.80273, 5557.401367, 2778.700684, 1389.350342, 694.6751709, 347.3375855, 173.6687927, 86.83439636, 43.41719818])
	task['initial-regressor'] = '/mnt/raid/home/trofim/dlr/dump-20140527-0/model.009.002'
#	task['params']['iterations'] = 10;
#	task['params']['lambda-1'] = get_lambda_list([pow(2, x) for x in xrange(10, -1, -1)])
	tasks.append(task)

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
		'train_tables':		'users/trofim/genkin/epsilon-train.ii2',
		'label_table':		'users/trofim/genkin/epsilon-train.label',
		'test_file':		'/mnt/raid/home/trofim/genkin/epsilon_normalized.t',
		'jobcount':		16,
		'params': {
			'iterations':		30,
#			'lambda-1':		2.0,
			'termination':		0.0,
			'combine-type':		0,
			'save-per-iter':	1,
			'beta-max':		100,
			},
		'etalon_loss': 105958.76,
		}

	task = deepcopy(base_task)
	task['params']['termination'] = 1.0e-3
	task['params']['linear-search'] = 1;
	task['params']['last-iter-sum'] = 1;
#	task['params']['lambda-1'] = get_lambda_list([40.0, 32.0, 16.0, 8.0, 4.0, 3.2, 1.6])
	task['params']['lambda-path'] = 20

#	tasks.append(task)
#
#----------------------------------------------------
#	"webspam" dataset
#----------------------------------------------------
#
	base_task = {
		'train_tables':		'users/trofim/genkin/webspam.train.ii2',
		'label_table':		'users/trofim/genkin/webspam.train.label',
		'test_file':		'/mnt/raid/home/trofim/dlr/webspam_wc_normalized_trigram.test',
		'jobcount':		16,
		'params': {
			'iterations':		30,
#			'lambda-1':		1 / 64.0,
			'termination':		0.0,
			'combine-type':		0,
			'save-per-iter':	1,
			'beta-max':		1.0e6,
			},
		'etalon_loss': 2005.81766840625
		}

	task = deepcopy(base_task)
	task['params']['termination'] = 1.0e-3
	task['params']['linear-search'] = 1;
	task['params']['last-iter-sum'] = 1;
#	task['params']['lambda-1'] = get_lambda_list([pow(2, -x) for x in xrange(-2, 12)])
	task['params']['lambda-path'] = 20

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
	task['params']['lambda-1'] = get_lambda_list([pow(2, x) for x in xrange(14, -1, -1)])
#	tasks.append(task)

	task = deepcopy(base_task)
	task['params']['termination'] = 1.0e-3;
	task['params']['last-iter-sum'] = 1;
	task['params']['lambda-1'] = get_lambda_list([pow(2, x) for x in xrange(10, -1, -1)])
#	tasks.append(task)

	for task in tasks:
		process_task(task)
