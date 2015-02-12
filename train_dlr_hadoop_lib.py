#!/usr/bin/env python
import os
import sys
import socket
import datetime
from time import sleep
from random import randint
from optparse import OptionParser
from socket import gethostname
from math import sqrt
import re
from copy import deepcopy
from tempfile import NamedTemporaryFile

from test_dlr_dist import local_dist_run
from test_dlr import get_models_metrics

'''
------------------------------------------------------------------

A wrapper for DLR training on a Hadoop cluster.

----------------------------------------------------------------------
'''

SPANNING_TREE_SERVER = '141.8.180.50'  # w050h

def execute(cmd):
	''' Simple wrapper for bash scripts '''
	print cmd
	return os.system(cmd)

def create_reducer(jobcount, features, head = False, initial = None, out_dir = None):
	''' Creates a text of reducer with given parameters '''

	if head:
		reducer = '''
cat | head -n 1000 > train;
'''
	else:
		reducer = '''
cat > train;
'''
	if initial:
		reducer += '''
mv %s initial;
''' % os.path.split(initial)[1]

	reducer += '''
nreducers=$mapreduce_job_reduces;
reducer_id=`echo $mapreduce_task_id | cut -d "_" -f 5`;
mapred_job_id=`echo "$mapreduce_job_id" | awk -F "_" '{print $NF}'`;
output_dir=$mapreduce_output_fileoutputformat_outputdir;

echo $reducer_id > /dev/stderr;
echo $nreducers > /dev/stderr;
echo $mapred_job_id > /dev/stderr;

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.;
gunzip -f label.tmp.gz;

uname -n > log;
./dlr -d train -l label.tmp -f model FEATURES SERVER_PARAMS INITIAL 1>> log 2>&1;

if ls *model* 1> /dev/null 2>&1 ; then
	tar cfz models.tar.gz *model*;
fi;

hdfs dfs -put -f models.tar.gz $output_dir/$reducer_id.models.tar.gz;
hdfs dfs -put -f log $output_dir/$reducer_id.log;
'''

	SERVER_PARAMS = '--server HOST --total $nreducers --unique-id $mapred_job_id --node $reducer_id'

	reducer = reducer.replace('SERVER_PARAMS', SERVER_PARAMS)
	reducer = reducer.replace('HOST', SPANNING_TREE_SERVER)
	reducer = reducer.replace('JOBCOUNT', str(jobcount))
	reducer = reducer.replace('FEATURES', features)
	if initial:
		reducer = reducer.replace('INITIAL', '-i initial')
	else:
		reducer = reducer.replace('INITIAL', '')

	return reducer

def table_exists(table):
	''' Checks is table exists'''

	return os.system("hdfs dfs -test -e %s" % table) == 0

def train_dlr(src_tables, label_table, jobcount, features, model, debug = 'debug', mapper_args = '', dump_dir = '.', save_dataset = False, head = False, initial = None):

	any_table_exists = False

	for table in src_tables.split(','):
		if table_exists(table):
			any_table_exists = True
			break

	if not any_table_exists:
		print 'ERROR: NO INPUT DATA!'
		return False

	out_dir = 'dlr-tmp/%d' % randint(0, 1000000)
	reducer = create_reducer(jobcount, features, head, initial, out_dir)

	reducer_file = '%d.sh' % randint(0, 1000000)

#	reducer = '/bin/cat'

	file = open('/tmp/%s' % reducer_file, 'w')
	file.write(reducer)
	file.close()

	print 'CREATED REDUCER: /tmp/%s' % reducer_file
	print
	print reducer
	print

	script = '''
./spanning_tree || true;
hadoop fs -rm -r OUT_DIR || true;

hdfs dfs -cat LABEL_TABLE > label.tmp;
gzip -f label.tmp;

hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
    -Dmapred.reduce.tasks=JOBCOUNT \
    -Dmapred.job.name="dlr allreduce" \
    -Dmapred.map.tasks.speculative.execution=true \
    -Dmapred.reduce.tasks.speculative.execution=true \
    -Dmapred.child.java.opts="-Xmx100m" \
    -Dmapred.task.timeout=600000000 \
    -Dmapred.job.map.memory.mb=1000 \
    SRC_TABLES \
    -output OUT_DIR \
    -mapper /bin/cat \
    -reducer REDUCER \
    -file /tmp/REDUCER \
    -file dlr \
    -file label.tmp.gz \
    -file libboost_program_options.so.1.46.1;

rm /tmp/REDUCER;
rm label.tmp.gz;
'''

	src_tables = src_tables.split(',')

	if initial:
		mapper_args += ' -file %s' % initial

	script = script.replace('REDUCER', reducer_file)
	script = script.replace('JOBCOUNT', str(jobcount))
	script = script.replace('SRC_TABLES', '-input ' + ' -input '.join(src_tables))
	script = script.replace('OUT_DIR', out_dir)
	script = script.replace('DUMP_DIR', dump_dir)
	script = script.replace('LABEL_TABLE', label_table)

	print 'RUNNING REDUCERS:'
	print 

	res = execute(script)

	for i in xrange(jobcount):
		log_table = '%s/%06d.log' % (out_dir, i)

		if table_exists(log_table):
			execute('hdfs dfs -cat %s > %s/%06d.log' % (log_table, dump_dir, i))
		else:
			return 0
			
		model_table = '%s/%06d.models.tar.gz' % (out_dir, i)

		if table_exists(model_table):
			execute('hdfs dfs -cat %s > %s/%06d.models.tar.gz' % (model_table, dump_dir, i))
		else:
			return 0

	return res == 0

def get_iter_stat(dump_dir):

	items = os.listdir(dump_dir)

	for item in items:
		if item.endswith('.log'):
			with open(os.path.join(dump_dir, item)) as log_file:
				text = log_file.read()	
				match = re.match('[\s\S]*(Iter\s*Loss[\s\w]*\n[\s*[\d.+-e]*\n]*)', text)
				s = match.group(1)

	return s

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

def read_task(a_dump_dir):
	with open('%s/task' % a_dump_dir, 'r') as f:
		s_task = f.read()
	return eval(s_task)

def read_file(f):
	file = open(f)
	s = file.read()
	file.close()

	return s

def merge_models(files_list, out_file):

	w = None

	for f in files_list:
		file_lines = read_file(f).strip("\n").split("\n")
		features_count = int(file_lines[0])

		weights = map(lambda x : x.split(':'), filter(lambda x : len(x) > 3, file_lines[1:]))
	
		if w is None:
			w = [0.0] * features_count

		for (idx, value) in weights:
			w[int(idx)] = value

	with open(out_file, 'w') as file:

		file.write("solver_type L1R_LR\n")
		file.write("nr_class 2\n")
		file.write("label 1 -1\n")
		file.write("nr_feature %d\n" % (features_count - 1))
		file.write("bias -1\n")
		file.write("w\n")

		for idx in xrange(1, features_count):
			file.write('%s\n' % w[idx])

def create_reports(a_dump_dir, task, calc_metrics = True):

	if task is None:
		task = read_task(a_dump_dir)

	iterations = int(task['params']['iterations'])
	jobcount = int(task['jobcount'])
	test_file = task['test_file']
	distributed = bool(jobcount > 1)

	if calc_metrics:
		for i in xrange(jobcount):
			execute('cd %s; mkdir %d; tar xfz %06d.models.tar.gz -C %d' % (a_dump_dir, i, i, i))
			
		for p in xrange(1000):
			if os.path.isfile('%s/%d/model.000.%03d' % (a_dump_dir, 0, p)):
				merge_models(['%s/%d/model.000.%03d' % (a_dump_dir, i, p) for i in xrange(jobcount)], '%s/model.000.%03d' % (a_dump_dir, p))

		#execute('cd %s; tar xfvz models.tar.gz' % a_dump_dir)
		str_all_metrics, all_metrics = get_models_metrics(a_dump_dir, iterations, test_file)

		with open('%s/test_metrics' % a_dump_dir, 'w') as f:
			f.write(str_all_metrics)

	with open('%s/log_metrics' % a_dump_dir, 'w') as f:
		f.write(get_iter_stat(a_dump_dir))

	execute('cat %s/log_metrics | tabmap \'$Iter=int($Iter)\'> %s/log_metrics2' % (a_dump_dir, a_dump_dir))
	execute('cat %s/test_metrics | tr " " "\\t" | tabmap --prepend \'$Iter||=0; $Iter++\' > %s/test_metrics2' % (a_dump_dir, a_dump_dir))
	execute('tabjoin --file1 %s/log_metrics2 --file2 %s/test_metrics2 --key Iter > %s/joined_stat' % (a_dump_dir, a_dump_dir, a_dump_dir))

	etalon_loss = task['etalon_loss']

	with NamedTemporaryFile() as tmpfile:
		execute('cp %s/joined_stat %s' % (a_dump_dir, tmpfile.name))
		execute('cat %s | tabmap \'$RelLoss = ($Loss - %f) / %f; $CoeffsNormAlpha = $CoeffsNorm * $Alpha\' | tabsort -g -k Iter > %s/joined_stat' % (tmpfile.name, etalon_loss, etalon_loss, a_dump_dir))

	with open('%s/time_summary' % a_dump_dir, 'w') as f:
		print >>f, get_time_summary(a_dump_dir, int(iterations))

	print 'Created dump ' + a_dump_dir

def process_task(task, should_create_reports = True):

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
					 mapper_args = '', head = 'head' in task, initial = task.get('initial-regressor', None)):
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

	if train_ok and should_create_reports:
		try:
			create_reports(a_dump_dir, task)
		except:
			pass
