#!/usr/bin/env python
import os
from math import exp, log, sqrt
from math import log

def mult(a, b):
	return a * b

def avg(array):
	if len(array) > 0:
		return sum(array, 0.0) / len(array)
	else:
		return 0.0

def cov(d1, d2):
	avg1 = avg(d1)
	avg2 = avg(d2)

	cov = 0.0

	for i in xrange(len(d1)):
		cov += (d1[i] - avg1) * (d2[i] - avg2)

	return cov

def inner_prod(a, b):
	res = 0.0

	for i in xrange(len(a)):
		res += a[i] * b[i]

	return res

def get_alpha(clicks, CTR):
	nominator = inner_prod(clicks, CTR)
	denominator = inner_prod(CTR, CTR)

	return (nominator / denominator) if (denominator > 1.0e-6) else 0.0

def get_linear_corr_from_array(clicks, CTR):
	nominator = cov(clicks, CTR)
	denominator = sqrt(cov(clicks, clicks) * cov(CTR, CTR))

	return (nominator / denominator) if (denominator > 1.0e-6) else 0.0

def GetMSEfromArray(clicks, CTR):
	mse = 0.0
	for i in range(0, len(clicks)):
		click = 1.0 if (int(clicks[i]) == 1) else 0.0
		mse += (click - CTR[i]) ** 2

	return mse / len(clicks)

def GetLLfromArray(clicks, CTR) :
	loglike = 0.0

	for i in range(0, len(clicks)):

		ctr = CTR[i]
		if ctr < 1.0e-6:
			ctr = 1.0e-6
		if ctr > (1.0 - 1.0e-6):
			ctr = (1.0 - 1.0e-6)

		if (int(clicks[i]) == 1) :
			loglike += log(ctr)
		else:
			loglike += log(1 - ctr)

	return loglike / len(clicks)

def GetMultiClassLLFromArray(class_values, probabilities):
	data_set_size = len(probabilities)
	multiclass_LL = 0.0
	for idx in xrange(data_set_size):
		class_value = class_values[idx]
		multiclass_LL += log(probabilities[idx][class_value])
	return multiclass_LL / data_set_size

def get_PRC(clicks, CTR):
	'''
	Calculates Precision-Recall statistics.

	returns: [auPRC, plot].
	auPRC - is area under the Precision-Recall curve.
	plot - a plot of [[precision, recall], ...]
	'''

	total_shows = len(clicks)
	total_clicks = 0
	indexes = range(total_shows)
	indexes.sort(lambda x, y: cmp(CTR[x] ,CTR[y]))

	thresholds_count = 32
	bin_clicks = [0] * thresholds_count
	bin_shows  = [0] * thresholds_count

	for i in xrange(total_shows):
		bin = i * thresholds_count / total_shows

		bin_clicks[bin] += 1 if (int(clicks[indexes[i]]) == 1) else 0
		total_clicks    += 1 if (int(clicks[indexes[i]]) == 1) else 0
		bin_shows[bin]  += 1

	auPRC      = 0.0
	sum_clicks = 0.0
	sum_shows  = 0.0

	plot = []

	for bin in xrange(thresholds_count - 1, -1, -1):
		sum_clicks += bin_clicks[bin]
		sum_shows  += bin_shows[bin]

		delta_recall = float(bin_clicks[bin]) / total_clicks
		precision = float(sum_clicks) / sum_shows
		recall = float(sum_clicks) / total_clicks

		auPRC += precision * delta_recall
		plot += [[precision, recall]]

	return [auPRC, plot]

def get_auPRC(clicks, CTR):
	return get_PRC(clicks, CTR)[0]

def get_metrics(clicks, CTR, metrics):

	all_metrics = {'MSE': GetMSEfromArray, 'LL': GetLLfromArray, 'NLL' : lambda x, y: -GetLLfromArray(x, y), \
			'LinCorr': get_linear_corr_from_array, 'Alpha' : get_alpha, 'auPRC' : get_auPRC, \
			'Cor': get_linear_corr_from_array, 'MULTI_LL' : GetMultiClassLLFromArray}
	res = {}

	for metric in metrics:
		res[metric] = all_metrics[metric](clicks, CTR)

	return res

def get_avg_ctr(file_name):
	in_file = open(fil_name)
	line = in_file.readline()

	clicks = 0
	shows = 0

	while len(line) > 0:
		click = int(line.split('\t')[1])

		clicks += click
		shows += 1
	
		line = in_file.readline()

	in_file.close()

	if shows > 0:
		ctr = float(clicks) / shows
	else:
		ctr = 0 

	return ctr

def logit(value):
	if isinstance(value, list) :
		answersSum = sum(value)
		return map(lambda x: x/answersSum, value)
	else :
		return 1 / (1 + exp(-value))

def inv_logit(value):

	MIN = 0.00001
	MAX = 0.99

	value = max(value, MIN)
	value = min(value, MAX)

	return log(value / (1 - value))

def get_regression(file_name, is_logit = True, base = None):
	file = open(file_name)
	line = file.readline()

	clicks = []
	CTR = []
	line_num = 0

	while len(line) > 0:
		fields = line.split('\t')

		classCount = len(fields[4:])
		classCount = 1 if (classCount == 2) else classCount

		clicks += [int(fields[1])]
		value = float(fields[4]) if (classCount == 1) else map(float, [fields[4 + idx] for idx in xrange(classCount)])

		if not (base is None):
			value += base[line_num]

		if is_logit:
			CTR += [logit(value)]
		else:
			CTR += [value]

		line_num += 1
		line = file.readline()

	file.close()

	return [clicks, CTR]

def normalize_ctr(in_file_name, out_file_name):
	[clicks, ctr] = get_regression(in_file_name)
	coef = inner_prod(clicks, ctr) / inner_prod(ctr, ctr)

	out_file = open(out_file_name, 'w')

	for i in range(len(clicks)):
		out_file.write('0\t%d\tURL\t0\t%f\n' % (clicks[i], inv_logit(ctr[i] * coef)))

	out_file.close()

	return coef

def GetMSE(fileName):
	[clicks, CTR] = get_regression(fileName)
	return GetMSEfromArray(clicks, CTR)

def GetLL(fileName):
	[clicks, CTR] = get_regression(fileName)
	return GetLLfromArray(clicks, CTR)

def get_linear_corr(file_name):
	[clicks, CTR] = get_regression(file_name)
	return get_linear_corr_from_array(clicks, CTR)

#
# Area Under PRC for multiclassification
#
def get_auPRC_multiclass(labels, probs):

	auPRC = 0.0
	classCount = len(set(labels))

	# let iClass be interpreted as 1 and jClass as 0
	for iClass in xrange(0,classCount):
		for jClass in xrange(iClass,classCount):

			(clicks,CTRs) = ([],[])
			for (idx,label) in enumerate(labels):
				if (label in [jClass,iClass]):
					clicks.append([jClass,iClass].index(label))
					CTRs.append(probs[idx][label])

			auPRC += get_auPRC(clicks, CTRs)
	
	auPRC /= (classCount - 1) * classCount / 2.0

	return auPRC

