#!/usr/bin/python
import os
import sys
import urllib
import StringIO

''' Decodes record values downloaded from MR'''

if __name__ == '__main__':

	values = {}

	for line in sys.stdin:
		line = line.rstrip('\n')
		items = line.split('\t')
		key = items[0]
		value = '\t'.join(items[1:])
		
		if key in values:
			values[key] = values[key] + [value]
		else:
			values[key] = [value]

	for key in values.keys():
		values[key].sort()

	for key in values.keys():
		united = ''

		for item in values[key]:
			united += item[3:]

		united = united.decode("string-escape")

		filename = '%s/%s.%s' % (sys.argv[1], key, sys.argv[2])
		
		file = open(filename, 'w')
		file.write(united)
		file.close()
