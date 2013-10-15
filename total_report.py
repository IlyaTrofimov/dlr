#!/usr/bin/env python
import os
import sys

#dumps = ['./dump-20130813-0', './dump-20130816-1', './dump-20130819-1', './dump-20130822-0', './dump-20130822-2']
#dumps = ['./dump-20130830-0', './dump-20130830-1', './dump-20130830-2', './dump-20130830-3']
dumps = ['./dump-20130910-1', './dump-20130911-0', './dump-20130911-1', './dump-20130910-0', 'dump-20130911-2']

for dump in dumps:
	a_file = os.path.join(dump, 'time_summary')
	sys.stdout.write(dump + '\n')
	os.system('cat ' + os.path.join(dump, 'task'))
	os.system('cat %s | tabpp' % a_file)
	sys.stdout.write('\n')
