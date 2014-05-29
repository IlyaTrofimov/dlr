#!/usr/bin/env python
import os
import sys
from marina.ts_parser import FactorLog

items = {}

os.system('cat %s | tr " " "\\t" | sed "/^$/d" > tmp' % sys.argv[1])

flog = FactorLog("tmp")
flog.Open()

print flog.get_str_header()

while not flog.IsDone():
	data = flog.Line.string()
	key = (flog.Line.id)[0:22] 
	new_metrics = flog.Line.id[23:]

	if key in items.keys():			
		metrics, _ = items[key]
		if float(metrics) < float(new_metrics):
			items[key] = (new_metrics, data)
	else:
		items[key] = (new_metrics, data)
		
	flog.Next()

for key in sorted(items.keys()):
	print items[key][1]
