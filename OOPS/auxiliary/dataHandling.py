from __future__ import division
import warnings
import time
import os

import pylab as pl
import h5py

#Expecting data to be a dictionary
# note that lists will be converted to np arrays
#excludeFields not yet implemented
#Don't try and save to the same file within the same second... that will fail due to group naming
#savetypes:
#		a		append/create
#		w		write/overwrite
#		c		Check for existence and write if doesn't exist (will save as filename+.BAK if exist in order not to lose data)
def saveDicts(filename,data,savetype='w',excludeFields=[]):
	if savetype == 'c':
		if os.path.exists(filename):
			warnings.warn("Output file exists! Output is tried saved as {:}.BAK".format(filename))
			if ".BAK"*5 in filename:
				warnings.warn("Givining up attempts to save {:} ... Will continue without".format(filename))
			else:
				saveDicts(filename+".BAK",data,savetype,excludeFields)
			return
		savetype = 'w'
	elif savetype in ['a','w']:
		pass
	else:
		raise ValueError("Savetype is invalid. Only 'a','c' or 'w' can be used.")
	
	h = h5py.File(filename,savetype)
	if type(data) == dict:
		data = [data]
	elif type(data) == list:
		pass
	else:
		warnings.warn("Received wrong input format (dict or list of dicts), so nothing will be saved for {:}".format(filename))

	timestr = time.strftime("%Y-%m-%d_%H:%M:%S")
	for i in range(len(data)):
		group = h.create_group(timestr+"_"+str(i).zfill(5))
		for key,dat in data[i].items():
			#http://stackoverflow.com/questions/37873311/h5py-store-list-of-list-of-strings
			if type(dat) == list:
				if type(dat[0]) == str:
					dat = pl.array(dat,dtype=object)
					group.create_dataset(key,data=dat,dtype=h5py.special_dtype(vlen=str))
			else:
				group.create_dataset(key,data=dat)


def loadDicts(filename):
	h = h5py.File(filename,'r')
	dicts = []
	for groupkey in sorted(h.keys()):
		group = h[groupkey]
		data = {}
		for key in group:
			data[key] = group[key].value
		dicts += [data]
	return dicts


#Make a list of dictionaries sliceable by installing a slice method
def dictSlice(theList,key):
	return [dic[key] for dic in theList]


if __name__ == "__main__":
	fname = "test.h5"
	data = {"ab":234,"bcd":[1,2,3],"aa":"stringi"}

	#for key in data:
	#	print key, "\t", data[key]

	saveDicts(fname,data,'a')
	data = loadDicts(fname)

	for datadict in data:
		for key in datadict:
			print((key, "\t", datadict[key]))
