from __future__ import division
import pylab as pl

FIN = "Si_GreenAndKeevers.csv"
FOUT = "SI_RIinfo.txt"

fin = open(FIN,'r')
fout = open(FOUT,'w')

#Sort header
fin.next()
fout.write("Wavelength\tn\tk\tSi from refractiveindex.info (Green and Keevers 1995)\n")

#Convert data from um to nm and change "," to "\t"
for line in fin:
	line = line.split(",")
	lam = str(float(line[0])*1e3)
	n = str(float(line[1]))
	try:
		k = str(float(line[2]))
	except IndexError:
		k = str(0.0)
	fout.write(lam+"\t"+n+"\t"+k+"\n")

fin.close()
fout.close()
