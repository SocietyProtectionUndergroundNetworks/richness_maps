from os import listdir
from os.path import isfile, join

fout=open("data/20211206_all_taxa_tedersoo_EricoidMycorrhizal.csv","a")

# sampled_data = [f for f in listdir("data/sampled_data") if isfile(join("data/sampled_data", f))]

sampled_data = listdir("data/sampled_data")
sampled_data = list(filter(lambda f: f.endswith('.csv'), sampled_data))

# first file:
for line in open("data/sampled_data/"+sampled_data[0]):
    fout.write(line)
# now the rest:
for num in range(1,len(sampled_data)):
	try:
		f = open("data/sampled_data/"+sampled_data[num])
		f.__next__()
		for line in f:
			fout.write(line)
		f.close()
	except IOError:
	    pass

fout.close()
