import numpy as np

def generate_file(alpha, positive, negative):
	f = open('./example3.py', 'r')
	out = open('./implementation' + str(alpha) + "-" + str(positive) + "-" + str(negative)+ '.py','w')
	lines = f.readlines()

	for line in lines:
		if line.startswith("alpha"):
			out.write("alpha=" + str(alpha) + "\n")
		elif line.startswith("positive"):
			out.write("positive=" + str(positive) + "\n")
		elif line.startswith("negative"):
			out.write("negative=" + str(positive) + "\n")
		else:
			out.write(line)

def generate_script(start, stop, step, start2, stop2, step2):
	out = open('./run.sh','w')
	out.write("module load python/2.7\n")
	for i in np.arange(start, stop, step):
		for j in np.logspace(start2, stop2, step2):
			for k in np.logspace(start2, stop2, step2):
				out.write("bsub python runner.py data/webscope-logs.txt data/webscope-articles.txt " + "implementation" + str(i) + "-" + str(j) + "-" + str(k) + ".py\n")



start = 0.001
stop = 0.1
step = 0.001

start2 = -5
stop2 = 5
step2 = 11

for i in np.arange(start, stop, step):
	for j in np.logspace(start2, stop2, step2):
		for k in np.logspace(start2, stop2, step2):
			generate_file(i,j,k)
generate_script(start, stop, step, start2, stop2, step2)
