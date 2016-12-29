import os

score_list = []
alpha_list = []
positive_list = []
negative_list = []

for filename in os.listdir("output"):
	f = open("output/" + filename, 'r')
	lines = f.readlines()

	for line in lines:
		if line.startswith("INFO"):
			print line[42:-1]
			score_list.append(float(line[42:-1]))
		if line.startswith("alpha"):
			print line
			alpha_list.append(float(line[7:-1]))
		if line.startswith("postive"):
			print line
			positive_list.append(float(line[9:-1]))
		if line.startswith("negative"):
			print line
			negative_list.append(float(line[10:-1]))


score_list, alpha_list, positive_list, negative_list = zip(*sorted(zip(score_list, alpha_list, positive_list, negative_list)))

for i in range(1,11):
	print "TOP", i
	print "====="
	print "score", score_list[-i]
	print "alpha", alpha_list[-i]
	print "positive", positive_list[-i]
	print "negative", negative_list[-i]
	print ""