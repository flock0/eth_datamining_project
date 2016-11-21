#!/bin/bash
for i in `seq 20 2 60`
do
	echo ''
	echo "Running with $i"
	echo '================='
	python 'DataAnalysis.py' $i | tee output_$i.txt
done

# Running with 38
# =================
# TP: 8087
# TN: 8119
# FP: 1902
# FN: 1892
# accuracy: 0.8103