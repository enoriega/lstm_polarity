#!/bin/bash

for seed in {4..5}
do
	echo ${seed}

	python training.py ${seed}
	

done
