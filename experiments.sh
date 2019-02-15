#!/bin/bash

for seed in {1..5}
do
	echo ${seed}

	python training.py ${seed}
	

done
