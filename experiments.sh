#!/bin/bash

for seed in {3..5}
do
	echo ${seed}

	python training.py ${seed}
	



done
