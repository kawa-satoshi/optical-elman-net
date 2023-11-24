#!/bin/bash

application=(
#    "airline-passengers"
#    "SP500"
    "iris"
    "speech_commands"
)
activation=(
    "relu"
    "tanh"
    "custom"
)


for app in "${application[@]}" ; do
	for af in "${activation[@]}" ; do
		for i in `seq 2 20`; do
#		for i in `seq 10 10`; do
			for j in `seq 2 30`; do
#			for j in `seq 20 20`; do
				echo "[ ${app} ${af} ${i} ${j} ]"
				python run_classification_var.py ${app} ${af} ${i} ${j} > ./results/${app}_${af}_${i}_${j}.txt
			done
		done
	done
done


