#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
for i in {0..56..1}
  do 
     echo "running graph $i "
     python rejection_sampling_all_graphs_with_IC.py "$i" 1500
 done