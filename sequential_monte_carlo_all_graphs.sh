#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
for i in {0..56..1}
  do 
     echo "running graph $i "
     python sequential_monte_carlo_all_graphs.py "$i" 1000
 done