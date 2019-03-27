#!/bin/sh

for eps in 0.1 0.2 0.4 0.5
do
    echo "eps=" $eps
    python bim.py $eps 0.08
done
