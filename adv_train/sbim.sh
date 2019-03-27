#!/bin/sh

for eps in 1.0 2.0 3.0 3.5 4.0 4.5 5.0 6.0
do
    echo "eps=" $eps
    python sbim.py $eps 3.0
done
