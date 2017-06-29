#!/bin/bash

mkdir -p pattern

for mode in full mask vertical horizontal middle; do
    flags=""
    if [ ! "$mode" = "full" ]; then
	flags="-background white -alpha deactivate -negate"
    fi
    ./make_pattern.py $mode > test.svg
    convert test.svg $flags pattern/$mode.png
done
