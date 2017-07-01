#!/bin/bash

set -e

target_dir=pattern
mkdir -p $target_dir

for mode in full mask vertical horizontal middle; do
    echo "Drawing $target_dir/$mode.svg"
    ./make_pattern.py $mode > $target_dir/$mode.svg
    echo "Converting $target_dir/$mode.svg to $target_dir/$mode.png"
    convert $target_dir/$mode.svg $target_dir/$mode.png
done
