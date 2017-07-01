#!/bin/bash

# Cobble together an animation showing evolution of output
# for different inputs.
# If you put some *.jpg files in a directory called "wild"
# prior to training, then at the end of every epoch inference
# will be run on each of those images and logged in a
# "snapshots" directory.  This script will make a little
# animated gif for one of these images.

set -e

index=0
if [ ! -z "$1" ]; then
    index=$1
fi

index=`printf "%06d" $index`

prefixes=`ls snapshots/${index}_*_in.jpg | sort | sed "s/_in.*//"`

for p in $prefixes; do
    target="${p}_montage.png"
    if [ ! -e $target ]; then
	echo "Making $target"
	montage -adjoin -tile 5x1 ${p}_in.jpg ${p}_out* $target
    fi
done

frames=`ls snapshots/${index}_*_montage.png | sort`
target=animation_${index}.gif
echo "Generating $target"
convert -delay 100 $frames -loop 10000 $target
identify $target
echo "Saved animation as $target"
