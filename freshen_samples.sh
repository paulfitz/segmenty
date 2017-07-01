#!/bin/bash

set -e
while true; do
    ./make_samples.py training 3000
    sleep 120
done
