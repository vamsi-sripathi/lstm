#!/bin/bash

set -v

icx -qmkl -xCORE-AVX512 -O3 -c lstm.c
icc -qmkl -xCORE-AVX512 -O3 -o a.out lstm.o
