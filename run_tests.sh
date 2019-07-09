#!/bin/bash

reset
pip3 install .
mv dna tmp_dna
python3 run_tests.py
mv tmp_dna dna
pip3 uninstall -y dna > /dev/null
