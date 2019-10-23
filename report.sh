#!/bin/bash


report_dir=./dev_report_test_set

rm -rf $report_dir

python3 -m dna report \
    --results-dir ./dev_results_test_set/aggregate \
    --report-dir $report_dir
