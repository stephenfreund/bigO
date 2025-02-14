#!/bin/bash

rm bigO_data.json 

test_files=(
    "facts.py"
    "exercise_timespace.py"
    "test_ab_sort.py"
    "test_bounds_linear_function.py"
    "test_check_insertion_sort.py"
    "test_infer_selection_sort.py"
    "test_limit_insertion_sort.py"
)

for test_file in "${test_files[@]}"; do
    python3 "$test_file"
done

bigo-report --open-plots
