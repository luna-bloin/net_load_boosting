#!/bin/bash

realization=A
conda activate eng_boost
for date in "2080-02-14" "2080-02-16" "2080-02-18" "2080-12-01" "2080-12-03" "2080-12-05" 
    do
    python preprocess_eng_data.py ${date} ${realization} SSP370
    python preprocess_CESM_bced_boosted.py ${date} ${realization} SSP370
done
conda deactivate
