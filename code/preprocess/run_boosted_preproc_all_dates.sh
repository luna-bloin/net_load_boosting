#!/bin/bash

realization="A"
scenario="SSP370"
conda activate eng_boost
for date in  "2088-12-02" "2088-12-05" "2088-12-08" "2088-12-11" "2081-12-26" "2081-12-29" "2082-01-01" "2082-01-04" #"1996-01-20" "1996-01-23" "1996-01-26" "1996-01-29" "1996-02-01"
    do
    python preprocess_eng_data.py ${date} ${realization} ${scenario}
    python preprocess_CESM_bced_boosted.py ${date} ${realization} ${scenario}
done
conda deactivate
