#!/bin/bash

realization="B"
scenario="historical"
conda activate eng_boost
for date in "1996-01-20" "1996-01-23" "1996-01-26" "1996-01-29" "1996-02-01"  
    do
    python preprocess_eng_data.py ${date} ${realization} SSP370
    python preprocess_CESM_bced_boosted.py ${date} ${realization} ${scenario}
done
conda deactivate
