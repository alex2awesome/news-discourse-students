#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=50
#SBATCH --mem=150G
#SBATCH --partition=sched_mit_psfc_gpu_r8

# Activate your environment or load modules if needed
source /home/spangher/.bashrc
conda activate alex

python merge_labels.py \
    --input_data_file ../../data/v3_discourse_summaries/news-discourse/all_extracted_discourse.csv.gz \
    --input_col_name discourse_label \
    --trained_sbert_model_name models/mpnet-base-all-nli-triplet/trained-model \
    --output_cluster_file ../../data/v3_discourse_summaries/news-discourse/cluster_centroids.npy \
    --output_data_file ../../data/v3_discourse_summaries/news-discourse/all_extracted_discourse_with_clusters.csv
