#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=50
#SBATCH --mem=150G
#SBATCH --partition=isi


source /home1/spangher/.bashrc
conda activate vllm-retriv-py39

start_idx=$1
step=$2
iterations=$3
iterations=$((iterations + 1))
end_idx=$((start_idx + step))

for ((i=0; i<iterations; i++)); do
    python extract_sources.py \
      --start_idx ${start_idx} \
      --end_idx ${end_idx} \
      --id_col url \
      --text_col article_text \
      --input_data_file ../../data/batch_article_text.csv \
      --output_file  ../../data/v3_source_summaries/extracted_sources.jsonl \
      --do_article_gen \
      --do_source_summ \
      --do_narr_key_prompt \
      --do_cent_prompt 

    start_idx=${end_idx}
    end_idx=$((start_idx + step))
done
