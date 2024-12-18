---
base_model: microsoft/mpnet-base
library_name: sentence-transformers
metrics:
- cosine_accuracy
- dot_accuracy
- manhattan_accuracy
- euclidean_accuracy
- max_accuracy
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:2785066
- loss:MultipleNegativesRankingLoss
widget:
- source_sentence: " \n\n\"Current Context\": This sentence explains the background\
    \ that contributes to the current situation, making it a relevant aspect of the\
    \ broader discussion within the document."
  sentences:
  - " \n\n\"Detail\": This sentence provides additional information about a specific\
    \ aspect of the event or topic being discussed, offering a more nuanced understanding\
    \ of the subject matter."
  - " \n\n\"Personal Reflection\": This sentence provides a personal perspective or\
    \ insight from the individual involved in the main event, adding an emotional\
    \ or reflective layer to the narrative."
  - " \n\n\"Specific Detail\": This sentence provides a specific example of the difficulties\
    \ RSOs faced with the new application process, highlighting a technical issue\
    \ that hindered their ability to complete the required modules."
- source_sentence: " \n\n\"Background Information\": This sentence provides context\
    \ about the past season's major events, setting the stage for the discussion of\
    \ emerging players and their potential to win a slam in the future."
  sentences:
  - " \n\n\"Supporting Detail\": This sentence provides additional information about\
    \ a specific esports athlete, Nicolai “dev1ce” Reedtz, to illustrate the point\
    \ made in the preceding sentence about the celebrity effect of esports athletes."
  - " \n\n\"Current Context\": This sentence explains the background that contributes\
    \ to the Ryder Cup's reputation, making it a relevant aspect of the broader discussion\
    \ within the document."
  - " \n\n\"Specific Detail\": This sentence provides a specific detail about a particular\
    \ event or situation, adding depth and context to the narrative."
- source_sentence: " \n\n\"Event Purpose\": This sentence explains the purpose and\
    \ goals of the event, which is a key aspect of the news article."
  sentences:
  - " \n\n\"Event Background\": This sentence provides context and background information\
    \ about the event being reported on, including its purpose and location."
  - " \n\n\"Current Context\": This sentence explains the background that contributes\
    \ to the team's current situation, making it a relevant aspect of the broader\
    \ discussion within the document."
  - " \n\n\"Closing Context\": This sentence provides a final piece of information\
    \ that helps to contextualize the timing of the event discussed in the article."
- source_sentence: " \n\n\"Background Information\": This sentence provides additional\
    \ context about the filmmaker's background and influences, which helps to understand\
    \ the filmmaker's perspective and connection to the subject matter."
  sentences:
  - " \n\n\"Current Context\": This sentence explains the background that contributes\
    \ to the player's current situation, making it a relevant aspect of the broader\
    \ discussion within the document."
  - " \n\n\"Broader Context\": This sentence provides a general explanation of the\
    \ significance of the event, highlighting its impact on societal norms and expectations."
  - " \n\n\"Supporting Detail\": This sentence provides additional information to\
    \ support the main point of the paragraph, which is that esports athletes attract\
    \ large audiences to watch them play."
- source_sentence: " \n\n\"Contextual Background\": This sentence provides additional\
    \ information that helps to understand the broader context of the issue, highlighting\
    \ the contrast between the university's wealth and the neighborhood's poverty."
  sentences:
  - " \n\n\"Contextualization\": This sentence provides additional information that\
    \ helps to understand the broader implications of the event described in the article."
  - " \n\n\"Contrast\": This sentence presents a contrast between the expected atmosphere\
    \ of a restaurant on the eve of its closure and the actual atmosphere, highlighting\
    \ a discrepancy between the two."
  - " \n\n\"Event Detail\": This sentence provides specific details about a particular\
    \ event or action within the larger narrative of the news article."
model-index:
- name: SentenceTransformer based on microsoft/mpnet-base
  results:
  - task:
      type: triplet
      name: Triplet
    dataset:
      name: all nli dev
      type: all-nli-dev
    metrics:
    - type: cosine_accuracy
      value: 0.948047984431271
      name: Cosine Accuracy
    - type: dot_accuracy
      value: 0.05403095792924415
      name: Dot Accuracy
    - type: manhattan_accuracy
      value: 0.9420373635133049
      name: Manhattan Accuracy
    - type: euclidean_accuracy
      value: 0.9449277756034857
      name: Euclidean Accuracy
    - type: max_accuracy
      value: 0.948047984431271
      name: Max Accuracy
---

# SentenceTransformer based on microsoft/mpnet-base

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [microsoft/mpnet-base](https://huggingface.co/microsoft/mpnet-base) on the json dataset. It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [microsoft/mpnet-base](https://huggingface.co/microsoft/mpnet-base) <!-- at revision 6996ce1e91bd2a9c7d7f61daec37463394f73f09 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 tokens
- **Similarity Function:** Cosine Similarity
- **Training Dataset:**
    - json
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: MPNetModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    ' \n\n"Contextual Background": This sentence provides additional information that helps to understand the broader context of the issue, highlighting the contrast between the university\'s wealth and the neighborhood\'s poverty.',
    ' \n\n"Contrast": This sentence presents a contrast between the expected atmosphere of a restaurant on the eve of its closure and the actual atmosphere, highlighting a discrepancy between the two.',
    ' \n\n"Event Detail": This sentence provides specific details about a particular event or action within the larger narrative of the news article.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Triplet
* Dataset: `all-nli-dev`
* Evaluated with [<code>TripletEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.TripletEvaluator)

| Metric             | Value     |
|:-------------------|:----------|
| cosine_accuracy    | 0.948     |
| dot_accuracy       | 0.054     |
| manhattan_accuracy | 0.942     |
| euclidean_accuracy | 0.9449    |
| **max_accuracy**   | **0.948** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### json

* Dataset: json
* Size: 2,785,066 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                             | positive                                                                           | negative                                                                           |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | string                                                                             |
  | details | <ul><li>min: 18 tokens</li><li>mean: 31.85 tokens</li><li>max: 60 tokens</li></ul> | <ul><li>min: 19 tokens</li><li>mean: 31.94 tokens</li><li>max: 57 tokens</li></ul> | <ul><li>min: 19 tokens</li><li>mean: 31.46 tokens</li><li>max: 47 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                | positive                                                                                                                                                                                                                 | negative                                                                                                                                                                          |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code> <br><br>"Specific Detail": This sentence provides a detailed explanation of a specific aspect of the main event, in this case, a particular proposition being discussed.</code>                | <code> <br><br>"Event Context": This sentence provides context to a specific event that is relevant to the broader discussion of gun control in the article.</code>                                                      | <code> <br><br>"Current Context": This sentence provides background information that supports the author's prediction for the Las Vegas Aces' chances in the playoffs.</code>     |
  | <code> <br><br>"Background Information": This sentence provides additional context about the team's coaching staff, which is relevant to the broader discussion of the team's upcoming season.</code> | <code> <br><br>"Explanatory Background": This sentence provides additional information that explains the context or background of the situation, helping to clarify the circumstances surrounding the main event.</code> | <code> <br><br>"Supporting Evidence": This sentence provides additional information that supports the claims made in the article, specifically the severity of the injury.</code> |
  | <code> <br><br>"Specific Example": This sentence provides a specific instance or illustration of a broader issue or concept discussed in the article.</code>                                          | <code> <br><br>"Supporting Detail": This sentence provides additional information that supports the perspective of a specific individual, adding depth to the narrative.</code>                                          | <code> <br><br>"Image Description": This sentence provides a description of an image that is included in the article, providing context for the visual content.</code>            |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Evaluation Dataset

#### json

* Dataset: json
* Size: 2,785,066 evaluation samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                             | positive                                                                           | negative                                                                           |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | string                                                                             |
  | details | <ul><li>min: 19 tokens</li><li>mean: 32.11 tokens</li><li>max: 49 tokens</li></ul> | <ul><li>min: 21 tokens</li><li>mean: 32.09 tokens</li><li>max: 55 tokens</li></ul> | <ul><li>min: 19 tokens</li><li>mean: 31.64 tokens</li><li>max: 51 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                        | positive                                                                                                                                                                                                                                     | negative                                                                                                                                                                                       |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code> <br><br>"Relevant Context": This sentence provides background information that is relevant to the current discussion, making it a contextual aspect of the broader topic within the document.</code>   | <code> <br><br>"Quotation Context": This sentence provides context for a quotation, explaining what the speaker is referring to.</code>                                                                                                      | <code> <br><br>"Detail": This sentence provides additional information about the event, elaborating on the circumstances surrounding the hijacking.</code>                                     |
  | <code> <br><br>"Supporting Detail": This sentence provides additional information about the locations of Café de Leche, which is a relevant aspect of the broader discussion about the business.</code>       | <code> <br><br>"Specific Detail": This sentence provides a specific example or detail about the organization's efforts to support mental health, elaborating on their mission and initiatives.</code>                                        | <code> <br><br>"Event Detail": This sentence provides specific information about the event, elaborating on the actions taken by DPS in response to the walkout.</code>                         |
  | <code> <br><br>"Supporting Detail": This sentence provides additional information that elaborates on the main point of the text, in this case, the efforts of TACO to distribute fentanyl test strips.</code> | <code> <br><br>"Background Information": This sentence provides additional context about Josh Jacobs' past performance, specifically his injury history, to support the author's argument about his potential for the current season.</code> | <code> <br><br>"Introduction": This sentence introduces the main topic of the article, setting the stage for the discussion of the presidential debate and its impact on Latino voters.</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
- `learning_rate`: 2e-05
- `warmup_ratio`: 0.1
- `fp16`: True
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 2e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `eval_use_gather_object`: False
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: proportional

</details>

### Training Logs
| Epoch  | Step   | Training Loss | Validation Loss | all-nli-dev_max_accuracy |
|:------:|:------:|:-------------:|:---------------:|:------------------------:|
| 0      | 0      | -             | -               | 0.7271                   |
| 0.2553 | 10000  | 3.8072        | -               | -                        |
| 0.5107 | 20000  | 3.6873        | 3.6056          | 0.9370                   |
| 0.7660 | 30000  | 3.6449        | -               | -                        |
| 1.0050 | 40000  | 3.3801        | 3.5329          | 0.9395                   |
| 1.2603 | 50000  | 3.5747        | -               | -                        |
| 1.5156 | 60000  | 3.5275        | 3.4488          | 0.9410                   |
| 1.7709 | 70000  | 3.4822        | -               | -                        |
| 2.0099 | 80000  | 3.2181        | 3.3667          | 0.9439                   |
| 2.2652 | 90000  | 3.3995        | -               | -                        |
| 2.5206 | 100000 | 3.3543        | 3.2828          | 0.9480                   |
| 2.7759 | 110000 | 3.3181        | -               | -                        |


### Framework Versions
- Python: 3.11.8
- Sentence Transformers: 3.2.1
- Transformers: 4.44.2
- PyTorch: 2.4.1
- Accelerate: 1.0.1
- Datasets: 3.0.2
- Tokenizers: 0.19.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->