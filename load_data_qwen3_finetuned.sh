#!/bin/bash


python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name syndrome_db_tcm_embedding_name_disease \
  --embedding-model custom:Qwen3-Embedding-0.6B-finetuned \

python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name syndrome_db_tcm_embedding_name_disease \
  --embedding-model custom:Qwen3-Embedding-0.6B-finetuned \
  --fields Name,Common_disease

python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name syndrome_db_tcm_embedding_disease_performance \
  --embedding-model custom:Qwen3-Embedding-0.6B-finetuned \
  --fields Common_disease,Typical_performance 
