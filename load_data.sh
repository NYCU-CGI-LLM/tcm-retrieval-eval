#!/bin/bash

# Simple RAG Data Loading Script
# Creates multiple databases with different field combinations for SelectiveRAG

echo "Starting data ingestion for multiple SelectiveRAG databases..."

# 1. Complete database with all fields (SimpleRAG mode)
echo "Creating syndrome_db (all fields)..."
python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name syndrome_db

# 2. Selective database: Name + Definition
echo "Creating selective_name_definition (Name + Definition)..."
python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name selective_name_definition \
  --fields Name,Definition

# 3. Selective database: Name + Typical_performance
echo "Creating selective_name_performance (Name + Typical_performance)..."
python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name selective_name_performance \
  --fields Name,Typical_performance

# 4. Selective database: Name + Common_disease
echo "Creating selective_name_disease (Name + Common_disease)..."
python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name selective_name_disease \
  --fields Name,Common_disease

# 5. Only Definition database
echo "Creating only_definition (Definition only)..."
python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name only_definition \
  --fields Definition

# 6. Only Common_disease database
echo "Creating only_disease (Common_disease only)..."
python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name only_disease \
  --fields Common_disease

# 7. Only Typical_performance database
echo "Creating only_performance (Typical_performance only)..."
python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name only_performance \
  --fields Typical_performance

# 8. Definition + Common_disease database
echo "Creating definition_disease (Definition + Common_disease)..."
python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name definition_disease \
  --fields Definition,Common_disease

# 9. Common_disease + Typical_performance database
echo "Creating disease_performance (Common_disease + Typical_performance)..."
python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name disease_performance \
  --fields Common_disease,Typical_performance

# 10. Definition + Typical_performance database
echo "Creating definition_performance (Definition + Typical_performance)..."
python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name definition_performance \
  --fields Definition,Typical_performance

# 11. Name + Definition + Common_disease
echo "Creating name_definition_disease (Name + Definition + Common_disease)..."
python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name name_definition_disease \
  --fields Name,Definition,Common_disease

# 12. Name + Definition + Typical_performance
echo "Creating name_definition_performance (Name + Definition + Typical_performance)..."
python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name name_definition_performance \
  --fields Name,Definition,Typical_performance

# 13. Name + Common_disease + Typical_performance
echo "Creating name_disease_performance (Name + Typical_performance + Common_disease)..."
python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name name_disease_performance \
  --fields Name,Typical_performance,Common_disease

echo "All databases created successfully!"
echo "Available databases:"
echo "- syndrome_db: Complete database (all fields)"
echo "- selective_name_definition: Name + Definition"
echo "- selective_name_performance: Name + Typical_performance"  
echo "- selective_name_disease: Name + Common_disease"
echo "- only_definition: Definition only"
echo "- only_disease: Common_disease only"
echo "- only_performance: Typical_performance only"
echo "- definition_disease: Definition + Common_disease"
echo "- disease_performance: Common_disease + Typical_performance"
echo "- definition_performance: Definition + Typical_performance"
echo "- name_definition_disease: Name + Definition + Common_disease"
echo "- name_definition_performance: Name + Definition + Typical_performance"
echo "- name_performance_disease: Name + Common_disease+ Typical_performance"
