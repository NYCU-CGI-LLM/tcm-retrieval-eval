#!/bin/bash

# vLLM Qwen3-Embedding-8B RAG Data Loading Script
# Creates multiple databases with different field combinations using vLLM Qwen3-Embedding-8B embedding model

echo "Starting data ingestion for multiple SelectiveRAG databases using vLLM Qwen3-Embedding-8B..."
echo "請確保 vLLM 服務正在運行: ./start_qwen3_embedding.sh"
echo ""

# 1. Complete database with all fields (SimpleRAG mode) - vLLM version
echo "Creating qwen3_syndrome_db (all fields)..."
python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name qwen3_syndrome_db \
  --embedding-model vllm:Qwen3-Embedding-8B

# 2. Selective database: Name + Definition - vLLM version
echo "Creating qwen3_selective_name_definition (Name + Definition)..."
python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name qwen3_selective_name_definition \
  --fields Name,Definition \
  --embedding-model vllm:Qwen3-Embedding-8B

# 3. Selective database: Name + Typical_performance - vLLM version
echo "Creating qwen3_selective_name_performance (Name + Typical_performance)..."
python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name qwen3_selective_name_performance \
  --fields Name,Typical_performance \
  --embedding-model vllm:Qwen3-Embedding-8B

# 4. Selective database: Name + Common_disease - vLLM version
echo "Creating qwen3_selective_name_disease (Name + Common_disease)..."
python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name qwen3_selective_name_disease \
  --fields Name,Common_disease \
  --embedding-model vllm:Qwen3-Embedding-8B

# 5. Only Definition database - vLLM version
echo "Creating qwen3_only_definition (Definition only)..."
python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name qwen3_only_definition \
  --fields Definition \
  --embedding-model vllm:Qwen3-Embedding-8B

# 6. Only Common_disease database - vLLM version
echo "Creating qwen3_only_disease (Common_disease only)..."
python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name qwen3_only_disease \
  --fields Common_disease \
  --embedding-model vllm:Qwen3-Embedding-8B

# 7. Only Typical_performance database - vLLM version
echo "Creating qwen3_only_performance (Typical_performance only)..."
python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name qwen3_only_performance \
  --fields Typical_performance \
  --embedding-model vllm:Qwen3-Embedding-8B

# 8. Definition + Common_disease database - vLLM version
echo "Creating qwen3_definition_disease (Definition + Common_disease)..."
python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name qwen3_definition_disease \
  --fields Definition,Common_disease \
  --embedding-model vllm:Qwen3-Embedding-8B

# 9. Common_disease + Typical_performance database - vLLM version
echo "Creating qwen3_disease_performance (Common_disease + Typical_performance)..."
python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name qwen3_disease_performance \
  --fields Common_disease,Typical_performance \
  --embedding-model vllm:Qwen3-Embedding-8B

# 10. Definition + Typical_performance database - vLLM version
echo "Creating qwen3_definition_performance (Definition + Typical_performance)..."
python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name qwen3_definition_performance \
  --fields Definition,Typical_performance \
  --embedding-model vllm:Qwen3-Embedding-8B

# 11. Name + Definition + Common_disease - vLLM version
echo "Creating qwen3_name_definition_disease (Name + Definition + Common_disease)..."
python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name qwen3_name_definition_disease \
  --fields Name,Definition,Common_disease \
  --embedding-model vllm:Qwen3-Embedding-8B

# 12. Name + Definition + Typical_performance - vLLM version
echo "Creating qwen3_name_definition_performance (Name + Definition + Typical_performance)..."
python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name qwen3_name_definition_performance \
  --fields Name,Definition,Typical_performance \
  --embedding-model vllm:Qwen3-Embedding-8B

# 13. Name + Common_disease + Typical_performance - vLLM version
echo "Creating qwen3_name_disease_performance (Name + Typical_performance + Common_disease)..."
python ingest.py \
  --json-file data/syndrome_knowledge_fixed.json \
  --db-name qwen3_name_disease_performance \
  --fields Name,Typical_performance,Common_disease \
  --embedding-model vllm:Qwen3-Embedding-8B

echo ""
echo "🎉 All vLLM databases created successfully!"
echo "Available vLLM databases (using Qwen3-Embedding-8B):"
echo "=================================================="
echo "- qwen3_syndrome_db: Complete database (all fields)"
echo "- qwen3_selective_name_definition: Name + Definition"
echo "- qwen3_selective_name_performance: Name + Typical_performance"  
echo "- qwen3_selective_name_disease: Name + Common_disease"
echo "- qwen3_only_definition: Definition only"
echo "- qwen3_only_disease: Common_disease only"
echo "- qwen3_only_performance: Typical_performance only"
echo "- qwen3_definition_disease: Definition + Common_disease"
echo "- qwen3_disease_performance: Common_disease + Typical_performance"
echo "- qwen3_definition_performance: Definition + Typical_performance"
echo "- qwen3_name_definition_disease: Name + Definition + Common_disease"
echo "- qwen3_name_definition_performance: Name + Definition + Typical_performance"
echo "- qwen3_name_disease_performance: Name + Common_disease+ Typical_performance"
echo ""
echo "📊 模型比較:"
echo "- 原始資料庫 (OpenAI): syndrome_db, selective_name_definition, ..."
echo "- vLLM 資料庫 (Qwen3): qwen3_syndrome_db, qwen3_selective_name_definition, ..."
echo ""
echo "💡 使用建議:"
echo "現在你可以比較不同嵌入模型的檢索效果:"
echo "- OpenAI text-embedding-3-large vs Qwen3-Embedding-8B"
echo "- 相同的欄位組合，不同的向量表示"
