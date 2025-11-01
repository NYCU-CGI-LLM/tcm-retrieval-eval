#!/usr/bin/env python3
"""
從tcm_sd_test_rc_direct.json中加載測試數據進行檢索評估
使用患者臨床信息作為查詢，評估檢索系統的性能
使用OpenAI text-embedding-3-large計算語意相似度並計算RR (Reciprocal Rank)
"""

import os
import sys
import json
import random
import logging
import argparse
import asyncio
import requests
import time
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
import matplotlib.pyplot as plt
import numpy as np
from tqdm.asyncio import tqdm
from tqdm import tqdm as sync_tqdm

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# 加載環境變量
load_dotenv()

# Supported embedding providers and models (same as ingest.py)
EMBEDDING_PROVIDERS = {
    "openai": {
        "models": [
            "text-embedding-3-large"
        ],
        "default": "text-embedding-3-large"
    },
    "vllm": {
        "models": [
            "Qwen3-Embedding-8B"
        ],
        "default": "Qwen3-Embedding-8B",
        "base_url": "http://localhost:8010/v1"
    },
    "custom": {
        "models": [
            "Qwen3-Embedding-0.6B-base",
            "Qwen3-Embedding-0.6B-finetuned",
            "Qwen3-Embedding-4B-base",
            "Qwen3-Embedding-4B-finetuned"
        ],
        "default": "Qwen3-Embedding-0.6B-finetuned",
        "base_url": "http://localhost:8000/v1"
    },
    "huggingface": {
        "models": [
            "BAAI/bge-large-zh-v1.5"
        ],
        "default": "BAAI/bge-large-zh-v1.5"
    }
}

class CustomEmbeddings(Embeddings):
    """Custom embeddings class that works with your fine-tuned model API."""
    
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        self.endpoint = f"{base_url}/embeddings"
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        embeddings = []
        for text in texts:
            response = requests.post(
                self.endpoint,
                json={"model": self.model, "input": text}
            )
            response.raise_for_status()
            embedding = response.json()["data"][0]["embedding"]
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        response = requests.post(
            self.endpoint,
            json={"model": self.model, "input": text}
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

def get_provider_and_model(embedding_model: str) -> tuple[str, str]:
    """Determine the provider and model name from the embedding_model string."""
    
    # Check if it's a provider:model format (e.g., "custom:Qwen3-Embedding-0.6B-finetuned")
    if ":" in embedding_model:
        provider, model = embedding_model.split(":", 1)
        if provider in EMBEDDING_PROVIDERS:
            if model in EMBEDDING_PROVIDERS[provider]["models"]:
                return provider, model
            else:
                logging.error(f"Model '{model}' not supported for provider '{provider}'")
                logging.info(f"Supported models for {provider}: {EMBEDDING_PROVIDERS[provider]['models']}")
                raise ValueError(f"Unsupported model '{model}' for provider '{provider}'")
        else:
            logging.error(f"Unsupported provider: {provider}")
            logging.info(f"Supported providers: {list(EMBEDDING_PROVIDERS.keys())}")
            raise ValueError(f"Unsupported provider: {provider}")
    
    # Check if it's a direct model name and find the matching provider
    for provider, config in EMBEDDING_PROVIDERS.items():
        if embedding_model in config["models"]:
            return provider, embedding_model
    
    # If not found, assume it's an OpenAI model (for backward compatibility)
    logging.warning(f"Model '{embedding_model}' not found in predefined models. Assuming OpenAI provider.")
    return "openai", embedding_model

def initialize_embeddings(embedding_model: str):
    """Initialize the appropriate embeddings class based on the model specification."""
    
    provider, model = get_provider_and_model(embedding_model)
    
    logging.info(f"Initializing {provider} embeddings with model: {model}")
    
    if provider == "openai":
        # Check if OPENAI_API_KEY is available
        if not os.getenv("OPENAI_API_KEY"):
            logging.error("OPENAI_API_KEY environment variable not set for OpenAI embeddings.")
            raise ValueError("OPENAI_API_KEY required for OpenAI embeddings")
        
        return OpenAIEmbeddings(model=model)
    
    elif provider == "vllm":
        # Use OpenAI-compatible client pointing to vLLM server
        base_url = EMBEDDING_PROVIDERS[provider]["base_url"]
        logging.info(f"Using vLLM server at: {base_url}")
        
        # Try to create OpenAI embeddings with different parameter names for compatibility
        try:
            # First try with newer langchain-openai version parameters
            return OpenAIEmbeddings(
                model=model,
                api_key="unused",  # vLLM doesn't require a real API key
                base_url=base_url
            )
        except TypeError:
            try:
                # Try with older parameter names
                return OpenAIEmbeddings(
                    model=model,
                    openai_api_key="unused",
                    openai_api_base=base_url
                )
            except Exception as e:
                logging.error(f"Failed to initialize vLLM embeddings: {e}")
                logging.info(f"Please ensure vLLM server is running at {base_url}")
                raise
    
    elif provider == "custom":
        # Use custom embeddings class for fine-tuned model server
        base_url = EMBEDDING_PROVIDERS[provider]["base_url"]
        logging.info(f"Using custom fine-tuned model server at: {base_url}")
        
        try:
            return CustomEmbeddings(base_url=base_url, model=model)
        except Exception as e:
            logging.error(f"Failed to initialize custom embeddings: {e}")
            logging.info(f"Please ensure custom server is running at {base_url}")
            raise
    
    elif provider == "huggingface":
        # Use local Hugging Face model
        logging.info(f"Loading Hugging Face model: {model}")
        
        return HuggingFaceEmbeddings(
            model_name=model,
            encode_kwargs={'normalize_embeddings': True}  # Normalize embeddings for better similarity search
        )
    
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


class RetrievalEvaluator:
    """檢索評估器"""
    
    def __init__(self, db_name: str = "syndrome_db", embedding_model: str = "text-embedding-3-large", max_concurrent: int = 5, 
                 use_rerank: bool = False, reranker_api_url: str = "http://localhost:8001", 
                 reranker_model: str = "Qwen/Qwen3-Reranker-0.6B", rerank_top_n: int = 100):
        self.db_name = db_name
        self.persist_directory = os.path.join("chroma_dbs", db_name)
        self.embedding_model = embedding_model
        self.max_concurrent = max_concurrent
        self.use_rerank = use_rerank
        self.reranker_api_url = reranker_api_url
        self.reranker_model = reranker_model
        self.rerank_top_n = rerank_top_n
        
        # 檢查數據庫是否存在
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(f"❌ 數據庫目錄不存在: {self.persist_directory}")
        
        # 初始化embeddings和vectorstore
        self.embeddings = initialize_embeddings(self.embedding_model)
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        
        # 創建信號量來控制併發
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        logging.info(f"✅ 已初始化數據庫: {db_name}")
        logging.info(f"🔍 使用embedding模型: {embedding_model}")
        logging.info(f"⚡ 最大併發數: {max_concurrent}")
        if self.use_rerank:
            logging.info(f"🔄 啟用兩階段檢索 - Reranker API: {reranker_api_url}")
            logging.info(f"🎯 Reranker模型: {reranker_model}")
            logging.info(f"📊 Rerank Top-N: {rerank_top_n}")
            # 檢查 Reranker API 健康狀態
            self._check_reranker_health()
    
    def _check_reranker_health(self):
        """檢查 Reranker API 健康狀態"""
        try:
            response = requests.get(f"{self.reranker_api_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                logging.info(f"✅ Reranker API 健康檢查通過")
                logging.info(f"📋 API狀態: {health_data.get('status', 'unknown')}")
                logging.info(f"🖥️  運行設備: {health_data.get('device', 'unknown')}")
                
                # 檢查模型是否匹配
                api_model = health_data.get('model', '')
                if api_model and api_model != self.reranker_model:
                    logging.warning(f"⚠️ 模型不匹配 - 配置: {self.reranker_model}, API: {api_model}")
            else:
                logging.error(f"❌ Reranker API 健康檢查失敗: {response.status_code}")
        except requests.exceptions.ConnectionError:
            logging.error(f"❌ 無法連接到 Reranker API: {self.reranker_api_url}")
            logging.error("請確保 Reranker API 服務正在運行")
        except Exception as e:
            logging.warning(f"⚠️ Reranker API 健康檢查異常: {str(e)}")
    
    def load_test_data(self, json_file_path: str) -> List[Dict[str, Any]]:
        """加載測試數據"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info(f"📊 已加載 {len(data)} 條測試數據")
            return data
        except Exception as e:
            logging.error(f"❌ 加載數據失敗: {e}")
            raise
    
    def select_random_queries(self, data: List[Dict[str, Any]], num_queries: int = 10) -> List[Dict[str, Any]]:
        """隨機選擇測試項目"""
        if len(data) < num_queries:
            logging.warning(f"⚠️  數據量 ({len(data)}) 少於請求數量 ({num_queries})")
            num_queries = len(data)
        
        selected = random.sample(data, num_queries)
        logging.info(f"🎲 已隨機選擇 {num_queries} 個測試項目")
        return selected
    
    def generate_query_text(self, item: Dict[str, Any], query_type: str = "prompt", use_pseudo_doc: bool = False, use_keywords: bool = False) -> str:
        """從測試項目中提取查詢文本"""
        if query_type == "prompt":
            # 如果啟用keywords檢索且存在keywords字段
            if use_keywords and "keywords" in item:
                keywords = item.get("keywords", "").strip()
                if keywords:
                    return keywords
            
            query_text = item.get("prompt", "").strip()
            
            # 如果啟用Query2Doc方法且存在pseudo_document
            if use_pseudo_doc and "pseudo_document" in item:
                pseudo_doc = item.get("pseudo_document", "").strip()
                if pseudo_doc:
                    # 將原始查詢與pseudo_document結合
                    query_text = f"{query_text}\n\n{pseudo_doc}"
            
            return query_text
        else:
            # 保持向後兼容，雖然對新數據格式可能無效
            return item.get(query_type, "").strip()
    
    def search_with_vectorstore(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """使用vectorstore進行相似度搜索（同步版本）"""
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            formatted_results = []
            
            for doc, score in results:
                # ChromaDB返回的是距離分數，越小越相似
                distance = score
                similarity = 1.0 / (1.0 + distance)  # 轉換為相似度分數，越大越相似 1.0 / (1.0 + L2_distance)
                
                result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "distance_score": distance,      # 距離分數：越小越相似
                    "similarity_score": similarity   # 相似度分數：越大越相似
                }
                formatted_results.append(result)
            
            return formatted_results
        except Exception as e:
            logging.error(f"❌ 搜索失敗: {e}")
            return []
    
    async def search_with_vectorstore_async(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """使用vectorstore進行相似度搜索（異步版本）"""
        async with self.semaphore:  # 控制併發數量
            # 在線程池中執行同步操作
            loop = asyncio.get_event_loop()
            try:
                formatted_results = await loop.run_in_executor(
                    None, self.search_with_vectorstore, query, k
                )
                return formatted_results
            except Exception as e:
                logging.error(f"❌ 異步搜索失敗: {e}")
                return []
    
    def _rerank_with_api(self, query: str, documents: List[str], top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """使用 Reranker API 進行批量重排 - 優化版本，支持批量推理和並發處理"""
        try:
            # 構建請求 - 利用新 API 的 top_n 參數
            request_data = {
                "model": self.reranker_model,
                "query": query,
                "documents": documents
            }
            
            if top_n and top_n > 0:
                request_data["top_n"] = top_n
            
            # 根據文檔數量調整超時時間
            timeout = min(60 + len(documents) * 0.1, 120)
            
            logging.debug(f"🔄 發送 {len(documents)} 個文檔到 Reranker API (top_n={top_n})")
            
            response = requests.post(
                f"{self.reranker_api_url}/v1/rerank",
                json=request_data,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                api_results = result["results"]
                usage = result.get("usage", {})
                
                logging.debug(f"✅ Reranker API 成功處理 {usage.get('total_documents', len(documents))} 個文檔，"
                            f"返回 {usage.get('returned_documents', len(api_results))} 個結果")
                
                return api_results
            else:
                logging.error(f"❌ Reranker API 調用失敗: {response.status_code}")
                if response.text:
                    logging.error(f"響應內容: {response.text[:200]}")
                return []
                
        except requests.exceptions.Timeout:
            logging.error(f"❌ Reranker API 調用超時 (文檔數: {len(documents)})")
            return []
        except requests.exceptions.ConnectionError:
            logging.error(f"❌ 無法連接到 Reranker API: {self.reranker_api_url}")
            return []
        except Exception as e:
            logging.error(f"❌ Reranker API 調用異常: {str(e)}")
            return []
    
    def rerank_search_results(self, query: str, search_results: List[Dict[str, Any]], max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """對搜索結果進行重排序 - 優化版本，利用服務器端排序和截取"""
        if not self.use_rerank or not search_results:
            return search_results
        
        logging.debug(f"🔄 對 {len(search_results)} 個結果進行重排序 (max_results={max_results})")
        
        # 提取文檔內容
        documents = [result['content'] for result in search_results]
        
        # 利用服務器端的 top_n 參數來減少網絡傳輸和處理時間
        # 如果指定了 max_results，讓服務器端只返回需要的數量
        api_top_n = max_results if max_results and max_results < len(documents) else None
        
        # 調用優化的 reranker API
        api_results = self._rerank_with_api(query, documents, top_n=api_top_n)
        
        if not api_results:
            logging.warning("⚠️ Reranker API 調用失敗，使用原始排序")
            # 如果 API 失敗，仍然應用 max_results 限制
            return search_results[:max_results] if max_results else search_results
        
        # 將 API 結果映射回原始搜索結果
        reranked_results = []
        for api_result in api_results:
            original_idx = api_result["index"]
            if original_idx < len(search_results):
                original_result = search_results[original_idx]
                relevance_score = api_result["relevance_score"]
                
                # 更新結果，添加 rerank 信息
                reranked_result = {
                    **original_result,
                    'rerank_score': relevance_score,
                    'rerank_label': "yes" if relevance_score > 0.5 else "no",
                    'original_rank': original_idx + 1  # 記錄原始排名
                }
                reranked_results.append(reranked_result)
        
        logging.debug(f"✅ 重排序完成，API返回 {len(api_results)} 個結果，映射成功 {len(reranked_results)} 個")
        
        # API 應該已經按 relevance_score 排序並限制了數量，但為了安全起見再次確保
        if max_results and len(reranked_results) > max_results:
            reranked_results = reranked_results[:max_results]
            
        return reranked_results

    def calculate_reciprocal_rank(self, query_item: Dict[str, Any], search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """計算倒數排名 (Reciprocal Rank)、R@5 和 R@50 並返回排名"""
        expected_doc_id = query_item.get("expected_doc_id")
        
        for rank, result in enumerate(search_results, 1):
            result_id = result.get("metadata", {}).get("id")
            
            # 如果找到匹配的項目
            if result_id == expected_doc_id:
                rr = 1.0 / rank
                r_at_5 = 1.0 if rank <= 5 else 0.0
                r_at_50 = 1.0 if rank <= 50 else 0.0
                
                logging.debug(f"🎯 找到匹配項目 (ID: {expected_doc_id}) 在排名 {rank}, RR = {rr:.4f}, R@5 = {r_at_5}, R@50 = {r_at_50}")
                
                return {
                    "reciprocal_rank": rr,
                    "rank": rank,
                    "recall_at_5": r_at_5,
                    "recall_at_50": r_at_50
                }
        
        # 沒有找到匹配項目（這種情況不應該發生）
        logging.debug(f"❌ 未找到匹配項目 (ID: {expected_doc_id}), RR = 0")
        return {
            "reciprocal_rank": 0.0,
            "rank": 1027,
            "recall_at_5": 0.0,
            "recall_at_50": 0.0
        }
    
    def get_expected_doc_content(self, doc_id: int) -> str:
        """根據doc_id獲取期望的文檔內容"""
        try:
            # 搜索整個資料庫來找到匹配的文檔
            all_docs = self.vectorstore.get()
            
            for i, metadata in enumerate(all_docs['metadatas']):
                if metadata.get('id') == doc_id:
                    return all_docs['documents'][i]
            
            # 如果沒找到，返回空字符串
            logging.warning(f"⚠️  未找到ID為 {doc_id} 的文檔")
            return ""
        except Exception as e:
            logging.error(f"❌ 獲取期望文檔內容失敗: {e}")
            return ""
    
    async def evaluate_single_query(self, query_item: Dict[str, Any], query_type: str, k: int, query_index: int, save_top_results: int = 3, use_pseudo_doc: bool = False, use_keywords: bool = False) -> Dict[str, Any]:
        """評估單個查詢（異步版本）"""
        total_start_time = time.time()
        
        query_text = self.generate_query_text(query_item, query_type, use_pseudo_doc, use_keywords)
        
        if not query_text:
            logging.warning(f"⚠️  查詢 {query_index} 的文本為空，跳過")
            return None
        
        # 階段1: 執行embedding檢索
        embedding_start_time = time.time()
        search_k = self.rerank_top_n if self.use_rerank else k
        embedding_results = await self.search_with_vectorstore_async(query_text, search_k)
        embedding_time = time.time() - embedding_start_time
        
        # 初始化時間統計
        rerank_time = 0.0
        
        if not embedding_results:
            logging.warning(f"⚠️  查詢 {query_index} 沒有返回結果")
            metrics = {
                "reciprocal_rank": 0.0,
                "rank": 1027,
                "recall_at_5": 0.0,
                "recall_at_50": 0.0
            }
            search_results = []
        else:
            # 如果啟用rerank，進行兩階段檢索
            if self.use_rerank:
                # 階段2: 使用reranker重排序，利用服務器端優化
                rerank_start_time = time.time()
                # 直接指定最終需要的數量，讓服務器端處理排序和截取
                search_results = self.rerank_search_results(query_text, embedding_results, max_results=k)
                rerank_time = time.time() - rerank_start_time
                
                logging.debug(f"🎯 查詢 {query_index} 兩階段檢索: {len(embedding_results)} -> {len(search_results)} (服務器端優化)")
            else:
                search_results = embedding_results
            
            metrics = self.calculate_reciprocal_rank(query_item, search_results)
        
        total_time = time.time() - total_start_time
        
        # 獲取期望文檔內容
        expected_doc_content = self.get_expected_doc_content(query_item.get("expected_doc_id"))
        
        result = {
            "user_id": query_item.get("user_id"),
            "expected_answer": query_item.get("expected_answer"),
            "expected_doc_id": query_item.get("expected_doc_id"),
            "query_text": query_text,
            "reciprocal_rank": metrics["reciprocal_rank"],
            "expected_doc_rank": metrics["rank"],
            "recall_at_5": metrics["recall_at_5"],
            "recall_at_50": metrics["recall_at_50"],
            "search_results": search_results[:save_top_results], # 保留前N個結果
            "expected_doc_content": expected_doc_content,
            "timing": {
                "total_time": total_time,
                "embedding_time": embedding_time,
                "rerank_time": rerank_time,
                "use_rerank": self.use_rerank
            }
        }
        
        return result
    
    async def evaluate_queries_async(self, queries: List[Dict[str, Any]], query_type: str = "prompt", k: int = 1027, save_top_results: int = 3, use_pseudo_doc: bool = False, use_keywords: bool = False) -> Dict[str, Any]:
        """評估查詢列表（異步版本）"""
        logging.info(f"🔍 開始異步評估 {len(queries)} 個查詢...")
        logging.info(f"⚡ 使用 {self.max_concurrent} 個併發連接")
        if use_keywords:
            logging.info(f"🔑 已啟用關鍵詞檢索方法：使用keywords字段進行檢索")
        elif use_pseudo_doc:
            logging.info(f"📄 已啟用Query2Doc方法：將原始query與pseudo_document結合檢索")
        else:
            logging.info(f"📝 使用原始query進行檢索")
        
        if self.use_rerank:
            logging.info(f"🔄 使用兩階段檢索：Embedding Top-{self.rerank_top_n} -> Reranker (批量優化)")
            if len(queries) >= 10:
                logging.info(f"🚀 大批量處理模式：API將使用批量推理加速處理")
        
        # 創建所有查詢任務
        tasks = []
        for i, query_item in enumerate(queries):
            task = self.evaluate_single_query(query_item, query_type, k, i + 1, save_top_results, use_pseudo_doc, use_keywords)
            tasks.append(task)
        
        # 使用 tqdm 顯示進度條並等待所有任務完成
        results_raw = []
        with sync_tqdm(total=len(tasks), desc="🔍 Processing queries", unit="query") as pbar:
            completed_tasks = asyncio.as_completed(tasks)
            for completed_task in completed_tasks:
                result_data = await completed_task
                if result_data is not None:
                    results_raw.append(result_data)
                pbar.update(1)
        
        # 按原始順序排序結果
        results = []
        reciprocal_ranks = []
        recall_at_5_scores = []
        recall_at_50_scores = []
        
        for i, query_item in enumerate(queries):
            # 找到對應的結果
            matching_result = None
            for result in results_raw:
                if result["user_id"] == query_item.get("user_id") and result["expected_doc_id"] == query_item.get("expected_doc_id"):
                    matching_result = result
                    break
            
            if matching_result:
                results.append(matching_result)
                reciprocal_ranks.append(matching_result["reciprocal_rank"])
                recall_at_5_scores.append(matching_result["recall_at_5"])
                recall_at_50_scores.append(matching_result["recall_at_50"])
        
        # 計算平均指標
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        mean_recall_at_5 = sum(recall_at_5_scores) / len(recall_at_5_scores) if recall_at_5_scores else 0.0
        mean_recall_at_50 = sum(recall_at_50_scores) / len(recall_at_50_scores) if recall_at_50_scores else 0.0
        
        # 計算時間統計
        total_times = [result.get('timing', {}).get('total_time', 0) for result in results]
        embedding_times = [result.get('timing', {}).get('embedding_time', 0) for result in results]
        rerank_times = [result.get('timing', {}).get('rerank_time', 0) for result in results]
        
        timing_stats = {
            "avg_total_time": sum(total_times) / len(total_times) if total_times else 0.0,
            "avg_embedding_time": sum(embedding_times) / len(embedding_times) if embedding_times else 0.0,
            "avg_rerank_time": sum(rerank_times) / len(rerank_times) if rerank_times else 0.0,
            "total_queries_time": sum(total_times),
            "total_embedding_time": sum(embedding_times),
            "total_rerank_time": sum(rerank_times)
        }
        
        evaluation_summary = {
            "database": self.db_name,
            "embedding_model": self.embedding_model,
            "query_type": query_type,
            "use_pseudo_doc": use_pseudo_doc,
            "use_keywords": use_keywords,
            "use_rerank": self.use_rerank,
            "reranker_model": self.reranker_model if self.use_rerank else None,
            "rerank_top_n": self.rerank_top_n if self.use_rerank else None,
            "num_queries": len(queries),
            "num_evaluated": len(results),
            "mean_reciprocal_rank": mrr,
            "mean_recall_at_5": mean_recall_at_5,
            "mean_recall_at_50": mean_recall_at_50,
            "timing_stats": timing_stats,
            "individual_results": results
        }
        
        logging.info(f"📊 異步評估完成!")
        logging.info(f"📈 平均倒數排名 (MRR): {mrr:.4f}")
        logging.info(f"📈 平均 Recall@5: {mean_recall_at_5:.4f}")
        logging.info(f"📈 平均 Recall@50: {mean_recall_at_50:.4f}")
        logging.info(f"⏱️  平均查詢時間: {timing_stats['avg_total_time']:.3f}秒")
        logging.info(f"🔍 平均embedding時間: {timing_stats['avg_embedding_time']:.3f}秒")
        if self.use_rerank:
            logging.info(f"🔄 平均rerank時間: {timing_stats['avg_rerank_time']:.3f}秒")
        
        return evaluation_summary
    
    def evaluate_queries(self, queries: List[Dict[str, Any]], query_type: str = "prompt", k: int = 1027, save_top_results: int = 3, use_pseudo_doc: bool = False, use_keywords: bool = False) -> Dict[str, Any]:
        """評估查詢列表（同步入口）"""
        # 運行異步版本
        return asyncio.run(self.evaluate_queries_async(queries, query_type, k, save_top_results, use_pseudo_doc, use_keywords))
    
    def save_results(self, results: Dict[str, Any], output_folder: str = None, use_pseudo_doc: bool = False, use_keywords: bool = False) -> str:
        """保存評估結果到文件夾"""
        # 創建基礎outputs目錄
        os.makedirs("outputs", exist_ok=True)
        
        if output_folder is None:
            # 使用默認文件夾命名
            folder_parts = [f"run_{self.db_name}_{results['num_queries']}"]
            
            if use_keywords:
                folder_parts.append("with_keywords")
            elif use_pseudo_doc:
                folder_parts.append("with_pseudo")
            
            if self.use_rerank:
                # 將模型名稱轉換為適合文件夾名稱的格式（替換特殊字符）
                safe_model_name = self.reranker_model.replace("/", "_").replace(":", "_")
                folder_parts.append(f"with_{safe_model_name}")
            
            folder_name = "_".join(folder_parts)
            output_folder = f"outputs/{folder_name}"
        else:
            # 如果用戶指定了文件夾但沒有包含outputs路徑，自動添加
            if not output_folder.startswith("outputs/"):
                output_folder = f"outputs/{output_folder}"
        
        # 創建輸出文件夾
        os.makedirs(output_folder, exist_ok=True)
        
        # 保存主要評估結果
        result_file = os.path.join(output_folder, "evaluation_results.json")
        
        try:
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logging.info(f"💾 結果已保存到文件夾: {output_folder}")
            logging.info(f"📋 評估結果文件: {result_file}")
            
            return output_folder  # 返回文件夾路徑供其他方法使用
        except Exception as e:
            logging.error(f"❌ 保存結果失敗: {e}")
            return None
    
    def print_summary(self, results: Dict[str, Any]):
        """打印評估摘要"""
        print("\n" + "="*70)
        print("📊 檢索評估結果摘要")
        print("="*70)
        print(f"數據庫: {results['database']}")
        print(f"Embedding模型: {results['embedding_model']}")
        print(f"查詢類型: {results['query_type']}")
        print(f"關鍵詞檢索: {'已啟用' if results.get('use_keywords', False) else '未啟用'}")
        print(f"Query2Doc: {'已啟用' if results.get('use_pseudo_doc', False) else '未啟用'}")
        print(f"兩階段檢索: {'已啟用' if results.get('use_rerank', False) else '未啟用'}")
        if results.get('use_rerank', False):
            print(f"Reranker模型: {results.get('reranker_model', 'N/A')}")
            print(f"Rerank Top-N: {results.get('rerank_top_n', 'N/A')}")
        print(f"查詢數量: {results['num_queries']}")
        print(f"評估數量: {results['num_evaluated']}")
        print(f"平均倒數排名 (MRR): {results['mean_reciprocal_rank']:.4f}")
        print(f"平均 Recall@5: {results['mean_recall_at_5']:.4f}")
        print(f"平均 Recall@50: {results['mean_recall_at_50']:.4f}")
        
        # 顯示時間統計
        timing_stats = results.get('timing_stats', {})
        if timing_stats:
            print(f"\n⏱️  時間統計:")
            print(f"平均查詢時間: {timing_stats.get('avg_total_time', 0):.3f}秒")
            print(f"平均embedding時間: {timing_stats.get('avg_embedding_time', 0):.3f}秒")
            if results.get('use_rerank', False):
                print(f"平均rerank時間: {timing_stats.get('avg_rerank_time', 0):.3f}秒")
                rerank_ratio = (timing_stats.get('avg_rerank_time', 0) / timing_stats.get('avg_total_time', 1)) * 100
                print(f"rerank佔比: {rerank_ratio:.1f}%")
            print(f"總處理時間: {timing_stats.get('total_queries_time', 0):.3f}秒")
        
        # 顯示個別結果
        print("\n📋 個別查詢結果:")
        if results.get('use_rerank', False):
            print(f"{'No.':<4} {'Expected Answer':<20} {'RR':<8} {'R@5':<6} {'R@50':<6} {'總時間':<8} {'Emb':<6} {'Rerank':<6}")
            print("-" * 84)
        else:
            print(f"{'No.':<4} {'Expected Answer':<25} {'RR':<8} {'R@5':<6} {'R@50':<6} {'時間':<8}")
            print("-" * 70)
        
        for i, result in enumerate(results['individual_results'], 1):
            expected_answer = result.get('expected_answer', 'N/A')
            rr = result['reciprocal_rank']
            r5 = result['recall_at_5']
            r50 = result['recall_at_50']
            timing = result.get('timing', {})
            total_time = timing.get('total_time', 0)
            emb_time = timing.get('embedding_time', 0)
            rerank_time = timing.get('rerank_time', 0)
            
            if results.get('use_rerank', False):
                print(f"{i:<4} {expected_answer[:19]:<20} {rr:<8.4f} {r5:<6.1f} {r50:<6.1f} {total_time:<8.3f} {emb_time:<6.3f} {rerank_time:<6.3f}")
            else:
                print(f"{i:<4} {expected_answer[:24]:<25} {rr:<8.4f} {r5:<6.1f} {r50:<6.1f} {total_time:<8.3f}")
        
        print("="*70)
    
    def visualize_ranking_distribution(self, results: Dict[str, Any], output_folder: str):
        """生成expected_doc_id排名分佈可視化圖表（分布圖和累積圖分開保存），包含l2_distance平均值折線圖"""
        try:
            # 確保輸出文件夾存在
            os.makedirs(output_folder, exist_ok=True)
            
            # 設定圖表輸出路徑
            hist_output_file = os.path.join(output_folder, "ranking_histogram.png")
            cumulative_output_file = os.path.join(output_folder, "ranking_cumulative.png")
            
            # 從已保存的結果中獲取排名和距離信息
            rankings = []
            rank_distances = {}  # 用於收集每個排名位置的所有距離值
            total_queries = len(results['individual_results'])
            
            print(f"🔍 Extracting ranking positions and distances from {total_queries} queries...")
            
            for i, result in enumerate(results['individual_results'], 1):
                expected_doc_id = result['expected_doc_id']
                expected_doc_rank = result.get('expected_doc_rank', 1027)
                search_results = result.get('search_results', [])
                
                rankings.append(expected_doc_rank)
                print(f"   Query {i}: Expected doc_id {expected_doc_id} found at rank {expected_doc_rank}")
                
                # 收集每個排名位置的l2_distance
                for rank_idx, search_result in enumerate(search_results):
                    rank_pos = rank_idx + 1  # 排名從1開始
                    distance = search_result.get('distance_score', 0)
                    
                    if rank_pos not in rank_distances:
                        rank_distances[rank_pos] = []
                    rank_distances[rank_pos].append(distance)
            
            # 計算每個排名位置的平均l2_distance和變化範圍
            avg_distances_by_rank = {}
            std_distances_by_rank = {}
            min_distances_by_rank = {}
            max_distances_by_rank = {}
            max_rank_with_data = 0
            
            for rank, distances in rank_distances.items():
                if distances:  # 確保有數據
                    avg_distances_by_rank[rank] = np.mean(distances)
                    std_distances_by_rank[rank] = np.std(distances) if len(distances) > 1 else 0.0
                    min_distances_by_rank[rank] = np.min(distances)
                    max_distances_by_rank[rank] = np.max(distances)
                    max_rank_with_data = max(max_rank_with_data, rank)
            
            print(f"📊 Calculated average l2_distance for ranks 1-{max_rank_with_data}")
            
            # 第一張圖：分布直方圖 + l2_distance折線圖
            fig, ax1 = plt.subplots(figsize=(12, 8))
            
            # 主軸：分布直方圖
            bins = range(1, 1029)  # 從1到1028，包含1027個排名位置
            n, bins_edges, patches = ax1.hist(rankings, bins=bins, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
            ax1.set_xlabel('Rank of Expected Doc ID', fontsize=12)
            ax1.set_ylabel('Count', fontsize=12, color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_xlim(1, min(1027, max_rank_with_data + 50))
            ax1.grid(True, alpha=0.3)
            
            # 添加統計信息到分布圖
            if rankings:
                mean_rank = np.mean(rankings)
                median_rank = np.median(rankings)
                ax1.axvline(mean_rank, color='red', linestyle='--', alpha=0.7, label=f'Mean Rank: {mean_rank:.1f}')
                ax1.axvline(median_rank, color='orange', linestyle='--', alpha=0.7, label=f'Median Rank: {median_rank:.1f}')
            
            # 副軸：l2_distance折線圖
            ax2 = ax1.twinx()
            if avg_distances_by_rank:
                ranks_for_line = sorted(avg_distances_by_rank.keys())
                distances_for_line = [avg_distances_by_rank[rank] for rank in ranks_for_line]
                
                # 計算上下界範圍（使用標準差）
                std_upper = [avg_distances_by_rank[rank] + std_distances_by_rank[rank] for rank in ranks_for_line]
                std_lower = [avg_distances_by_rank[rank] - std_distances_by_rank[rank] for rank in ranks_for_line]
                
                # 繪製範圍帶（淺綠色）
                ax2.fill_between(ranks_for_line, std_lower, std_upper, 
                                color='lightgreen', alpha=0.3, label='±1 Std Dev Range')
                
                # 繪製平均值折線圖（綠色）
                ax2.plot(ranks_for_line, distances_for_line, color='darkgreen', linewidth=2, marker='o', 
                        markersize=3, alpha=0.8, label='Average L2 Distance')
                ax2.set_ylabel('Average L2 Distance', fontsize=12, color='darkgreen')
                ax2.tick_params(axis='y', labelcolor='darkgreen')
                
                # 添加百分位數標記（PR25, 50, 75）
                percentiles = [25, 50, 75]
                percentile_colors = ['purple', 'magenta', 'skyblue']
                percentile_values = np.percentile(distances_for_line, percentiles)
                
                for i, (pct, value, color) in enumerate(zip(percentiles, percentile_values, percentile_colors)):
                    ax2.axhline(value, color=color, linestyle='-.', alpha=0.8, linewidth=1.5,
                               label=f'PR{pct}: {value:.3f}')

            # 統一標題和圖例
            plt.title(f'Ranking Distribution with Average L2 Distance\n(Total Queries: {total_queries})', fontsize=14)
            
            # 合併圖例
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            plt.tight_layout()
            plt.savefig(hist_output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 第二張圖：累積分布圖
            plt.figure(figsize=(10, 6))
            sorted_rankings = sorted(rankings)
            # 計算累積比例
            cumulative_counts = np.arange(1, len(sorted_rankings) + 1) / len(sorted_rankings)
            
            plt.plot(sorted_rankings, cumulative_counts, marker='o', markersize=4, 
                    linewidth=2, color='darkgreen', alpha=0.8)
            plt.xlabel('Rank of Expected Doc ID')
            plt.ylabel('Cumulative Proportion')
            plt.title(f'Cumulative Distribution of Rankings\n(Total Queries: {total_queries})')
            plt.grid(True, alpha=0.3)
            plt.xlim(1, 1027)
            plt.ylim(0, 1)
            
            # 添加重要的累積指標線，每個使用不同顏色
            recall_points = [5, 10, 20, 50, 100]
            colors = ['red', 'blue', 'purple', 'orange', 'brown']
            
            for i, recall_k in enumerate(recall_points):
                recall_proportion = sum(1 for rank in rankings if rank <= recall_k) / len(rankings)
                if recall_proportion > 0:
                    color = colors[i % len(colors)]
                    plt.axhline(recall_proportion, color=color, linestyle=':', alpha=0.7, 
                               label=f'Recall@{recall_k}: {recall_proportion:.4f}')
                    plt.axvline(recall_k, color=color, linestyle=':', alpha=0.7)
            
            plt.legend(fontsize=9)
            plt.tight_layout()
            plt.savefig(cumulative_output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Histogram chart saved to: {hist_output_file}")
            print(f"📊 Cumulative distribution chart saved to: {cumulative_output_file}")
            
            # 打印詳細統計信息
            print(f"📈 Ranking Statistics:")
            print(f"   Total queries analyzed: {total_queries}")
            print(f"   Mean rank: {np.mean(rankings):.1f}")
            print(f"   Median rank: {np.median(rankings):.1f}")
            print(f"   Best rank: {min(rankings)}")
            print(f"   Worst rank: {max(rankings)}")
            
            # 打印L2距離統計信息
            if avg_distances_by_rank:
                all_avg_distances = list(avg_distances_by_rank.values())
                all_std_distances = list(std_distances_by_rank.values())
                print(f"📏 L2 Distance Statistics:")
                print(f"   Ranks with distance data: 1-{max_rank_with_data}")
                print(f"   Mean average distance: {np.mean(all_avg_distances):.4f}")
                print(f"   Median average distance: {np.median(all_avg_distances):.4f}")
                print(f"   Mean standard deviation: {np.mean(all_std_distances):.4f}")
                print(f"   Min average distance: {min(all_avg_distances):.4f} (rank {min(avg_distances_by_rank, key=avg_distances_by_rank.get)})")
                print(f"   Max average distance: {max(all_avg_distances):.4f} (rank {max(avg_distances_by_rank, key=avg_distances_by_rank.get)})")
                
                # 顯示變化範圍最大的幾個rank
                rank_variance = [(rank, std) for rank, std in std_distances_by_rank.items() if std > 0]
                if rank_variance:
                    rank_variance.sort(key=lambda x: x[1], reverse=True)
                    print(f"   Top 3 ranks with highest variance:")
                    for i, (rank, std) in enumerate(rank_variance[:3]):
                        print(f"     Rank {rank}: std={std:.4f}, avg={avg_distances_by_rank[rank]:.4f}")
            
            # 添加累積統計信息
            print(f"📊 Cumulative Statistics:")
            for recall_k in [5, 10, 20, 50, 100]:
                count_within_k = sum(1 for rank in rankings if rank <= recall_k)
                proportion = count_within_k / total_queries
                # 同時顯示該rank處的平均l2距離和標準差
                avg_distance_at_k = avg_distances_by_rank.get(recall_k, "N/A")
                std_distance_at_k = std_distances_by_rank.get(recall_k, "N/A")
                if isinstance(avg_distance_at_k, float) and isinstance(std_distance_at_k, float):
                    print(f"   Recall@{recall_k}: {count_within_k}/{total_queries} = {proportion:.4f}, Avg L2 distance: {avg_distance_at_k:.4f}±{std_distance_at_k:.4f}")
                else:
                    print(f"   Recall@{recall_k}: {count_within_k}/{total_queries} = {proportion:.4f}, Avg L2 distance: {avg_distance_at_k}")
            
            print(f"   Rank distribution: {sorted(rankings)}")
                
        except Exception as e:
            logging.error(f"❌ 可視化生成失敗: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_simple_recall_curve(self, results: Dict[str, Any], output_folder: str):
        """生成精簡的recall曲線圖（只顯示recall@5,10,20,50,100五個關鍵點）"""
        try:
            # 確保輸出文件夾存在
            os.makedirs(output_folder, exist_ok=True)
            
            # 設定圖表輸出路徑
            recall_curve_output_file = os.path.join(output_folder, "recall_curve_simplified.png")
            
            # 從結果中獲取排名信息
            rankings = []
            for result in results['individual_results']:
                expected_doc_rank = result.get('expected_doc_rank', 1027)
                rankings.append(expected_doc_rank)
            
            total_queries = len(rankings)
            
            # 計算關鍵recall點
            recall_k_values = [5, 10, 20, 50, 100]
            recall_proportions = []
            
            for recall_k in recall_k_values:
                count_within_k = sum(1 for rank in rankings if rank <= recall_k)
                proportion = count_within_k / total_queries
                recall_proportions.append(proportion)
            
            # 生成精簡的recall曲線圖
            plt.figure(figsize=(10, 6))
            plt.plot(recall_k_values, recall_proportions, 
                    marker='o', markersize=6, linewidth=3, 
                    color='darkblue', alpha=0.8, markerfacecolor='darkblue', 
                    markeredgecolor='darkblue', markeredgewidth=2)
            
            plt.xlabel('K (Rank Threshold)', fontsize=12)
            plt.ylabel('Recall@K', fontsize=12)
            plt.title(f'Simplified Recall Curve\n(Total Queries: {total_queries})', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xlim(0, 1027)
            plt.ylim(0, max(1.0, max(recall_proportions) + 0.1))
            
            plt.tight_layout()
            plt.savefig(recall_curve_output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Simplified recall curve saved to: {recall_curve_output_file}")
            
            # 打印recall統計信息
            print(f"📈 Recall Statistics:")
            for k, recall in zip(recall_k_values, recall_proportions):
                count = int(recall * total_queries)
                print(f"   Recall@{k}: {count}/{total_queries} = {recall:.3f}")
                
        except Exception as e:
            logging.error(f"❌ 精簡recall曲線生成失敗: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="檢索評估工具")
    parser.add_argument("--json-file", default="data/tcm_sd_test_rc_direct.json",
                       help="測試數據JSON文件路徑 (默認: data/tcm_sd_test_rc_direct.json)")
    parser.add_argument("--db-name", default="syndrome_db",
                       help="ChromaDB數據庫名稱 (默認: syndrome_db)")
    parser.add_argument("--num-queries", type=int, default=10,
                       help="隨機選擇的查詢數量 (默認: 10)")
    parser.add_argument("--query-type", choices=["prompt"],
                       default="prompt", help="查詢文本類型 (默認: prompt)")
    parser.add_argument("--k", type=int, default=1027,
                       help="檢索結果數量 (默認: 1027)")
    parser.add_argument("--output-folder",
                       help="輸出文件夾路徑 (可選, 默認為 outputs/run_{db_name}_{num_queries})")
    parser.add_argument("--seed", type=int, default=42,
                       help="隨機種子 (可重現結果)")
    parser.add_argument("--embedding-model", default="text-embedding-3-large",
                       help="Embedding模型 (默認: text-embedding-3-large)。格式: 'provider:model' 或 'model'。支持 openai, vllm, custom, huggingface")
    parser.add_argument("--max-concurrent", type=int, default=10,
                       help="最大併發請求數 (默認: 10)")
    parser.add_argument("--save-top-results", type=int, default=1027,
                       help="保存到結果中的搜索結果數量 (默認: 1027)")
    parser.add_argument("--use-pseudo-doc", action="store_true", default=False,
                       help="啟用Query2Doc方法：將原始query與pseudo_document結合進行檢索 (默認: False)")
    parser.add_argument("--use-keywords", action="store_true", default=False,
                       help="啟用關鍵詞檢索方法：使用keywords字段進行檢索 (默認: False)")
    parser.add_argument("--list-models", action="store_true",
                       help="列出所有可用的embedding模型並退出")
    parser.add_argument("--rerank", action="store_true", default=False,
                       help="啟用兩階段檢索：先用embedding排序取top-k，再用reranker重排 (默認: False)")
    parser.add_argument("--reranker-api-url", default="http://localhost:8001",
                       help="Reranker API URL (默認: http://localhost:8001)")
    parser.add_argument("--reranker-model", default="Qwen/Qwen3-Reranker-0.6B",
                       help="Reranker模型名稱 (默認: Qwen/Qwen3-Reranker-0.6B)")
    parser.add_argument("--rerank-top-n", type=int, default=100,
                       help="送給reranker的候選文檔數量 (默認: 100)")
    
    args = parser.parse_args()
    
    # Handle --list-models option
    if args.list_models:
        print("Available embedding models by provider:")
        print("=" * 50)
        
        for provider, config in EMBEDDING_PROVIDERS.items():
            print(f"\n{provider.upper()}:")
            for model in config["models"]:
                default_marker = " (default)" if model == config["default"] else ""
                print(f"  - {provider}:{model}{default_marker}")
        
        print(f"\nUsage examples:")
        print(f"  --embedding-model openai:text-embedding-3-large")
        print(f"  --embedding-model vllm:Qwen3-Embedding-8B")
        print(f"  --embedding-model custom:Qwen3-Embedding-0.6B-finetuned")
        print(f"  --embedding-model huggingface:BAAI/bge-large-zh-v1.5")
        print(f"  --embedding-model text-embedding-3-large  # OpenAI provider")
        return
    
    # 檢查embedding模型的要求
    try:
        provider, model = get_provider_and_model(args.embedding_model)
        
        # 只有當使用OpenAI提供者時才檢查API Key
        if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            print("❌ OPENAI_API_KEY 環境變量未設置")
            print("請設置 API key 後再運行")
            sys.exit(1)
            
        logging.info(f"使用 {provider} embedding 模型: {model}")
        
    except ValueError as e:
        logging.error(str(e))
        print("使用 --list-models 查看可用的embedding模型")
        sys.exit(1)
    
    # 設置隨機種子
    if args.seed:
        random.seed(args.seed)
        logging.info(f"🎲 設置隨機種子: {args.seed}")
    
    try:
        # 初始化評估器
        evaluator = RetrievalEvaluator(
            db_name=args.db_name, 
            embedding_model=args.embedding_model,
            max_concurrent=args.max_concurrent,
            use_rerank=args.rerank,
            reranker_api_url=args.reranker_api_url,
            reranker_model=args.reranker_model,
            rerank_top_n=args.rerank_top_n
        )
        
        # 加載測試數據
        data = evaluator.load_test_data(args.json_file)
        
        # 選擇隨機查詢
        queries = evaluator.select_random_queries(data, args.num_queries)
        
        # 顯示查詢模式
        if args.use_keywords:
            logging.info(f"🔑 啟用關鍵詞檢索模式")
        elif args.use_pseudo_doc:
            logging.info(f"🔄 啟用Query2Doc實驗模式")
        else:
            logging.info(f"📝 使用標準查詢模式")
        
        # 執行評估
        results = evaluator.evaluate_queries(queries, args.query_type, args.k, args.save_top_results, args.use_pseudo_doc, args.use_keywords)
        
        # 顯示結果
        evaluator.print_summary(results)
        
        # 保存結果
        output_folder = evaluator.save_results(results, args.output_folder, args.use_pseudo_doc, args.use_keywords)
        
        # 可視化圖表
        if output_folder:
            evaluator.visualize_ranking_distribution(results, output_folder)
            evaluator.generate_simple_recall_curve(results, output_folder)
        
    except Exception as e:
        logging.error(f"❌ 程序執行失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
