#!/usr/bin/env python3
"""
TCM-SD Prompt Generator for LLM Evaluation
將 TCM-SD 資料轉換為適合評估 GPT-4o-mini 等 LLM 的 prompt 格式
"""

import json
import argparse
import random
from typing import List, Dict, Any
from pathlib import Path


class TCMPromptGenerator:
    def __init__(self, seed: int = None):
        self.syndrome_vocab = []
        self.syndrome_knowledge = {}
        self.syndrome_id_mapping = {}
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
            print(f"已設定隨機種子: {self.seed}")
        self.load_syndrome_vocab()
        self.load_syndrome_knowledge()
        self.load_syndrome_id_mapping()
    
    def load_syndrome_vocab(self, vocab_path: str = "Data/TCM_SD/syndrome_vocab.txt"):
        """載入證候詞彙表"""
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.syndrome_vocab = [line.strip() for line in f if line.strip()]
            print(f"已載入 {len(self.syndrome_vocab)} 個證候類別")
        except FileNotFoundError:
            print(f"警告: 找不到證候詞彙表檔案 {vocab_path}")
            self.syndrome_vocab = []
    
    def load_syndrome_knowledge(self, knowledge_path: str = "Data/TCM_SD/syndrome_knowledge.json"):
        """載入證候知識庫"""
        try:
            with open(knowledge_path, 'r', encoding='utf-8') as f:
                knowledge_data = []
                for line in f:
                    if line.strip():
                        knowledge_data.append(json.loads(line.strip()))
                
                # 將知識轉換為字典，方便查詢
                self.syndrome_knowledge = {}
                for item in knowledge_data:
                    syndrome_name = item.get('Name', '')
                    if syndrome_name:
                        self.syndrome_knowledge[syndrome_name] = {
                            'definition': item.get('Definition', ''),
                            'typical_performance': item.get('Typical_performance', ''),
                            'common_disease': item.get('Common_isease', '')  # Note: 原文件使用的是 Common_isease
                        }
                print(f"已載入 {len(self.syndrome_knowledge)} 個證候知識條目")
        except FileNotFoundError:
            print(f"警告: 找不到證候知識庫檔案 {knowledge_path}")
            self.syndrome_knowledge = {}
    
    def load_syndrome_id_mapping(self, mapping_path: str = "output/syndrome_knowledge_fixed.json"):
        """載入證候名稱到id的映射"""
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                syndrome_data = json.load(f)
                
                # 建立證候名稱到id的映射
                self.syndrome_id_mapping = {}
                for item in syndrome_data:
                    syndrome_name = item.get('Name', '')
                    syndrome_id = item.get('id', None)
                    if syndrome_name and syndrome_id is not None:
                        self.syndrome_id_mapping[syndrome_name] = syndrome_id
                print(f"已載入 {len(self.syndrome_id_mapping)} 個證候ID映射")
        except FileNotFoundError:
            print(f"警告: 找不到證候ID映射檔案 {mapping_path}")
            self.syndrome_id_mapping = {}
    
    def get_syndrome_id(self, syndrome_name: str) -> int:
        """獲取證候的id"""
        return self.syndrome_id_mapping.get(syndrome_name, None) 
    
    def get_syndrome_knowledge_text(self, syndrome_name: str) -> str:
        """獲取證候的詳細知識描述 - 包含所有可用信息"""
        if syndrome_name not in self.syndrome_knowledge:
            return f"{syndrome_name}：暫無詳細知識描述。"
        
        knowledge = self.syndrome_knowledge[syndrome_name]
        definition = knowledge.get('definition', '')
        typical_performance = knowledge.get('typical_performance', '')
        common_disease = knowledge.get('common_disease', '')
        
        # 構建完整的知識描述
        knowledge_text = f"{syndrome_name}：{definition}"
        
        if typical_performance:
            knowledge_text += f" 典型表現：{typical_performance}"
        
        if common_disease:
            knowledge_text += f" 常見疾病：{common_disease}"
        
        return knowledge_text
    
    def load_tcm_data(self, data_path: str) -> List[Dict[str, Any]]:
        """載入 TCM-SD 資料"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data
    
    def format_direct_diagnosis(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """直接診斷格式 - 讓模型直接預測證候"""
        
        # 建構病情描述
        clinical_info = []
        if sample.get('chief_complaint'):
            clinical_info.append(f"主訴：{sample['chief_complaint']}")
        if sample.get('description'):
            clinical_info.append(f"現病史：{sample['description']}")
        if sample.get('detection'):
            clinical_info.append(f"體格檢查：{sample['detection']}")
        
        clinical_text = "\n".join(clinical_info)
        
        prompt = f"""
患者临床信息：
{clinical_text}
"""
        
        return {
            "user_id": sample.get('user_id', ''),
            "prompt": prompt,
            "expected_answer": sample.get('norm_syndrome', ''),
            "format_type": "direct_diagnosis",
            "expected_doc_id": self.get_syndrome_id(sample.get('norm_syndrome', ''))
        }
    
    def format_multiple_choice(self, sample: Dict[str, Any], num_options: int = 5) -> Dict[str, str]:
        """多選題格式 - 提供候選選項"""
        
        # 建構病情描述
        clinical_info = []
        if sample.get('chief_complaint'):
            clinical_info.append(f"主诉：{sample['chief_complaint']}")
        if sample.get('description'):
            clinical_info.append(f"现病史：{sample['description']}")
        if sample.get('detection'):
            clinical_info.append(f"体格检查：{sample['detection']}")
        
        clinical_text = "\n".join(clinical_info)
        
        # 建立選項（包含正確答案和干擾項）
        correct_answer = sample.get('norm_syndrome', '')
        options = [correct_answer]
        
        # 加入干擾項 - 隨機選擇
        available_syndromes = [s for s in self.syndrome_vocab if s != correct_answer]
        num_distractors_needed = min(num_options - 1, len(available_syndromes))
        if num_distractors_needed > 0:
            selected_distractors = random.sample(available_syndromes, num_distractors_needed)
            options.extend(selected_distractors)
        
        # 如果證候詞彙表不夠，用樣本中的其他欄位
        if len(options) < num_options and sample.get('syndrome'):
            if sample['syndrome'] != correct_answer and sample['syndrome'] not in options:
                options.append(sample['syndrome'])
        
        # 打亂選項順序（簡單輪換）
        random.shuffle(options)
        
        # 找到正確答案的位置
        correct_index = options.index(correct_answer)
        correct_letter = chr(ord('A') + correct_index)
        
        # 建構選項文字
        option_text = ""
        for i, option in enumerate(options):
            letter = chr(ord('A') + i)
            option_text += f"{letter}. {option}\n"
        
        prompt = f"""
患者临床信息：
{clinical_text}

请从以下选项中选择：
{option_text}
"""
        
        return {
            "user_id": sample.get('user_id', ''),
            "prompt": prompt,
            "expected_answer": correct_letter,
            "expected_syndrome": correct_answer,
            "options": options,
            "format_type": "multiple_choice",
            "expected_doc_id": self.get_syndrome_id(correct_answer)
        }
    
    def format_multiple_choice_upperbound(self, sample: Dict[str, Any], num_options: int = 5) -> Dict[str, str]:
        """多選題格式上界 - 提供候選選項和詳細證候知識"""
        
        # 建構病情描述
        clinical_info = []
        if sample.get('chief_complaint'):
            clinical_info.append(f"主诉：{sample['chief_complaint']}")
        if sample.get('description'):
            clinical_info.append(f"现病史：{sample['description']}")
        if sample.get('detection'):
            clinical_info.append(f"体格检查：{sample['detection']}")
        
        clinical_text = "\n".join(clinical_info)
        
        # 建立選項（包含正確答案和干擾項）
        correct_answer = sample.get('norm_syndrome', '')
        options = [correct_answer]
        
        # 加入干擾項 - 隨機選擇
        available_syndromes = [s for s in self.syndrome_vocab if s != correct_answer]
        num_distractors_needed = min(num_options - 1, len(available_syndromes))
        if num_distractors_needed > 0:
            selected_distractors = random.sample(available_syndromes, num_distractors_needed)
            options.extend(selected_distractors)
        
        # 如果證候詞彙表不夠，用樣本中的其他欄位
        if len(options) < num_options and sample.get('syndrome'):
            if sample['syndrome'] != correct_answer and sample['syndrome'] not in options:
                options.append(sample['syndrome'])
        
        # 打亂選項順序（簡單輪換）
        random.shuffle(options)
        
        # 找到正確答案的位置
        correct_index = options.index(correct_answer)
        correct_letter = chr(ord('A') + correct_index)
        
        # 構建詳細的背景知識
        knowledge_sections = []
        for i, syndrome in enumerate(options, 1):
            knowledge_text = self.get_syndrome_knowledge_text(syndrome)
            knowledge_sections.append(f"{i}. {knowledge_text}")
        
        knowledge_context = "\n\n".join(knowledge_sections)
        
        # 建構選項文字
        option_text = ""
        for i, option in enumerate(options):
            letter = chr(ord('A') + i)
            option_text += f"{letter}. {option}\n"
        
        prompt = f"""
中医证候知识：
{knowledge_context}

患者临床信息：
{clinical_text}

请从以下选项中选择：
{option_text}
"""
        
        return {
            "user_id": sample.get('user_id', ''),
            "prompt": prompt,
            "expected_answer": correct_letter,
            "expected_syndrome": correct_answer,
            "options": options,
            "format_type": "multiple_choice_upperbound",
            "expected_doc_id": self.get_syndrome_id(correct_answer)
        }

    def format_reading_comprehension_with_knowledge(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """閱讀理解格式 - 包含5個證候的詳細背景知識"""
        
        # 建構問題（病情描述）
        clinical_info = []
        if sample.get('chief_complaint'):
            clinical_info.append(f"主诉：{sample['chief_complaint']}")
        if sample.get('description'):
            clinical_info.append(f"现病史：{sample['description']}")
        if sample.get('detection'):
            clinical_info.append(f"体格检查：{sample['detection']}")
        
        clinical_text = "\n".join(clinical_info)
        
        # 獲取正確答案和候選證候
        correct_answer = sample.get('norm_syndrome', '')
        candidate_syndromes = [correct_answer]
        
        # 加入干擾項 - 隨機選擇4個
        available_syndromes = [s for s in self.syndrome_vocab if s != correct_answer]
        num_distractors_needed = min(4, len(available_syndromes))
        if num_distractors_needed > 0:
            selected_distractors = random.sample(available_syndromes, num_distractors_needed)
            candidate_syndromes.extend(selected_distractors)
        
        # 如果證候詞彙表不夠，用樣本中的其他欄位
        if len(candidate_syndromes) < 5 and sample.get('syndrome'):
            if sample['syndrome'] != correct_answer and sample['syndrome'] not in candidate_syndromes:
                candidate_syndromes.append(sample['syndrome'])
        
        # 隨機打亂候選證候順序（避免位置偏見）
        random.shuffle(candidate_syndromes)
        
        # 構建詳細的背景知識
        knowledge_sections = []
        for i, syndrome in enumerate(candidate_syndromes, 1):
            knowledge_text = self.get_syndrome_knowledge_text(syndrome)
            knowledge_sections.append(f"{i}. {knowledge_text}")
        
        knowledge_context = "\n\n".join(knowledge_sections)
        
        # 建構選項文字（候選證候名稱）
        option_text = "、".join(candidate_syndromes) + "等"
        
        prompt = f"""
中医证候知识：
{knowledge_context}

患者临床信息：
{clinical_text}

请从以下选项中选择：
{option_text}
"""
        
        return {
            "user_id": sample.get('user_id', ''),
            "prompt": prompt,
            "expected_answer": correct_answer,
            "candidate_syndromes": candidate_syndromes,
            "format_type": "reading_comprehension_five_upperbound",
            "expected_doc_id": self.get_syndrome_id(correct_answer)
        }
    
    def format_reading_comprehension(self, sample: Dict[str, Any], use_five_syndromes: bool = False) -> Dict[str, str]:
        """閱讀理解格式 - 類似原始MRC但適合LLM"""
        
        # 建構問題（病情描述）
        clinical_info = []
        if sample.get('chief_complaint'):
            clinical_info.append(f"主诉：{sample['chief_complaint']}")
        if sample.get('description'):
            clinical_info.append(f"现病史：{sample['description']}")
        if sample.get('detection'):
            clinical_info.append(f"体格检查：{sample['detection']}")
        
        clinical_text = "\n".join(clinical_info)
        
        # 建構選項文字
        option_text = ""
        if hasattr(sample, 'knowledge_para') and sample.get('knowledge_para'):
            option_text = sample['knowledge_para']
        else:
            if use_five_syndromes:
                # 使用5個候選證候（類似multiple choice邏輯）
                correct_answer = sample.get('norm_syndrome', '')
                candidate_syndromes = [correct_answer]
                
                # 加入干擾項 - 隨機選擇4個
                available_syndromes = [s for s in self.syndrome_vocab if s != correct_answer]
                num_distractors_needed = min(4, len(available_syndromes))
                if num_distractors_needed > 0:
                    selected_distractors = random.sample(available_syndromes, num_distractors_needed)
                    candidate_syndromes.extend(selected_distractors)
                
                # 如果證候詞彙表不夠，用樣本中的其他欄位
                if len(candidate_syndromes) < 5 and sample.get('syndrome'):
                    if sample['syndrome'] != correct_answer and sample['syndrome'] not in candidate_syndromes:
                        candidate_syndromes.append(sample['syndrome'])
                
                # 隨機打亂候選證候順序（避免位置偏見）
                random.shuffle(candidate_syndromes)
                
                option_text = "、".join(candidate_syndromes) + "等"
            else:
                # 使用所有證候詞彙表
                if self.syndrome_vocab:
                    option_text = "、".join(self.syndrome_vocab) + "等"
        
        prompt = f"""
患者临床信息：
{clinical_text}

请从以下选项中选择：
{option_text}
"""
        
        format_suffix = "_five" if use_five_syndromes else "_all"
        
        return {
            "user_id": sample.get('user_id', ''),
            "prompt": prompt,
            "expected_answer": sample.get('norm_syndrome', ''),
            "format_type": f"reading_comprehension{format_suffix}",
            "expected_doc_id": self.get_syndrome_id(sample.get('norm_syndrome', ''))
        }
    
    def generate_prompts(self, data_path: str, output_path: str, format_type: str = "all", limit: int = None):
        """產生prompts並儲存到檔案"""
        
        # 載入資料
        data = self.load_tcm_data(data_path)
        if limit:
            data = data[:limit]
        
        results = []
        
        print(f"處理 {len(data)} 個樣本...")
        
        for i, sample in enumerate(data):
            if i % 100 == 0:
                print(f"已處理 {i}/{len(data)} 個樣本")
            
            sample_results = {}
            
            if format_type in ["all", "direct"]:
                sample_results["direct"] = self.format_direct_diagnosis(sample)
            
            if format_type in ["all", "multiple_choice"]:
                sample_results["multiple_choice"] = self.format_multiple_choice(sample, num_options=5) # 5 options for paper MRC setting
            
            if format_type in ["all", "multiple_choice_upperbound"]:
                sample_results["multiple_choice_upperbound"] = self.format_multiple_choice_upperbound(sample, num_options=5)
            
            if format_type in ["all", "reading_comprehension", "reading_comprehension_all"]:
                sample_results["reading_comprehension_all"] = self.format_reading_comprehension(sample, use_five_syndromes=False)
            
            if format_type in ["all", "reading_comprehension_five"]:
                sample_results["reading_comprehension_five"] = self.format_reading_comprehension(sample, use_five_syndromes=True)
            
            if format_type in ["all", "reading_comprehension_five_upperbound"]:
                sample_results["reading_comprehension_five_upperbound"] = self.format_reading_comprehension_with_knowledge(sample)
            
            if format_type == "all":
                results.append(sample_results)
            else:
                # 向後兼容性處理
                if format_type == "reading_comprehension":
                    format_type = "reading_comprehension_all"
                results.append(sample_results[format_type])
        
        # 儲存結果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"已產生 {len(results)} 個prompt樣本，儲存到 {output_path}")
        
        # 顯示範例
        if results:
            print("\n=== 範例 Prompt ===")
            if format_type == "all":
                for fmt in results[0].keys():
                    print(f"\n--- {fmt} 格式 ---")
                    print(results[0][fmt]["prompt"])
            else:
                print(results[0]["prompt"])


def main():
    parser = argparse.ArgumentParser(description="TCM-SD Prompt Generator for LLM Evaluation")
    parser.add_argument("--data_path", type=str, required=True, help="TCM-SD資料檔案路徑")
    parser.add_argument("--output_path", type=str, required=True, help="輸出prompt檔案路徑")
    parser.add_argument("--format", type=str, choices=["all", "direct", "multiple_choice", "multiple_choice_upperbound", "reading_comprehension", "reading_comprehension_all", "reading_comprehension_five", "reading_comprehension_five_upperbound"], 
                       default="all", help="Prompt格式類型")
    parser.add_argument("--limit", type=int, help="限制處理的樣本數量（用於測試）")
    parser.add_argument("--vocab_path", type=str, default="Data/TCM_SD/syndrome_vocab.txt", 
                       help="證候詞彙表路徑")
    parser.add_argument("--seed", type=int, default=42, help="設定隨機種子，確保結果可重現")
    
    args = parser.parse_args()
    
    # 建立產生器
    generator = TCMPromptGenerator(seed=args.seed)
    if args.vocab_path != "Data/TCM_SD/syndrome_vocab.txt":
        generator.load_syndrome_vocab(args.vocab_path)
    
    # 產生prompts
    generator.generate_prompts(
        data_path=args.data_path,
        output_path=args.output_path,
        format_type=args.format,
        limit=args.limit
    )


if __name__ == "__main__":
    main() 