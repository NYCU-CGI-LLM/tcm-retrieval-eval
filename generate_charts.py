#!/usr/bin/env python3
"""
從現有的evaluation_results.json重新生成可視化圖表
支持生成分布直方圖和累積分布圖
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List


def load_evaluation_results(json_file_path: str) -> Dict[str, Any]:
    """讀取評估結果JSON文件"""
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"結果文件不存在: {json_file_path}")
    
    print(f"📂 正在讀取結果文件: {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results


def extract_rankings(results: Dict[str, Any]) -> List[int]:
    """從評估結果中提取排名信息"""
    rankings = []
    total_queries = len(results['individual_results'])
    
    print(f"🔍 從 {total_queries} 個查詢結果中提取排名信息...")
    
    for result in results['individual_results']:
        expected_doc_rank = result.get('expected_doc_rank', 1027)
        rankings.append(expected_doc_rank)
    
    return rankings


def generate_histogram(rankings: List[int], total_queries: int, output_file: str):
    """生成分布直方圖"""
    print(f"📊 正在生成分布直方圖...")
    
    plt.figure(figsize=(10, 6))
    bins = range(1, 1029)  # 從1到1028，包含1027個排名位置
    plt.hist(rankings, bins=bins, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
    plt.xlabel('Rank of Expected Doc ID')
    plt.ylabel('Count')
    plt.title(f'Distribution of Expected Doc ID Rankings\n(Total Queries: {total_queries})')
    plt.grid(True, alpha=0.3)
    plt.xlim(1, 1027)
    
    # 添加統計信息
    if rankings:
        mean_rank = np.mean(rankings)
        median_rank = np.median(rankings)
        plt.axvline(mean_rank, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_rank:.1f}')
        plt.axvline(median_rank, color='orange', linestyle='--', alpha=0.7, label=f'Median: {median_rank:.1f}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Histogram chart saved: {output_file}")


def generate_cumulative_distribution(rankings: List[int], total_queries: int, output_file: str):
    """生成累積分布圖"""
    print(f"📊 正在生成累積分布圖...")
    
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
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Cumulative distribution chart saved: {output_file}")


def generate_simple_recall_curve(rankings: List[int], total_queries: int, output_file: str):
    """生成精簡的recall曲線圖（只顯示recall@5,10,20,50,100五個關鍵點）"""
    print(f"📊 正在生成精簡recall曲線圖...")
    
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
            marker='o', markersize=8, linewidth=3, 
            color='darkblue', alpha=0.8, markerfacecolor='darkblue', 
            markeredgecolor='darkblue', markeredgewidth=2)
    
    plt.xlabel('K (Rank Threshold)', fontsize=12)
    plt.ylabel('Recall@K', fontsize=12)
    plt.title(f'Simplified Recall Curve\n(Total Queries: {total_queries})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1027)
    plt.ylim(0, max(1.0, max(recall_proportions) + 0.1))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Simplified recall curve saved: {output_file}")
    
    # 打印recall統計信息
    print(f"📈 Recall Statistics:")
    for k, recall in zip(recall_k_values, recall_proportions):
        count = int(recall * total_queries)
        print(f"   Recall@{k}: {count}/{total_queries} = {recall:.3f}")


def generate_comparison_cumulative_distribution(rankings1: List[int], rankings2: List[int], 
                                               exp1_name: str, exp2_name: str, output_file: str):
    """生成兩組實驗的累積分布比較圖"""
    print(f"📊 正在生成比較累積分布圖...")
    
    plt.figure(figsize=(12, 8))
    
    # 處理第一組實驗數據
    sorted_rankings1 = sorted(rankings1)
    cumulative_counts1 = np.arange(1, len(sorted_rankings1) + 1) / len(sorted_rankings1)
    
    # 處理第二組實驗數據
    sorted_rankings2 = sorted(rankings2)
    cumulative_counts2 = np.arange(1, len(sorted_rankings2) + 1) / len(sorted_rankings2)
    
    # 繪制兩條累積分布曲線
    plt.plot(sorted_rankings1, cumulative_counts1, marker='o', markersize=3, 
             linewidth=2, color='darkblue', alpha=0.8, label=f'{exp1_name} (n={len(rankings1)})')
    plt.plot(sorted_rankings2, cumulative_counts2, marker='s', markersize=3, 
             linewidth=2, color='darkred', alpha=0.8, label=f'{exp2_name} (n={len(rankings2)})')
    
    plt.xlabel('Rank of Expected Doc ID', fontsize=12)
    plt.ylabel('Cumulative Proportion', fontsize=12)
    plt.title(f'Cumulative Distribution Comparison\n{exp1_name} vs {exp2_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(1, 1027)
    plt.ylim(0, 1)
    
    # 添加重要的recall指標線
    recall_points = [5, 10, 20, 50, 100]
    colors = ['gray', 'lightgray', 'silver', 'gainsboro', 'whitesmoke']
    
    for i, recall_k in enumerate(recall_points):
        color = colors[i % len(colors)]
        plt.axvline(recall_k, color=color, linestyle=':', alpha=0.5, linewidth=1)
    
    # 在圖例中添加recall統計信息
    legend_text = []
    legend_text.append(f'{exp1_name} (n={len(rankings1)})')
    legend_text.append(f'{exp2_name} (n={len(rankings2)})')
    
    # 計算並顯示關鍵recall指標
    print(f"\n📈 Comparison Statistics:")
    print(f"{'Metric':<12} {'Exp1':<15} {'Exp2':<15} {'Difference':<12}")
    print("=" * 55)
    
    for recall_k in [5, 10, 20, 50, 100]:
        recall1 = sum(1 for rank in rankings1 if rank <= recall_k) / len(rankings1)
        recall2 = sum(1 for rank in rankings2 if rank <= recall_k) / len(rankings2)
        diff = recall2 - recall1
        print(f"Recall@{recall_k:<3} {recall1:<15.4f} {recall2:<15.4f} {diff:<+12.4f}")
    
    plt.legend(fontsize=10, loc='lower right')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison cumulative distribution chart saved: {output_file}")


def print_statistics(rankings: List[int], total_queries: int):
    """打印統計信息"""
    print(f"\n📈 Ranking Statistics:")
    print(f"   Total queries analyzed: {total_queries}")
    print(f"   Mean rank: {np.mean(rankings):.1f}")
    print(f"   Median rank: {np.median(rankings):.1f}")
    print(f"   Best rank: {min(rankings)}")
    print(f"   Worst rank: {max(rankings)}")
    
    # 累積統計信息
    print(f"\n📊 Cumulative Statistics:")
    for recall_k in [5, 10, 20, 50, 100]:
        count_within_k = sum(1 for rank in rankings if rank <= recall_k)
        proportion = count_within_k / total_queries
        print(f"   Recall@{recall_k}: {count_within_k}/{total_queries} = {proportion:.3f}")


def generate_charts_from_json(json_file_path: str, output_folder: str = None):
    """從JSON結果文件生成可視化圖表"""
    try:
        # 讀取評估結果
        results = load_evaluation_results(json_file_path)
        
        # 如果沒有指定輸出文件夾，使用JSON文件所在的目錄
        if output_folder is None:
            output_folder = os.path.dirname(json_file_path)
        
        # 確保輸出文件夾存在
        os.makedirs(output_folder, exist_ok=True)
        
        # 提取排名信息
        rankings = extract_rankings(results)
        total_queries = len(results['individual_results'])
        
        # 設定圖表輸出路徑
        hist_output_file = os.path.join(output_folder, "ranking_histogram.png")
        cumulative_output_file = os.path.join(output_folder, "ranking_cumulative.png")
        recall_curve_output_file = os.path.join(output_folder, "recall_curve_simplified.png")
        
        # 生成三張圖表
        generate_histogram(rankings, total_queries, hist_output_file)
        generate_cumulative_distribution(rankings, total_queries, cumulative_output_file)
        generate_simple_recall_curve(rankings, total_queries, recall_curve_output_file)
        
        # 打印統計信息
        print_statistics(rankings, total_queries)
        
        print(f"\n✅ 圖表重新生成完成! 輸出文件夾: {output_folder}")
        
    except Exception as e:
        print(f"❌ 生成圖表失敗: {e}")
        import traceback
        traceback.print_exc()


def compare_experiments(exp1_path: str, exp2_path: str, output_folder: str = None):
    """比較兩組實驗的累積分布圖"""
    try:
        # 構建兩個實驗的evaluation_results.json路徑
        json_file1 = os.path.join(exp1_path, "evaluation_results.json")
        json_file2 = os.path.join(exp2_path, "evaluation_results.json")
        
        print(f"🔍 比較實驗:")
        print(f"   實驗1: {json_file1}")
        print(f"   實驗2: {json_file2}")
        
        # 讀取兩個實驗的結果
        results1 = load_evaluation_results(json_file1)
        results2 = load_evaluation_results(json_file2)
        
        # 提取排名信息
        rankings1 = extract_rankings(results1)
        rankings2 = extract_rankings(results2)
        
        # 獲取實驗名稱（從路徑中提取）
        exp1_name = os.path.basename(exp1_path)
        exp2_name = os.path.basename(exp2_path)
        
        # 設定輸出文件夾
        if output_folder is None:
            # 如果沒有指定輸出文件夾，在兩個實驗的公共父目錄創建comparison文件夾
            common_parent = os.path.commonpath([exp1_path, exp2_path])
            output_folder = os.path.join(common_parent, f"comparison_{exp1_name}_vs_{exp2_name}")
        
        # 確保輸出文件夾存在
        os.makedirs(output_folder, exist_ok=True)
        
        # 生成比較圖表
        comparison_output_file = os.path.join(output_folder, "ranking_cumulative_comparison.png")
        generate_comparison_cumulative_distribution(rankings1, rankings2, exp1_name, exp2_name, comparison_output_file)
        
        print(f"\n✅ 實驗比較完成! 輸出文件夾: {output_folder}")
        
    except Exception as e:
        print(f"❌ 比較實驗失敗: {e}")
        import traceback
        traceback.print_exc()


def batch_generate_charts(base_folder: str, pattern: str = "evaluation_results.json"):
    """批量為多個實驗結果生成圖表"""
    print(f"🔍 搜索文件夾: {base_folder}")
    print(f"📋 搜索模式: {pattern}")
    
    generated_count = 0
    
    # 遍歷所有子文件夾
    for root, dirs, files in os.walk(base_folder):
        if pattern in files:
            json_file_path = os.path.join(root, pattern)
            print(f"\n{'='*50}")
            print(f"🎯 處理: {json_file_path}")
            print(f"{'='*50}")
            
            generate_charts_from_json(json_file_path, root)
            generated_count += 1
    
    print(f"\n🎉 批量處理完成! 總共處理了 {generated_count} 個實驗結果")


def main():
    parser = argparse.ArgumentParser(description="從現有的evaluation_results.json重新生成可視化圖表")
    parser.add_argument("json_file", nargs='?',
                       help="evaluation_results.json 文件路徑")
    parser.add_argument("--output-folder", "-o",
                       help="輸出文件夾路徑 (可選，默認為JSON文件所在目錄)")
    parser.add_argument("--batch", "-b",
                       help="批量模式：指定包含多個實驗結果的基礎文件夾路徑")
    parser.add_argument("--compare", "-c", nargs=2, metavar=("EXP1_PATH", "EXP2_PATH"),
                       help="比較模式：指定兩個實驗文件夾路徑來生成比較圖表")
    
    args = parser.parse_args()
    
    if args.compare:
        # 比較模式
        exp1_path, exp2_path = args.compare
        compare_experiments(exp1_path, exp2_path, args.output_folder)
    elif args.batch:
        # 批量處理模式
        batch_generate_charts(args.batch)
    elif args.json_file:
        # 單個文件處理模式
        generate_charts_from_json(args.json_file, args.output_folder)
    else:
        # 沒有提供參數，顯示幫助
        parser.print_help()
        print("\n使用範例:")
        print("  # 單個文件:")
        print("  python generate_charts.py outputs/run_syndrome_db_5486/evaluation_results.json")
        print("\n  # 指定輸出文件夾:")
        print("  python generate_charts.py outputs/run_syndrome_db_5486/evaluation_results.json -o my_charts/")
        print("\n  # 批量處理:")
        print("  python generate_charts.py --batch outputs/")
        print("\n  # 比較兩組實驗:")
        print("  python generate_charts.py --compare outputs/run_syndrome_db_5466 outputs/run_syndrome_db_5466_with_pseudo")


if __name__ == "__main__":
    main()
