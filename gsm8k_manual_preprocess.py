
#!/usr/bin/env python3
"""
完全手动处理 GSM8K 数据，避免 HuggingFace 和权限问题
"""
import os
import re
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq


def extract_solution(solution_str):
    """从回答中提取最终答案（数字）"""
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    if solution is None:
        return None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


def main():
    print("=" * 80)
    print("GSM8K 手动数据预处理")
    print("=" * 80)
    print()
    
    # 1. 查找本地数据
    possible_paths = [
        "/Users/diaby24/文档/gsm8k",
        "/Users/diaby24/Desktop/verl/data/gsm8k",
        "~/data/gsm8k_raw",
    ]
    
    local_dataset_path = None
    for p in possible_paths:
        p = os.path.expanduser(p)
        if os.path.exists(p):
            print(f"找到本地数据目录: {p}")
            local_dataset_path = p
            break
    
    if local_dataset_path is None:
        print("错误：未找到本地数据！")
        return
    
    # 2. 加载本地数据（使用 datasets.load_from_disk）
    print(f"从本地加载数据集: {local_dataset_path}")
    import datasets
    dataset = datasets.load_from_disk(local_dataset_path)
    
    train_table = dataset["train"]
    test_table = dataset["test"]
    
    print()
    print(f"训练集样本数: {len(train_table)}")
    print(f"测试集样本数: {len(test_table)}")
    print()
    
    # 3. 处理数据（转换为 verl 格式）
    instruction_following = 'Let\'s think step by step and output the final answer after "####".'
    
    def process_split(dataset, split_name):
        """处理一个数据集分割"""
        data_list = []
        
        for idx in range(len(dataset)):
            row = dataset[idx]
            question_raw = row["question"]
            answer_raw = row["answer"]
            
            question = question_raw + " " + instruction_following
            solution = extract_solution(answer_raw)
            
            data_item = {
                "data_source": "openai/gsm8k",
                "prompt": json.dumps([
                    {"role": "user", "content": question}
                ]),  # 序列化为字符串，方便存 parquet
                "ability": "math",
                "reward_model": json.dumps({
                    "style": "rule",
                    "ground_truth": solution
                }),
                "extra_info": json.dumps({
                    "split": split_name,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                })
            }
            
            data_list.append(data_item)
        
        return data_list
    
    print("处理训练集...")
    train_data = process_split(train_table, "train")
    
    print("处理测试集...")
    test_data = process_split(test_table, "test")
    
    # 4. 保存为 parquet（使用项目目录，避免权限问题）
    save_dir = "/Users/diaby24/Desktop/verl/data/gsm8k_processed"
    os.makedirs(save_dir, exist_ok=True)
    
    # 转换为 PyArrow 表
    def save_parquet(data_list, filepath):
        if not data_list:
            return
        table = pa.Table.from_pylist(data_list)
        pq.write_table(table, filepath)
        print(f"保存到: {filepath}")
    
    save_parquet(train_data, os.path.join(save_dir, "train.parquet"))
    save_parquet(test_data, os.path.join(save_dir, "test.parquet"))
    
    # 同时也保存到 ~/data/gsm8k（如果有权限）
    try:
        home_save_dir = os.path.expanduser("~/data/gsm8k")
        os.makedirs(home_save_dir, exist_ok=True)
        save_parquet(train_data, os.path.join(home_save_dir, "train.parquet"))
        save_parquet(test_data, os.path.join(home_save_dir, "test.parquet"))
    except Exception as e:
        print(f"保存到用户目录失败: {e}")
        print(f"没关系，数据已保存到项目目录！")
    
    print()
    print("=" * 80)
    print("✅ 预处理完成！")
    print(f"主要保存位置: {save_dir}")
    print(f"  - train.parquet: {len(train_data)} 样本")
    print(f"  - test.parquet: {len(test_data)} 样本")
    print("=" * 80)
    print()
    print("数据样本（前 1 条）:")
    sample = train_data[0]
    print(f"  question: {json.loads(sample['prompt'])[0]['content'][:80]}...")
    print(f"  ground_truth: {json.loads(sample['reward_model'])['ground_truth']}")


if __name__ == "__main__":
    main()
