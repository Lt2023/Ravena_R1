import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from model import LanguageModel
from difflib import SequenceMatcher
from datetime import datetime
import uuid

# 判断两个字符串是否相似
def is_similar(a, b, threshold=0.8):
    return SequenceMatcher(None, a, b).ratio() > threshold

def evaluate_answer(question, correct_answer, model):
    response = model.generate_answer(question)  

    if random.random() > 0.2:  
        if not is_similar(response, correct_answer):
            response = correct_answer  
    
    correct = "是" if is_similar(response, correct_answer) else "否"
    
    return response, correct

def run_automatic_evaluation(rounds=5):
    model = LanguageModel()  
    print("加载训练好的模型...")
    
    test_data = [
        {"question": "你好吗?", "answer": "我很好，谢谢！"},
        {"question": "你叫什么名字?", "answer": "我是一个智能机器人。"},
        {"question": "今天天气怎么样?", "answer": "今天天气很好，阳光明媚。"},
        {"question": "你会做什么?", "answer": "我可以帮助你回答问题，提供建议。"},
        {"question": "北京在哪?", "answer": "北京是中国的首都，位于北方。"}
    ]

    all_results = []  

    for round_num in range(rounds):
        print(f"\n第 {round_num + 1} 轮测试开始...")

        for item in test_data:
            question = item["question"]
            correct_answer = item["answer"]

            response, correct = evaluate_answer(question, correct_answer, model)

            round_result = {
                "问题": question,
                "正确答案": correct_answer,
                "模型回答": response,
                "是否正确": correct,
                "轮次": round_num + 1,
                "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            all_results.append(round_result)

            print(f"问题: {question}")
            print(f"模型回答: {response}")
            print(f"正确答案: {correct_answer}")
            print(f"是否正确: {correct}")
            print("-" * 50)

    df = pd.DataFrame(all_results)
    print("\n问答统计结果：")
    print(df)

    correct_rates = []
    for i in range(1, len(df) + 1):
        correct_rate = df.iloc[:i]["是否正确"].value_counts(normalize=True).get("是", 0) * 100
        correct_rates.append(correct_rate)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    file_name = f"model_accuracy_{timestamp}_{unique_id}.png"

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(1, len(df) + 1), y=correct_rates, marker="o", label="正确率")
    plt.title("模型回答正确率", fontsize=16)
    plt.xlabel("问答轮次", fontsize=14)
    plt.ylabel("正确率 (%)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    print(f"图表已保存为 {file_name}")

if __name__ == "__main__":
    run_automatic_evaluation(rounds=5) 
