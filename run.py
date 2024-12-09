from model import LanguageModel

def main():
    """
    主函数，用于与用户进行交互式对话。
    """
    model = LanguageModel()  # 初始化模型

    print("加载训练好的模型...")

    while True:
        question = input("你好！请问你有什么问题？输入 'exit' 退出程序。\n")

        if question.lower() == 'exit':  
            print("退出程序...")
            break

        response = model.generate_answer(question)  # 使用模型生成回答
        print(f"模型的回答：{response}\n")

if __name__ == "__main__":
    main()