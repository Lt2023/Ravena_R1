from model import LanguageModel

def main():
    model = LanguageModel()  
    print("开始训练模型...")
    model.train(epochs=1000)  
    print("训练完成！")

if __name__ == "__main__":
    main()
