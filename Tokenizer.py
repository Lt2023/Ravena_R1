import json
from termcolor import colored

def display_tokenizer(tokenizer_file='model/tokenizer.json', output_file='Ravena_tokenizer_output.txt'):
    

    try:
        with open(tokenizer_file, 'r', encoding='utf-8') as file:
            tokenizer_data = json.load(file)
        print(colored(f"成功加载Tokenizer文件: {tokenizer_file}", "green"))
    except FileNotFoundError:
        print(colored(f"未找到Tokenizer文件: {tokenizer_file}", "red"))
        return
    

    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    tokenizer = tokenizer_from_json(tokenizer_data)
    

    word_index = tokenizer.word_index
    

    try:
        with open(output_file, 'w', encoding='utf-8') as output:
            output.write("Tokenizer 词汇表：\n")
            for word, index in word_index.items():
                output.write(f"{word}: {index}\n")
        
        print(colored(f"词汇表已成功保存到: {output_file}", "green"))
    
    except Exception as e:
        print(colored(f"保存词汇表时发生错误: {str(e)}", "red"))
        return
    
    print(colored("Tokenizer 词汇表（部分）:", "yellow"))
    for i, (word, index) in enumerate(word_index.items()):
        print(f"{word}: {index}")
        if i >= 5000:  
            break

if __name__ == "__main__":
    display_tokenizer()
