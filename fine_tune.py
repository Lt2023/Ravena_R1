import os
import json
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
import random
import pickle
import jieba
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.layers import Layer
import re


# 自定义的 PositionalEncoding 层
class PositionalEncoding(Layer):
    def __init__(self, max_len, model_dim, trainable=True, **kwargs):
        super(PositionalEncoding, self).__init__(trainable=trainable, **kwargs)
        self.max_len = max_len
        self.model_dim = model_dim

    def build(self, input_shape):
        # 计算位置编码
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.model_dim, 2) * -(np.log(10000.0) / self.model_dim))
        pos_encoding = np.zeros((self.max_len, self.model_dim))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        # 将预计算的 pos_encoding 作为权重，不使用 Keras 初始化器
        self.pos_encoding = tf.Variable(initial_value=pos_encoding, trainable=False, dtype=tf.float32, name='pos_encoding')

    def call(self, inputs):
        return inputs + self.pos_encoding

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'max_len': self.max_len,
            'model_dim': self.model_dim
        })
        return config


class FineTuningTool:
    def __init__(self, model_file='model/.Ravena-LLM_Model.keras', tokenizer_file='model/tokenizer.json', faq_file='model/.faq_data.pkl', max_seq_length=20, vocab_size=10000):
        # 将模型保存路径设置为 models 文件夹，并使用 .keras 后缀
        self.model_file = model_file
        self.tokenizer_file = tokenizer_file
        self.faq_file = faq_file
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size

        # 加载模型和Tokenizer
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()

        # 初始化FAQ数据
        self.faq_data = self._load_faq_data()

    def _load_model(self):
        """加载模型，如果文件不存在则返回None"""
        if os.path.exists(self.model_file):
            return load_model(self.model_file, custom_objects={'PositionalEncoding': PositionalEncoding})
        else:
            print(f"模型文件 {self.model_file} 不存在，无法进行微调。请先训练模型。")
            return None

    def _load_tokenizer(self):
        """加载Tokenizer，如果文件不存在则返回None"""
        if os.path.exists(self.tokenizer_file):
            with open(self.tokenizer_file, 'r', encoding='utf-8') as f:
                tokenizer_json = json.load(f)
                return tokenizer_from_json(tokenizer_json)
        else:
            print(f"Tokenizer文件 {self.tokenizer_file} 不存在，无法进行微调。请先训练模型。")
            return None

    def _load_faq_data(self):
        """加载FAQ数据，如果文件不存在则初始化为空字典"""
        if os.path.exists(self.faq_file):
            with open(self.faq_file, 'rb') as f:
                return pickle.load(f)
        else:
            return {}

    def _save_faq_data(self):
        """保存FAQ数据"""
        os.makedirs(os.path.dirname(self.faq_file), exist_ok=True)
        with open(self.faq_file, 'wb') as f:
            pickle.dump(self.faq_data, f)

    def augment_data(self, data):
        """数据增强：打乱词序和随机插入/删除词语"""
        augmented_data = []
        for item in data:
            words = item.split()
            random.shuffle(words)  # 打乱顺序

            if random.random() > 0.5:
                words.append(random.choice(words))  # 随机插入词语
            if len(words) > 2 and random.random() > 0.5:
                words.remove(random.choice(words))  # 随机移除词语

            augmented_data.append(' '.join(words))
        return augmented_data

    def prepare_data(self, questions, answers):
        """处理问题和答案，进行分词并生成序列"""
        # 数据增强
        questions = self.augment_data(questions)
        answers = self.augment_data(answers)

        # 进行分词
        questions = [" ".join(jieba.cut(q)) for q in questions]
        answers = [" ".join(jieba.cut(a)) for a in answers]

        # 将文本转换为序列
        question_sequences = pad_sequences(self.tokenizer.texts_to_sequences(questions), maxlen=self.max_seq_length)
        answer_sequences = [self.tokenizer.texts_to_sequences([a])[0] for a in answers]
        answer_sequences = np.array([seq[0] for seq in answer_sequences])

        return question_sequences, answer_sequences

    def scheduler(self, epoch, lr):
        """学习率调度"""
        if epoch < 5:
            return float(lr * (epoch + 1) / 5)  # Warm-Up 阶段增加学习率
        else:
            return float(lr * np.exp(-0.05))  # 更温和地衰减学习率

    def fine_tune(self, new_data_file, epochs=10000, batch_size=64):
        """微调模型"""
        # 加载新数据
        try:
            with open(new_data_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
                questions = [item['question'] for item in data['data']]
                answers = [item['answer'] for item in data['data']]
        except FileNotFoundError:
            print(f"数据文件 {new_data_file} 不存在。")
            return

        # 准备数据
        question_sequences, answer_sequences = self.prepare_data(questions, answers)

        # 设置学习率调度器、早停和保存回调
        lr_scheduler = LearningRateScheduler(self.scheduler)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)  # 增加耐心
        model_checkpoint = ModelCheckpoint(self.model_file, monitor='val_loss', save_best_only=True, verbose=1)

        # 开始微调
        print("开始微调模型...")
        self.model.fit(question_sequences, answer_sequences,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_split=0.1,
                       callbacks=[lr_scheduler, early_stopping, model_checkpoint],
                       verbose=1)

        # 微调后保存模型和数据
        self.model.save(self.model_file)
        self._save_faq_data()
        print(f"微调完成并保存模型到 {self.model_file}！")

    def generate_answer(self, input_text, max_length=20, temperature=0.7, top_p=0.9):
        """生成回答"""
        if input_text in self.faq_data:
            answers = self.faq_data[input_text]
            selected_answer = random.choice(answers)
            return self.format_text(selected_answer)

        seq = pad_sequences(self.tokenizer.texts_to_sequences([input_text]), maxlen=self.max_seq_length)
        generated_seq = list(seq[0])
        generated_text = ''

        for _ in range(max_length):
            if len(generated_seq) > self.max_seq_length:
                generated_seq = generated_seq[-self.max_seq_length:]

            pred = self.model.predict(np.array([generated_seq]), verbose=0)
            pred = np.log(pred) / temperature
            pred = np.exp(pred) / np.sum(np.exp(pred))

            sorted_indices = np.argsort(pred[0])[::-1]
            cumulative_probs = np.cumsum(pred[0][sorted_indices])
            top_p_indices = sorted_indices[cumulative_probs <= top_p]

            next_word_index = np.random.choice(top_p_indices)
            generated_seq.append(next_word_index)

            if next_word_index == 0:
                break

            word = self.tokenizer.index_word.get(next_word_index, '')
            generated_text += word + ' '

        generated_text = self.clean_text(generated_text)

        self.faq_data[input_text] = [generated_text]  # 更新FAQ数据
        self._save_faq_data()

        return self.format_text(generated_text)

    def clean_text(self, text):
        """清理生成的文本"""
        cleaned_text = ' '.join(text.split())
        cleaned_text = cleaned_text.replace("  ", " ").strip()
        return cleaned_text

    def format_text(self, text):
        """格式化文本"""
        text = re.sub(r'\s([?.!,":;(){}])', r'\1', text)  # 处理标点
        text = re.sub(r'\n', ' ', text)  # 替换换行符
        text = text.strip()
        return text


if __name__ == '__main__':
    # 创建微调工具
    fine_tuning_tool = FineTuningTool()

    # 指定新的训练数据文件路径
    new_data_file = 'new_data.json'
    
    # 微调模型 10000 次
    fine_tuning_tool.fine_tune(new_data_file, epochs=10000)
