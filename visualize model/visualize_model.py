import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, LayerNormalization, Dropout, Dense
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei'] 
matplotlib.rcParams['axes.unicode_minus'] = False  

class LanguageModel:
    def __init__(self, vocab_size=10000, max_seq_length=20):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.model = self.build_model()  

    def build_model(self):
        input_layer = Input(shape=(self.max_seq_length,))

        embedding_layer = Embedding(self.vocab_size, 128)(input_layer)

        attention_layer = MultiHeadAttention(num_heads=8, key_dim=128)(embedding_layer, embedding_layer)
        attention_layer = LayerNormalization()(attention_layer)  #
        attention_layer = Dropout(0.1)(attention_layer)  

        output_layer = Dense(self.vocab_size, activation='softmax')(attention_layer)

        model = Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def visualize_model(self):
        image_path = 'visualize model/model_structure.png'
        plot_model(self.model, to_file=image_path, show_shapes=True, show_layer_names=True)

        img = plt.imread(image_path)
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(img)
        ax.axis('off')  

        ax.text(0.5, 0.95, "Transformer 模型结构", fontsize=24, ha='center', va='center', transform=ax.transAxes, color='black')

        plt.savefig(image_path, bbox_inches='tight', pad_inches=0.1)

        print(f"模型结构图已生成并保存在 'visualize model' 目录下的 {image_path} 文件中。")

if __name__ == "__main__":
    model = LanguageModel(vocab_size=10000, max_seq_length=20)
    model.visualize_model()
