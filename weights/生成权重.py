import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import plot_model

def build_simple_model(input_dim=100):
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def visualize_weights(model, output_dir='weights'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_img = os.path.join(output_dir, 'model_weights.png')
    output_txt = os.path.join(output_dir, 'model_weights.txt')

    with open(output_txt, 'w') as f:
        f.write("模型权重信息\n")
        f.write("-----------------------------\n")
        
        fig, axes = plt.subplots(len(model.layers), 1, figsize=(10, len(model.layers)*2))
        if len(model.layers) == 1: 
            axes = [axes]

        for i, layer in enumerate(model.layers):
            if isinstance(layer, tf.keras.layers.Dense):
                weights = layer.get_weights()[0]  # 权重矩阵
                biases = layer.get_weights()[1]  # 偏置

                f.write(f"\n第 {i + 1} 层: {layer.name}\n")
                f.write(f"权重形状: {weights.shape}\n")
                f.write(f"偏置形状: {biases.shape}\n")
                f.write(f"偏置值: {biases}\n")

                ax = axes[i]
                sns.heatmap(weights, ax=ax, cmap='coolwarm', annot=True, fmt='.2f')
                ax.set_title(f'第{i+1}层的权重: {layer.name}')

        plt.tight_layout()
        plt.savefig(output_img)
        print(f"模型权重可视化并保存到 {output_img}")

if __name__ == "__main__":
    model = build_simple_model(input_dim=100)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    X_train = np.random.rand(100, 100)  
    y_train = np.random.randint(2, size=100)  
    model.fit(X_train, y_train, epochs=5, batch_size=32)
    visualize_weights(model, output_dir='weights')
