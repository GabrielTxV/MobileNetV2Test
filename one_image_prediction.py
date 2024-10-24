import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt

# Carregar o modelo pré-treinado MobileNetV2
model = MobileNetV2(weights='imagenet')

# Carregar uma imagem real para teste
current_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(current_dir, 'dog.jpg')  # Substitua pelo nome da imagem escolhida

# Carregar e redimensionar a imagem para 224x224 pixels, que é o formato esperado pelo modelo
img = image.load_img(img_path, target_size=(224, 224))

# Exibir a imagem carregada
plt.imshow(img)
plt.title("Imagem Carregada")
plt.axis('off')
plt.show()

# Converter a imagem para um array de NumPy e normalizar os valores dos pixels (0 a 1)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Adicionar a dimensão do lote (batch)
img_array = preprocess_input(img_array)  # Pré-processar a imagem para o MobileNetV2

# Fazer a previsão
predicao = model.predict(img_array)

# Decodificar a previsão para obter as top 5 classes previstas
decoded_preds = decode_predictions(predicao, top=5)[0]

for i, (imagenet_id, label, score) in enumerate(decoded_preds):
    print(f"{i + 1}: {label} ({score:.2f})")

# Obter a classe prevista e a confiança da previsão mais provável
classe_prevista = decoded_preds[0][1]
confidencia = decoded_preds[0][2]

print(f"Classe prevista: {classe_prevista} com confiança de {confidencia:.2f}")
