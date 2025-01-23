import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Carregar o modelo salvo
model = tf.keras.models.load_model('/content/modelo_cats_dogs.h5')  # Substitua pelo caminho do seu modelo

# Caminho da imagem a ser testada (insira o caminho para a imagem que você deseja testar)
image_path = '/content/4.jpg'  # Substitua pelo caminho da imagem

# Passo 1: Carregar e pré-processar a imagem
img = image.load_img(image_path, target_size=(150, 150))  # Redimensiona a imagem para 150x150 pixels
img_array = image.img_to_array(img)  # Converte a imagem para um array NumPy
img_array = np.expand_dims(img_array, axis=0)  # Adiciona uma dimensão extra para corresponder ao batch size
img_array = img_array / 255.0  # Normaliza a imagem (mesmo que durante o treinamento)

# Passo 2: Fazer a previsão
prediction = model.predict(img_array)

# Passo 3: Exibir o resultado
if prediction[0] > 0.5:
    print("A imagem é de um cachorro!")
else:
    print("A imagem é de um gato!")