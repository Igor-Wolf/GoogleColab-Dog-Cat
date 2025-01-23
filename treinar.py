import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Caminho da pasta com suas imagens
base_dir = '/content/pets'  # Substitua pelo caminho correto da pasta pets

# Passo 1: Carregar as imagens usando ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,               # Normalizar os valores dos pixels
    rotation_range=40,            # Aumentar o conjunto de dados com rotações
    width_shift_range=0.2,        # Deslocamento horizontal
    height_shift_range=0.2,       # Deslocamento vertical
    shear_range=0.2,              # Cisalhamento
    zoom_range=0.2,               # Zoom
    horizontal_flip=True,         # Flip horizontal
    fill_mode='nearest'           # Preenchimento de espaços criados pelas transformações
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Apenas normalizar as imagens de teste

# Definir os diretórios das subpastas de cats e dogs
train_dir = os.path.join(base_dir, 'train')  # Caminho onde estará as imagens de treino
validation_dir = os.path.join(base_dir, 'validation')  # Caminho onde estarão as imagens de validação

# Preparar o gerador de imagens para treinamento
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # Redimensionar as imagens
    batch_size=32,           # Tamanho do lote
    class_mode='binary'      # Como temos apenas 2 classes (cachorros e gatos), usamos binary
)

# Preparar o gerador de imagens para validação
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),  # Redimensionar as imagens
    batch_size=32,
    class_mode='binary'      # Binary porque temos 2 classes: gatos e cachorros
)

# Passo 2: Definir o modelo

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Saída binária: cachorro ou gato
])

# Passo 3: Compilar o modelo
model.compile(
    loss='binary_crossentropy',  # Usando binário para duas classes
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

# Passo 4: Treinar o modelo
history = model.fit(
    train_generator,
    steps_per_epoch=100,  # Quantos lotes processar por época
    epochs=20,            # Número de épocas
    validation_data=validation_generator,
    validation_steps=50   # Quantos lotes processar para validação
)

# Passo 5: Avaliar o modelo (opcional)
loss, accuracy = model.evaluate(validation_generator)
print(f'Perda: {loss:.4f}, Acurácia: {accuracy:.4f}')

# Salvar o modelo
model.save('modelo_cats_dogs.h5')
