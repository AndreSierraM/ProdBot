import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import LabelEncoder

#

# Tokenización
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

# Normalización
def normalize_word(word):
    word = word.lower()
    return word

# Reducción a raíz
stemmer = LancasterStemmer()

# Eliminación de palabras irrelevantes
stop_words = set(nltk.corpus.stopwords.words('english'))

corpus = ['Hi there',
          'How are you',
          'Is anyone there?',
          'Hello',
          'Good day',
          'What is your name?',
          'Bye',
          'See you later',
          'Goodbye']

# Crear etiquetas
labels = []
for sentence in corpus:
    words = tokenizer.tokenize(sentence)
    words = [normalize_word(word) for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    labels.append(words[0])

# Crear patrones
patterns = []
for sentence in corpus:
    words = tokenizer.tokenize(sentence)
    words = [normalize_word(word) for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    pattern = []
    for i in range(len(labels)):
        if labels[i] == words[0]:
            pattern.append(1)
        else:
            pattern.append(0)
    patterns.append(pattern)
    
patterns = np.array(patterns)
labels = np.array(labels)

# Crear modelo
model = keras.Sequential([
    layers.Dense(8, input_shape=(len(patterns[0]),), activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(len(set(labels)), activation='softmax')
])

# Compilar modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar modelo
model.fit(patterns, labels, epochs=1000)


label_encoder = LabelEncoder()
label_encoder.fit(labels)
encoded_labels = label_encoder.transform(labels)

# Entrenar el modelo con las etiquetas codificadas
model.fit(patterns, encoded_labels, epochs=1000)

# Realizar predicciones
test_sentence = 'Hi'
test_words = tokenizer.tokenize(test_sentence)
test_words = [normalize_word(word) for word in test_words if word not in stop_words]
test_words = [stemmer.stem(word) for word in test_words]
test_pattern = []
for i in range(len(labels)):
    if labels[i] == test_words[0]:
        test_pattern.append(1)
    else:
        test_pattern.append(0)

test_pattern = np.array(test_pattern)
# Realizar predicciones
predictions = model.predict(np.array([test_pattern]))
predicted_label_index = np.argmax(predictions)
predicted_label = labels[predicted_label_index]
print(predicted_label)

