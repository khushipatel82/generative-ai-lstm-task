import numpy as np
import tensorflow as tf
import requests
import os

# 1. Setup & Data Loading [cite: 15, 17]
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
FILE_NAME = "shakespeare_data.txt"

if not os.path.exists(FILE_NAME):
    print("Fetching dataset...")
    r = requests.get(DATA_URL)
    with open(FILE_NAME, 'wb') as f:
        f.write(r.content)

# Preprocessing [cite: 18]
raw_text = open(FILE_NAME, 'r').read().lower() 
chars = sorted(list(set(raw_text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# 2. Preparing Sequences [cite: 19, 20]
SEQ_LENGTH = 40
STEP_SIZE = 3
inputs = []
targets = []

for i in range(0, len(raw_text) - SEQ_LENGTH, STEP_SIZE):
    inputs.append(raw_text[i : i + SEQ_LENGTH])
    targets.append(raw_text[i + SEQ_LENGTH])

# Vectorizing [cite: 20]
x = np.zeros((len(inputs), SEQ_LENGTH, len(chars)), dtype=np.bool_)
y = np.zeros((len(inputs), len(chars)), dtype=np.bool_)

for i, sentence in enumerate(inputs):
    for t, char in enumerate(sentence):
        x[i, t, char_to_idx[char]] = 1
    y[i, char_to_idx[targets[i]]] = 1

# 3. Model Architecture [cite: 21, 22, 23]
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, input_shape=(SEQ_LENGTH, len(chars))),
    tf.keras.layers.Dense(len(chars), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy') # [cite: 24]

# 4. Training [cite: 25, 28, 29]
print("Starting model training...")
model.fit(x, y, batch_size=128, epochs=10) 

# 5. Generation Function [cite: 30, 32, 34]
def generate_sample_text(seed, length=100):
    current_seq = seed
    generated = ""
    for _ in range(length):
        x_pred = np.zeros((1, SEQ_LENGTH, len(chars)))
        for t, char in enumerate(current_seq):
            if char in char_to_idx:
                x_pred[0, t, char_to_idx[char]] = 1
        
        preds = model.predict(x_pred, verbose=0)[0]
        next_idx = np.argmax(preds)
        next_char = idx_to_char[next_idx]
        
        generated += next_char
        current_seq = current_seq[1:] + next_char
    return generated

# Final Execution [cite: 33, 35, 44]
test_seed = raw_text[500:540]
print("\n--- Seed Input ---")
print(test_seed)
print("\n--- Generated Result ---")
print(generate_sample_text(test_seed))