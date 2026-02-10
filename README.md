# LSTM-Based Text Generation Engine

### Project Overview
This project implements a character-level text generator using Long Short-Term Memory (LSTM) networks. [cite_start]The goal was to train a model that captures the linguistic patterns of Shakespearean text and generates new content based on a seed input[cite: 4, 5].

### [cite_start]Implementation Details [cite: 14]
1. **Data Pipeline**:
   - [cite_start]Loaded the Shakespeare dataset in `.txt` format[cite: 9, 17].
   - [cite_start]Normalized text to lowercase and mapped characters to unique integers[cite: 18, 19].
   - [cite_start]Created a sliding window of 40 characters to generate input-output pairs[cite: 20].

2. **Model Specs**:
   - [cite_start]**Type**: Recurrent Neural Network (LSTM)[cite: 22].
   - [cite_start]**Layers**: One LSTM layer with 128 units followed by a Softmax Dense layer for probability distribution[cite: 23].
   - [cite_start]**Loss Function**: Categorical Crossentropy[cite: 24].

3. **Text Generation**:
   - [cite_start]The model takes a 40-character seed and predicts the 41st character[cite: 32, 34].
   - [cite_start]This process repeats iteratively to build a full sentence/paragraph[cite: 34].

### [cite_start]Bonus & Observations [cite: 45, 46]
During development, I noticed that a `STEP_SIZE` of 3 provided a good balance between dataset size and training speed. Increasing the LSTM units to 256 improves coherence but significantly increases training time per epoch.

### How to Execute
1. Install dependencies: `pip install -r requirements.txt`
2. Run the main script: `python lstm_text_gen.py`
