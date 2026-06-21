MiniBERT for Extractive Question Answering from Scratch

A complete implementation of a MiniBERT-style Transformer for Extractive Question Answering built entirely from scratch using TensorFlow/Keras. This project recreates the core components of BERT without relying on pretrained models or high-level NLP libraries, providing an end-to-end understanding of Transformer-based question answering systems.

The model is trained on the SQuAD v1.1 dataset and predicts the start and end positions of the answer span within a given context passage.

Features
🚀 MiniBERT implementation from scratch using TensorFlow/Keras
🧠 BERT-style Token, Position, and Segment Embeddings
🔍 Multi-Head Self Attention
🔄 Transformer Encoder Stack with Residual Connections
⚡ Feed Forward Networks with GELU activation
❓ Question Answering Head for start/end span prediction
📚 Custom SQuAD v1.1 preprocessing pipeline
🔧 TensorFlow tf.data input pipeline
⚙️ Mixed Precision Training for faster GPU execution
📈 Learning Rate Warmup with Linear Decay
✂️ Gradient Clipping
📊 Evaluation using Exact Match (EM) and F1 Score


# Architecture
Input
 │
 ├── Token Embeddings
 ├── Position Embeddings
 └── Segment Embeddings
          │
          ▼
Layer Normalization + Dropout
          │
          ▼
MiniBERT Encoder (N Layers)
          │
          ├── Multi-Head Self Attention
          ├── Residual Connection
          ├── Layer Normalization
          ├── Feed Forward Network (GELU)
          ├── Residual Connection
          └── Layer Normalization
          │
          ▼
Question Answering Head
          │
          ▼
Start Logits      End Logits


# Model Configuration
MAX_SEQ_LEN = 128

VOCAB_SIZE = 30000

D_MODEL = 256
NUM_HEADS = 4
NUM_LAYERS = 4
FFN_DIM = 1024

DROPOUT_RATE = 0.1

BATCH_SIZE = 16
EPOCHS = 3

LEARNING_RATE = 5e-5

# Dataset
This project uses the Stanford Question Answering Dataset (SQuAD v1.1).

# Training Pipeline
The training pipeline includes:
Adam Optimizer
Learning Rate Warmup
Linear Learning Rate Decay
Mixed Precision Training
Gradient Clipping
Custom Training Loop using tf.GradientTape
Validation after every epoch
Exact Match (EM) Evaluation
F1 Score Evaluation
Best Model Checkpoint Saving
Results

The primary objective of this repository is to understand and implement BERT internals from first principles.

Although the architecture is correctly implemented, the current performance is limited due to simplified preprocessing and tokenization strategies.

# Current Limitations
Uses a simple tokenizer instead of WordPiece tokenization.
Answer span generation can be improved using SQuAD character-level offsets.
Does not include BERT pretraining objectives such as:
Masked Language Modeling (MLM)
Next Sentence Prediction (NSP)
Sequence length is limited to 128 tokens.
Intended for educational purposes rather than state-of-the-art performance.
Future Improvements
Implement WordPiece Tokenizer
Use character-level answer alignment
Add sliding-window preprocessing for long contexts
Support larger MiniBERT configurations
Implement MLM and NSP pretraining
Export the model for inference and deployment
