# 🧠 Decoder-Only Transformer (GPT) From Scratch

An educational implementation of a **decoder-only Transformer (GPT)** built **fully from scratch using PyTorch**.  
The model is trained on **20 classic thriller novels** as a practical medium to **study the internal architecture, training workflow, and inference behavior of GPT-style models**, rather than to achieve high-quality text generation.

---

## 📌 Project Objective

The primary goal of this project is to **study and understand the internal workings of the decoder-only Transformer (GPT) architecture** by implementing it fully from scratch using PyTorch.

Specifically, this project focuses on:
- Understanding the **decoder-only Transformer (GPT) architecture** at a low level
- Implementing **self-attention, causal masking, multi-head attention, and feed-forward networks** from first principles
- Learning the **training and inference workflow** of GPT-style language models
- Exploring the roles of **tokenization, `<eos>` handling, and sampling strategies** in autoregressive text generation

---

## 📚 Dataset

- **Data Source:** 20 classic thriller novels (e.g., Dracula, Sherlock Holmes, etc.)
- All novels were:
  - Manually cleaned
  - Merged into a single `.txt` corpus
  - Structured chapter-wise
- A special token `<eos>` was inserted **after most chapters** to mark logical sequence endings

### Dataset Statistics
- Total characters: ~9,057,159
- Language: English
- Domain: Thriller / Mystery fiction

---

## 🔤 Tokenization

- **Tokenizer:** Byte Pair Encoding (BPE)
- **Library:** `tokenizers.ByteLevelBPETokenizer`
- **Vocabulary Size:** 8,000
- **Special Tokens:** `<eos>`

Tokenization steps:
1. Train BPE tokenizer on the cleaned corpus
2. Encode the full text into token IDs
3. Convert tokens into fixed-length blocks for training

---

## 🔄 Data Preparation

- **Block size:** 256 tokens
- Tokens reshaped into sequences of fixed length
- Dataset split:
  - **90% Training**
  - **10% Validation**

### Language Modeling Setup
- Input (`x`): tokens `[0 … T-1]`
- Target (`y`): tokens `[1 … T]`
- This follows the standard **next-token prediction** objective used in GPT models

---

## 🏗️ Model Architecture (From Scratch)

### Core Configuration
- **Architecture:** Decoder-only Transformer (GPT-style)
- **Embedding size:** 256
- **Number of layers:** 10
- **Number of attention heads:** 4
- **Hidden dimension (FFN):** 4 × embedding size
- **Context length:** 255 tokens

---

### Implemented Components

The model was implemented **without using `nn.Transformer` or prebuilt GPT modules**.

#### 1️⃣ Token Embedding  
- Maps token IDs to dense vectors

#### 2️⃣ Positional Embedding  
- Learned positional embeddings added to token embeddings

#### 3️⃣ Multi-Head Self-Attention
- Manual implementation of:
  - Query, Key, Value projections
  - Scaled dot-product attention
  - **Causal (triangular) masking**
  - Head splitting and recombination

#### 4️⃣ Residual Connections + Layer Normalization
- Applied before attention and feed-forward blocks

#### 5️⃣ Feed-Forward Network (FFN)
- Two linear layers with **GELU activation**

#### 6️⃣ Output Head
- Linear projection from embeddings to vocabulary size

---

## 🧪 Training Details

- **Loss Function:** Cross-Entropy Loss
- **Optimizer:** AdamW
- **Learning Rate:** 3e-4
- **Training Steps:** 1700
- **Device:** GPU (CUDA) when available

Training involved:
- Random batch sampling
- Continuous evaluation on validation data
- Monitoring training and validation loss

---

## ✍️ Text Generation

After training, the model generates text using:

- **Autoregressive decoding**
- **Top-p (nucleus) sampling**
  - `p = 0.90`
- Generation stops when:
  - `<eos>` token is produced
  - Maximum token limit is reached

### Generation Process
1. Encode input prompt
2. Use last 255 tokens as context
3. Predict next token
4. Sample using top-p strategy
5. Append token and repeat

---

## 📈 Results & Observations

After training, the model demonstrates a clear understanding of:

- English sentence structure and grammar
- Dialogue formatting commonly found in novels
- Narrative-style text flow
- Thriller and mystery tone learned from the dataset

The generated text shows coherent sentence-level structure and realistic literary formatting, indicating that the model successfully learned **syntactic and stylistic patterns** from the training data.

However, the model does not consistently generate fully coherent or logically connected stories over long passages. This behavior is expected due to:

- Training on a relatively small dataset (20 novels)
- Limited training steps and compute
- Small vocabulary size and context window
- Absence of large-scale pretraining

The primary intention of this project was **not to produce high-quality storytelling**, but to **understand and implement the decoder-only Transformer (GPT) architecture from first principles**, including its training and generation workflow.

---

## 🖥️ Demo & Notebook

### 📓 Google Colab Notebook
The full implementation (training + generation) is available here:

👉 **Open in Colab:**  
https://colab.research.google.com/drive/1JMZW0zqDP1iFWVM4I6sLwTYiwhHunGHS?usp=drive_link

---

### 🎥 Demo Video
A short demo showing text generation:

👉 **Watch Demo:**  
https://drive.google.com/file/d/1EFmuhUZjSecsa5cRqjwiQ4nr8dkz7aOG/view?usp=drive_link

---

## 🛠️ Tech Stack

- **Language:** Python
- **Framework:** PyTorch
- **Libraries:**
  - NumPy
  - tokenizers
  - Google Colab
- **Hardware:** Colab GPU (CUDA)

---

## 📚 Key Learnings

- Deep understanding of decoder-only Transformer (GPT) architecture
- Implementation of self-attention, causal masking, and multi-head attention from scratch
- Practical role of tokenization and `<eos>` tokens in language modeling
- How sampling strategies influence text generation behavior
- End-to-end workflow of training and inference in GPT-style models
- Insight into the limitations of small-scale language models trained with limited data and compute
