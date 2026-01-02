# 🧠 Decoder-Only Transformer From Scratch (GPT-Style)

An educational implementation of a **decoder-only Transformer (GPT-style)** model built **fully from scratch using PyTorch**.  
The model is trained on **20 classic thriller novels** to study **text generation**, **GPT architecture internals**, and the **end-to-end workflow of decoder-only Transformers**.

> ⚠️ This project is created **purely for learning and educational purposes** to understand how GPT-style models work internally.

---

## 📌 Project Objective

The goal of this project is to:
- Understand the **decoder-only Transformer (GPT) architecture**
- Implement **self-attention, masking, multi-head attention, and feed-forward layers** from scratch
- Learn how **language models are trained and used for text generation**
- Explore how **tokenization, EOS handling, and sampling strategies** affect generation quality

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

## 📈 Results

- The model successfully learned:
  - Sentence structure
  - Narrative flow
  - Thriller-style storytelling patterns
- Generated text shows:
  - Coherent paragraphs
  - Context awareness
  - Story-like progression

> This project prioritizes **learning and understanding** over benchmark performance.

---

## 🖥️ Demo & Notebook

### 📓 Google Colab Notebook
The full implementation (training + generation) is available here:

👉 **Open in Colab:**  
`PASTE_YOUR_COLAB_LINK_HERE`

> Note: Opening the link creates a **view-only copy**. You can safely explore or duplicate it to your own Drive.

---

### 🎥 Demo Video
A short demo showing text generation:

👉 **Watch Demo:**  
`PASTE_YOUR_VIDEO_LINK_HERE`

---

## 🛠️ Tech Stack

- **Language:** Python
- **Framework:** PyTorch
- **Libraries:**
  - NumPy
  - tokenizers
  - Google Colab
- **Hardware:** GPU (CUDA)

---

## ⚠️ Limitations

- Trained on a relatively small corpus
- No large-scale optimization or distributed training
- Intended for **educational exploration**, not production use

---

## 📚 Key Learnings

- How GPT-style models work internally
- Importance of causal masking
- Role of tokenization and EOS tokens
- Impact of sampling strategies on text generation
- End-to-end Transformer training workflow
