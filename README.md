# KV Cache Implementation for a Transformer Decoder

## Project Overview
This project explores the impact of implementing a **Key-Value (KV) Cache** in a Transformer decoder on **end-to-end inference latency** and **memory usage**. The goal is to evaluate how caching key and value vectors during the self-attention mechanism can optimize the performance of a Transformer model during inference.

### Objective
- **Primary Question**: How does implementing a KV cache in a Transformer decoder affect end-to-end inference latency and memory usage?
- **Focus**: Analyze the trade-offs between computational efficiency and memory consumption when using a KV cache.

---

## Methodology
1. **Dataset**: Train and evaluate the model on the **Penn Treebank (PTB) dataset** for text completion tasks.
2. **Model**: Implement a custom Transformer decoder with a self-attention mechanism.
3. **KV Cache Implementation**:
   - Integrate a KV cache into the self-attention mechanism to store and reuse key-value pairs from previous tokens during inference.
   - Compare the performance of the model with and without the KV cache.
4. **Evaluation Metrics**:
   - **Inference Latency**: Measure the time taken for text generation with and without the KV cache.
   - **Memory Usage**: Track GPU memory consumption during inference.

---

## Key Features
- **Custom Transformer Decoder**: A lightweight Transformer decoder tailored for text completion tasks.
- **KV Cache Integration**: Efficient implementation of a KV cache to reduce redundant computations during inference.
- **Benchmarking Scripts**: Tools to measure inference latency and memory usage.

---
