# Local QA System with Milvus and GPT-2

A lightweight, local question-answering system built with Milvus for vector search and GPT-2 for answer generation. Supports GPU acceleration for faster inference.

## Features
- Retrieve similar questions from a Milvus vector database.
- Generate concise, helpful answers using GPT-2 (or GPT-2 Medium).
- Command-line interface for interactive Q&A.
- Runs locally, no external API required.

## Prerequisites
- Python 3.8+
- NVIDIA GPU (optional, for faster inference)
- Milvus server running locally (`localhost:19530`)

## Usage
```bash
python data_manager.py
python retrieval.py
python app.py
```
## Project Structure
data_manager.py: Import FAQ data into Milvus.
retrieval.py: Retrieve similar questions from Milvus.
app.py: Generate answers using GPT-2.

## Example:
==================================================
成功连接到 Milvus
集合 quora_faq 已加载
查询向量形状: (1, 384)
检索到的相似文本:
1. How do I learn coding?
2. How should I learn coding?
3. I want to start to learning how to code. (No coding experience at all)?
使用设备: GPU
Device set to use cuda:0

本地模型回答:
"How do I learn coding?" In other words, do you want to learn coding, or do you want to learn coding?.
The first question that comes up is how do you learn coding? The answer is: you need to be a good programmer. You need to have the skills you need to make a career in programming. This is what I mean by the phrase "you need to be a good programmer". If you do not have the skills, you won't make it in the industry. This is the key to a successful career in programming.
What does the word "programming" mean?
It means: a skill in programming.
I will say that this is a very simple phrase. But it is very important. In fact, I think it is very important to understand this phrase. It means: a skill in programming.      


## Customization:
**Model**:Edit app.py to use gpt2 or other models
**Search Limit**:Adjust k in retrieve_similar_texts(retrieval.py) for more or less context
**Answer Length**:Change max_new_tokens in app.py

