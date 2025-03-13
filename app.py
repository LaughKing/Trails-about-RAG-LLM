import os
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline
import torch

def retrieve_similar_texts(question, collection_name="quora_faq", k=3):
    """从 Milvus 检索相似文本"""
    try:
        connections.connect(host="localhost", port="19530")
        print("成功连接到 Milvus")
    except Exception as e:
        raise Exception(f"无法连接到 Milvus: {e}")
    
    try:
        collection = Collection(collection_name)
        collection.load()
        print(f"集合 {collection_name} 已加载")
    except Exception as e:
        raise Exception(f"加载集合 {collection_name} 失败: {e}")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_vector = model.encode([question], convert_to_numpy=True).astype('float32')
    print(f"查询向量形状: {query_vector.shape}")
    
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=query_vector,
        anns_field="embedding",
        param=search_params,
        limit=k,
        output_fields=["text"]
    )
    
    similar_texts = [hit.entity.get("text") for hit in list(results[0])]
    return similar_texts

def get_local_response(question, similar_texts):
    """使用本地 GPT-2 模型生成回答"""
    # 强化提示
    prompt = f"Provide a concise, helpful answer to '{question}' using this context: {'; '.join(similar_texts)}. Avoid repeating the question: "

    # 检查 GPU 可用性
    device = 0 if torch.cuda.is_available() else -1
    print(f"使用设备: {'GPU' if device == 0 else 'CPU'}")

    # 初始化 GPT-2 模型
    generator = pipeline("text-generation", model="gpt2", device=device)
    response = generator(
        prompt,
        max_new_tokens=200,
        num_return_sequences=1,
        truncation=True,
        pad_token_id=50256,
        temperature=0.7,  # 稍提高随机性
        top_p=0.9,  # 增强连贯性
        top_k=50  # 扩大词汇范围
    )
    
    # 提取回答，去掉提示
    answer = response[0]["generated_text"].replace(prompt, "").strip()
    return answer

def answer_question(question):
    """完整问答流程"""
    similar_texts = retrieve_similar_texts(question)
    print("检索到的相似文本:")
    for i, text in enumerate(similar_texts):
        print(f"{i+1}. {text}")
    
    answer = get_local_response(question, similar_texts)
    print(f"\n本地模型回答:\n{answer}")
    return answer

if __name__ == "__main__":
    questions = [
        "How do I learn coding?",
        
    ]
    for question in questions:
        print("\n" + "="*50)
        try:
            answer_question(question)
        except Exception as e:
            print(f"问答失败: {e}")