from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import numpy as np

def retrieve_similar_texts(question, collection_name="quora_faq", k=3):
    """
    从 Milvus 中检索与输入问题相似的前 k 个文本。
    
    参数:
    - question (str): 输入的问题
    - collection_name (str): Milvus 集合名称，默认为 "quora_faq"
    - k (int): 返回的相似文本数量，默认为 3
    
    返回:
    - list: 相似文本列表
    """
    # 连接 Milvus
    try:
        connections.connect(host="localhost", port="19530")
        print("成功连接到 Milvus")
    except Exception as e:
        raise Exception(f"无法连接到 Milvus: {e}")
    
    # 加载集合
    try:
        collection = Collection(collection_name)
        collection.load()
        print(f"集合 {collection_name} 已加载")
    except Exception as e:
        raise Exception(f"加载集合 {collection_name} 失败: {e}")
    
    # 向量化输入问题
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_vector = model.encode([question], convert_to_numpy=True).astype('float32')
    print(f"查询向量形状: {query_vector.shape}")
    
    # 检索
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=query_vector,
        anns_field="embedding",
        param=search_params,
        limit=k,
        output_fields=["text"]
    )
    
    # 调试输出结果类型
    print(f"results[0] 类型: {type(results[0])}")
    
    # 提取相似文本
    similar_texts = [hit.entity.get("text") for hit in list(results[0])]  # 强制转换为列表
    return similar_texts

if __name__ == "__main__":
    # 测试检索
    question = "How do I learn coding?"
    try:
        similar_texts = retrieve_similar_texts(question)
        print("检索到的相似文本:")
        for i, text in enumerate(similar_texts):
            print(f"{i+1}. {text}")
    except Exception as e:
        print(f"检索失败: {e}")