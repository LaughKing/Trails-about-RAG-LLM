import os
import pandas as pd
import kaggle
import zipfile
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer

def download_quora_data():
    dataset = "quora/question-pairs-dataset"
    download_path = "./data"
    kaggle.api.dataset_download_files(dataset, path=download_path, unzip=False)
    zip_path = os.path.join(download_path, "question-pairs-dataset.zip")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(download_path)
    csv_path = os.path.join(download_path, "questions.csv")
    return csv_path

def prepare_data(csv_path, sample_size=10000):
    df = pd.read_csv(csv_path)
    df_sample = df.sample(n=sample_size, random_state=42)
    texts = df_sample["question1"].fillna("").str.strip().tolist()
    texts = [text if len(text) <= 500 else text[:500] for text in texts]
    texts = [text for text in texts if len(text) > 0]
    pd.DataFrame({"text": texts}).to_csv("./data/faq_data.csv", index=False)
    return texts

def setup_collection(texts, collection_name="quora_faq", overwrite=True):
    # 连接 Milvus
    try:
        connections.connect(host="localhost", port="19530")
        print("成功连接到 Milvus")
    except Exception as e:
        raise Exception(f"无法连接到 Milvus: {e}")

    # 定义 schema
    fields = [
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=384),
        FieldSchema("text", DataType.VARCHAR, max_length=65535)
    ]
    schema = CollectionSchema(fields, description="Quora FAQ collection")
    
    # 检查并清理集合
    if utility.has_collection(collection_name):
        if overwrite:
            utility.drop_collection(collection_name)
            print(f"删除已有集合: {collection_name}")
        else:
            collection = Collection(collection_name)
            print(f"集合 {collection_name} 已存在，跳过插入")
            collection.load()
            print(f"集合中的实际条数: {collection.num_entities}")
            return collection

    # 创建新集合
    collection = Collection(collection_name, schema)
    print(f"创建新集合: {collection_name}")

    # 向量化
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, convert_to_numpy=True).astype('float32')
    print(f"向量形状: {embeddings.shape}, 文本数量: {len(texts)}")

    # 插入数据
    collection.insert([embeddings, texts])
    collection.flush()
    print(f"插入 {collection.num_entities} 条数据")

    # 创建索引
    index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 100}}
    collection.create_index("embedding", index_params)
    print("索引创建完成")

    # 加载集合
    collection.load()
    print("集合加载完成")
    
    print(f"集合中的实际条数: {collection.num_entities}")
    return collection

if __name__ == "__main__":
    # 直接用已有的 faq_data.csv
    df = pd.read_csv("./data/faq_data.csv")
    texts = df["text"].tolist()
    
    # 设置 Milvus 集合，强制覆盖
    collection = setup_collection(texts, overwrite=True)
    print(f"已存储 {collection.num_entities} 条数据到 Milvus")