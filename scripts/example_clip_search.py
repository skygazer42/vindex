#!/usr/bin/env python3
"""
CN-CLIP + Milvus 使用示例

演示如何使用 clip_milvus_search.py 进行多模态检索：
1. 索引图像库
2. 图搜图
3. 文搜图
4. 图文匹配
"""

from pathlib import Path
from clip_milvus_search import CNClipEncoder, MilvusVectorStore, ClipSearchEngine


def example_basic_usage():
    """基础使用示例"""
    print("=" * 60)
    print("示例 1: 基础使用流程")
    print("=" * 60)

    # 1. 初始化编码器 (首次会下载模型)
    encoder = CNClipEncoder(model_name="ViT-B-16")

    # 2. 初始化向量库 (使用 Milvus Lite 本地文件)
    vector_store = MilvusVectorStore(
        collection_name="demo_images",
        dim=encoder.embedding_dim,
        uri="./demo_milvus.db"  # 本地文件模式
    )

    # 3. 创建搜索引擎
    engine = ClipSearchEngine(encoder, vector_store)

    # 4. 索引图像 (示例：假设有一个图像文件夹)
    # engine.index_folder("./test_images", recursive=True)

    print("初始化完成!")
    print(f"向量维度: {encoder.embedding_dim}")
    print(f"当前索引数量: {vector_store.count()}")

    return engine


def example_encode_and_search():
    """编码和搜索示例"""
    print("\n" + "=" * 60)
    print("示例 2: 手动编码和搜索")
    print("=" * 60)

    encoder = CNClipEncoder(model_name="ViT-B-16")

    # 单张图像编码
    # image_vec = encoder.encode_image("./test.jpg")
    # print(f"图像向量形状: {image_vec.shape}")

    # 文本编码
    text_vec = encoder.encode_text("一只可爱的猫咪")
    print(f"文本向量形状: {text_vec.shape}")

    # 批量文本编码
    texts = ["猫", "狗", "汽车", "风景"]
    text_vecs = encoder.encode_text(texts)
    print(f"批量文本向量形状: {text_vecs.shape}")

    # 计算文本之间的相似度
    import numpy as np
    similarity_matrix = text_vecs @ text_vecs.T
    print("\n文本相似度矩阵:")
    for i, t1 in enumerate(texts):
        for j, t2 in enumerate(texts):
            print(f"  '{t1}' vs '{t2}': {similarity_matrix[i, j]:.4f}")


def example_image_search():
    """图搜图示例"""
    print("\n" + "=" * 60)
    print("示例 3: 图搜图 (Image → Image)")
    print("=" * 60)

    print("""
工作流程:
1. 用户上传查询图像
2. 用 CLIP 图像编码器提取特征向量
3. 在 Milvus 中搜索最近邻
4. 返回相似图像列表

代码示例:
    results = engine.search_by_image(
        query_image="./query.jpg",
        top_k=10,
        threshold=0.5  # 只返回相似度 > 0.5 的结果
    )

    for r in results:
        print(f"相似度: {r.score:.4f}, 路径: {r.path}")
    """)


def example_text_search():
    """文搜图示例"""
    print("\n" + "=" * 60)
    print("示例 4: 文搜图 (Text → Image)")
    print("=" * 60)

    print("""
工作流程:
1. 用户输入文本描述 (支持中英文)
2. 用 CLIP 文本编码器提取特征向量
3. 在 Milvus 中搜索最近邻图像向量
4. 返回匹配图像列表

代码示例:
    # 中文搜索
    results = engine.search_by_text(
        query_text="一只在草地上奔跑的金毛犬",
        top_k=10
    )

    # 英文搜索
    results = engine.search_by_text(
        query_text="a golden retriever running on grass",
        top_k=10
    )

    # 简短关键词
    results = engine.search_by_text("日落", top_k=5)
    """)


def example_similarity_match():
    """图文匹配示例"""
    print("\n" + "=" * 60)
    print("示例 5: 图文匹配 (Image ↔ Text)")
    print("=" * 60)

    print("""
工作流程:
1. 编码图像得到图像向量
2. 编码文本得到文本向量
3. 计算余弦相似度

代码示例:
    similarity = engine.compute_image_text_similarity(
        image="./cat.jpg",
        text="一只猫"
    )
    print(f"相似度: {similarity:.4f}")

    # 批量匹配: 一张图 vs 多个文本
    image_vec = encoder.encode_image("./photo.jpg")

    candidates = ["猫", "狗", "汽车", "风景", "人物"]
    text_vecs = encoder.encode_text(candidates)

    scores = image_vec @ text_vecs.T

    for text, score in zip(candidates, scores):
        print(f"'{text}': {score:.4f}")
    """)


def example_full_pipeline():
    """完整流水线示例"""
    print("\n" + "=" * 60)
    print("示例 6: 完整流水线")
    print("=" * 60)

    code = '''
from clip_milvus_search import CNClipEncoder, MilvusVectorStore, ClipSearchEngine

# ========== 初始化 ==========
encoder = CNClipEncoder(model_name="ViT-B-16")

vector_store = MilvusVectorStore(
    collection_name="my_image_library",
    dim=512,
    uri="./my_milvus.db"  # 本地模式
    # 或连接服务器:
    # host="localhost", port=19530
)

engine = ClipSearchEngine(encoder, vector_store)

# ========== 索引图像库 ==========
# 方式1: 索引整个文件夹
engine.index_folder("./images", recursive=True, batch_size=32)

# 方式2: 索引指定图像列表
# engine.index_images(["img1.jpg", "img2.jpg", "img3.jpg"])

# ========== 图搜图 ==========
results = engine.search_by_image("./query.jpg", top_k=10)
for r in results:
    print(f"Score: {r.score:.4f} | {r.path}")

# ========== 文搜图 ==========
results = engine.search_by_text("蓝天白云下的草原", top_k=10)
for r in results:
    print(f"Score: {r.score:.4f} | {r.path}")

# ========== 图文匹配 ==========
score = engine.compute_image_text_similarity("./photo.jpg", "美丽的风景")
print(f"图文相似度: {score:.4f}")
'''
    print(code)


def example_milvus_server():
    """连接 Milvus Server 示例"""
    print("\n" + "=" * 60)
    print("示例 7: 连接 Milvus Server (生产环境)")
    print("=" * 60)

    print("""
# Docker 启动 Milvus (单机版)
docker run -d --name milvus-standalone \\
    -p 19530:19530 -p 9091:9091 \\
    milvusdb/milvus:latest

# Python 连接
vector_store = MilvusVectorStore(
    collection_name="production_images",
    dim=512,
    host="localhost",
    port=19530
    # uri=None  # 不使用 Milvus Lite
)

# 对于大规模数据 (>100万)，建议使用更高效的索引:
# - IVF_PQ: 压缩存储，速度快
# - HNSW: 召回率高，内存占用大
# - SCANN: Google 的高性能索引
    """)


def main():
    print("CN-CLIP + Milvus 多模态检索系统 - 使用示例")
    print("=" * 60)

    # 运行各示例
    # example_basic_usage()  # 需要实际环境
    example_encode_and_search()
    example_image_search()
    example_text_search()
    example_similarity_match()
    example_full_pipeline()
    example_milvus_server()

    print("\n" + "=" * 60)
    print("命令行使用方式:")
    print("=" * 60)
    print("""
# 索引图像
python clip_milvus_search.py index --folder ./images

# 图搜图
python clip_milvus_search.py search-image --query ./photo.jpg --top-k 10

# 文搜图
python clip_milvus_search.py search-text --query "一只猫" --top-k 10

# 图文匹配
python clip_milvus_search.py match --image ./cat.jpg --text "猫咪"
    """)


if __name__ == "__main__":
    main()
