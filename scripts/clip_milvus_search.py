#!/usr/bin/env python3
"""
基于 CN-CLIP + Milvus 的多模态检索系统

功能：
  - 图搜图 (Image → Image): 用图像查找相似图像
  - 文搜图 (Text → Image): 用文本查找匹配图像
  - 图文匹配 (Image ↔ Text): 计算图文相似度

原理：
  CLIP 模型将图像和文本编码到同一向量空间，
  通过余弦相似度/L2距离实现跨模态检索。

使用：
  python clip_milvus_search.py --help
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

# Milvus
try:
    from pymilvus import (
        connections, utility,
        Collection, CollectionSchema, FieldSchema, DataType,
        MilvusClient
    )
    HAS_MILVUS = True
except ImportError:
    HAS_MILVUS = False
    print("Warning: pymilvus not installed. Run: pip install pymilvus")

# CN-CLIP
try:
    import cn_clip.clip as clip
    from cn_clip.clip import load_from_name
    HAS_CN_CLIP = True
except ImportError:
    HAS_CN_CLIP = False
    print("Warning: cn_clip not installed. Run: pip install cn_clip")


@dataclass
class SearchResult:
    """搜索结果"""
    id: int
    path: str
    score: float  # 相似度分数 (越高越相似)
    distance: float  # L2距离 (越低越相似)


class CNClipEncoder:
    """CN-CLIP 编码器 - 统一的图像/文本编码接口"""

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        device: str = None
    ):
        """
        初始化 CN-CLIP 编码器

        Args:
            model_name: 模型名称，支持 ViT-B-16, ViT-L-14, ViT-L-14-336, ViT-H-14, RN50
            device: 设备，None 时自动选择
        """
        if not HAS_CN_CLIP:
            raise ImportError("cn_clip not installed. Run: pip install cn_clip")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CN-CLIP model: {model_name} on {self.device}")

        self.model, self.preprocess = load_from_name(
            model_name,
            device=self.device,
            download_root='./assets/models/cn-clip'
        )
        self.model.eval()

        # 获取特征维度
        self.embedding_dim = self.model.visual.output_dim
        print(f"Embedding dimension: {self.embedding_dim}")

    @torch.no_grad()
    def encode_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """
        编码单张图像

        Args:
            image: 图像路径、PIL Image 或 numpy 数组

        Returns:
            归一化的特征向量 (embedding_dim,)
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')

        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)  # L2 归一化

        return features.cpu().numpy().flatten()

    @torch.no_grad()
    def encode_images(self, images: List[Union[str, Path, Image.Image]], batch_size: int = 32) -> np.ndarray:
        """
        批量编码图像

        Args:
            images: 图像列表
            batch_size: 批次大小

        Returns:
            特征矩阵 (N, embedding_dim)
        """
        all_features = []

        for i in tqdm(range(0, len(images), batch_size), desc="Encoding images"):
            batch = images[i:i + batch_size]
            batch_tensors = []

            for img in batch:
                if isinstance(img, (str, Path)):
                    img = Image.open(img).convert('RGB')
                elif isinstance(img, np.ndarray):
                    img = Image.fromarray(img).convert('RGB')
                batch_tensors.append(self.preprocess(img))

            batch_tensor = torch.stack(batch_tensors).to(self.device)
            features = self.model.encode_image(batch_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())

        return np.vstack(all_features)

    @torch.no_grad()
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        编码文本

        Args:
            text: 单个文本或文本列表

        Returns:
            特征向量或特征矩阵
        """
        if isinstance(text, str):
            text = [text]

        text_tokens = clip.tokenize(text).to(self.device)
        features = self.model.encode_text(text_tokens)
        features = features / features.norm(dim=-1, keepdim=True)

        result = features.cpu().numpy()
        return result.flatten() if len(text) == 1 else result

    def compute_similarity(
        self,
        query_features: np.ndarray,
        gallery_features: np.ndarray
    ) -> np.ndarray:
        """
        计算相似度分数

        Args:
            query_features: 查询特征 (D,) 或 (N, D)
            gallery_features: 库特征 (M, D)

        Returns:
            相似度矩阵，余弦相似度 [-1, 1]，已归一化的向量点积即为余弦
        """
        if query_features.ndim == 1:
            query_features = query_features.reshape(1, -1)

        # 对于归一化向量，点积 = 余弦相似度
        similarity = query_features @ gallery_features.T
        return similarity


class MilvusVectorStore:
    """Milvus 向量存储 - 管理图像向量的存储和检索"""

    def __init__(
        self,
        collection_name: str = "image_vectors",
        dim: int = 512,
        host: str = "localhost",
        port: int = 19530,
        uri: str = None  # 使用 Milvus Lite 时的文件路径
    ):
        """
        初始化 Milvus 向量存储

        Args:
            collection_name: 集合名称
            dim: 向量维度
            host: Milvus 服务地址
            port: Milvus 服务端口
            uri: Milvus Lite 文件路径 (如 "./milvus.db")
        """
        if not HAS_MILVUS:
            raise ImportError("pymilvus not installed. Run: pip install pymilvus")

        self.collection_name = collection_name
        self.dim = dim
        self.uri = uri

        # 连接 Milvus
        if uri:
            # Milvus Lite (本地文件模式)
            print(f"Connecting to Milvus Lite: {uri}")
            self.client = MilvusClient(uri=uri)
            self._use_client_api = True
        else:
            # Milvus Server
            print(f"Connecting to Milvus Server: {host}:{port}")
            connections.connect(host=host, port=port)
            self._use_client_api = False

        self._ensure_collection()

    def _ensure_collection(self):
        """确保集合存在"""
        if self._use_client_api:
            # Milvus Lite / Client API
            if not self.client.has_collection(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    dimension=self.dim,
                    metric_type="COSINE",  # 使用余弦相似度
                    auto_id=False
                )
                print(f"Created collection: {self.collection_name}")
            else:
                print(f"Collection exists: {self.collection_name}")
        else:
            # Server API
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                print(f"Collection exists: {self.collection_name}")
            else:
                # 创建 Schema
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                    FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=512),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
                ]
                schema = CollectionSchema(fields, description="Image vectors for CLIP search")

                self.collection = Collection(self.collection_name, schema)

                # 创建索引 (IVF_FLAT 适合中等规模数据)
                index_params = {
                    "metric_type": "IP",  # Inner Product (对归一化向量等价于余弦)
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }
                self.collection.create_index("vector", index_params)
                print(f"Created collection with IVF_FLAT index: {self.collection_name}")

    def insert(
        self,
        ids: List[int],
        paths: List[str],
        vectors: np.ndarray
    ) -> int:
        """
        插入向量

        Args:
            ids: ID 列表
            paths: 图像路径列表
            vectors: 向量矩阵 (N, dim)

        Returns:
            插入数量
        """
        if self._use_client_api:
            data = [
                {"id": int(id_), "path": path, "vector": vec.tolist()}
                for id_, path, vec in zip(ids, paths, vectors)
            ]
            self.client.insert(self.collection_name, data)
        else:
            entities = [ids, paths, vectors.tolist()]
            self.collection.insert(entities)
            self.collection.flush()

        return len(ids)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[SearchResult]:
        """
        搜索最相似的向量

        Args:
            query_vector: 查询向量 (dim,)
            top_k: 返回数量
            threshold: 相似度阈值 (0-1)

        Returns:
            搜索结果列表
        """
        query_vector = query_vector.reshape(1, -1).tolist()

        if self._use_client_api:
            results = self.client.search(
                collection_name=self.collection_name,
                data=query_vector,
                limit=top_k,
                output_fields=["path"]
            )

            search_results = []
            for hit in results[0]:
                # Milvus Lite 使用 COSINE，distance 就是相似度
                score = hit['distance']  # COSINE metric 返回的是相似度
                if score >= threshold:
                    search_results.append(SearchResult(
                        id=hit['id'],
                        path=hit['entity']['path'],
                        score=score,
                        distance=1 - score  # 转换为距离
                    ))
        else:
            self.collection.load()

            search_params = {"metric_type": "IP", "params": {"nprobe": 16}}
            results = self.collection.search(
                data=query_vector,
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["path"]
            )

            search_results = []
            for hit in results[0]:
                score = hit.score  # IP 对归一化向量就是余弦相似度
                if score >= threshold:
                    search_results.append(SearchResult(
                        id=hit.id,
                        path=hit.entity.get('path', ''),
                        score=score,
                        distance=1 - score
                    ))

        return search_results

    def count(self) -> int:
        """获取向量数量"""
        if self._use_client_api:
            stats = self.client.get_collection_stats(self.collection_name)
            return stats.get('row_count', 0)
        else:
            return self.collection.num_entities

    def drop(self):
        """删除集合"""
        if self._use_client_api:
            self.client.drop_collection(self.collection_name)
        else:
            utility.drop_collection(self.collection_name)
        print(f"Dropped collection: {self.collection_name}")


class ClipSearchEngine:
    """
    CLIP 多模态搜索引擎

    统一接口实现：
    - 图搜图 (search_by_image)
    - 文搜图 (search_by_text)
    - 图文匹配 (compute_similarity)
    """

    def __init__(
        self,
        encoder: CNClipEncoder,
        vector_store: MilvusVectorStore
    ):
        self.encoder = encoder
        self.vector_store = vector_store

    def index_images(
        self,
        image_paths: List[str],
        batch_size: int = 32,
        start_id: int = 0
    ) -> int:
        """
        将图像编码并存入向量库

        Args:
            image_paths: 图像路径列表
            batch_size: 编码批次大小
            start_id: 起始 ID

        Returns:
            成功索引的数量
        """
        print(f"Indexing {len(image_paths)} images...")

        # 过滤有效图像
        valid_paths = []
        for p in image_paths:
            if Path(p).exists() and Path(p).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                valid_paths.append(p)

        if not valid_paths:
            print("No valid images found.")
            return 0

        # 批量编码
        vectors = self.encoder.encode_images(valid_paths, batch_size)

        # 生成 ID
        ids = list(range(start_id, start_id + len(valid_paths)))

        # 插入向量库
        count = self.vector_store.insert(ids, valid_paths, vectors)
        print(f"Indexed {count} images. Total: {self.vector_store.count()}")

        return count

    def index_folder(
        self,
        folder_path: str,
        recursive: bool = True,
        batch_size: int = 32
    ) -> int:
        """
        索引文件夹中的所有图像

        Args:
            folder_path: 文件夹路径
            recursive: 是否递归子文件夹
            batch_size: 批次大小

        Returns:
            索引数量
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # 收集图像文件
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        if recursive:
            image_paths = [str(p) for p in folder.rglob('*') if p.suffix.lower() in extensions]
        else:
            image_paths = [str(p) for p in folder.glob('*') if p.suffix.lower() in extensions]

        print(f"Found {len(image_paths)} images in {folder_path}")

        # 获取当前最大 ID
        start_id = self.vector_store.count()

        return self.index_images(image_paths, batch_size, start_id)

    def search_by_image(
        self,
        query_image: Union[str, Path, Image.Image],
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[SearchResult]:
        """
        图搜图：用图像查找相似图像

        Args:
            query_image: 查询图像 (路径或 PIL Image)
            top_k: 返回数量
            threshold: 相似度阈值

        Returns:
            搜索结果列表
        """
        # 编码查询图像
        query_vector = self.encoder.encode_image(query_image)

        # 向量检索
        results = self.vector_store.search(query_vector, top_k, threshold)

        return results

    def search_by_text(
        self,
        query_text: str,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[SearchResult]:
        """
        文搜图：用文本查找匹配图像

        Args:
            query_text: 查询文本 (支持中英文)
            top_k: 返回数量
            threshold: 相似度阈值

        Returns:
            搜索结果列表
        """
        # 编码查询文本
        query_vector = self.encoder.encode_text(query_text)

        # 向量检索
        results = self.vector_store.search(query_vector, top_k, threshold)

        return results

    def compute_image_text_similarity(
        self,
        image: Union[str, Path, Image.Image],
        text: str
    ) -> float:
        """
        计算图文相似度

        Args:
            image: 图像
            text: 文本

        Returns:
            相似度分数 [0, 1]
        """
        image_features = self.encoder.encode_image(image)
        text_features = self.encoder.encode_text(text)

        # 余弦相似度 (归一化向量的点积)
        similarity = float(np.dot(image_features, text_features))

        # 转换到 [0, 1] 范围
        return (similarity + 1) / 2


def scan_images(folder: str, recursive: bool = True) -> List[str]:
    """扫描文件夹中的图像"""
    folder = Path(folder)
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    if recursive:
        return [str(p) for p in folder.rglob('*') if p.suffix.lower() in extensions]
    else:
        return [str(p) for p in folder.glob('*') if p.suffix.lower() in extensions]


def main():
    parser = argparse.ArgumentParser(
        description="CN-CLIP + Milvus 多模态检索系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 索引图像文件夹
  python clip_milvus_search.py index --folder ./images

  # 图搜图
  python clip_milvus_search.py search-image --query ./query.jpg --top-k 10

  # 文搜图
  python clip_milvus_search.py search-text --query "一只猫" --top-k 10

  # 计算图文相似度
  python clip_milvus_search.py match --image ./cat.jpg --text "一只可爱的猫"
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="命令")

    # 索引命令
    index_parser = subparsers.add_parser("index", help="索引图像文件夹")
    index_parser.add_argument("--folder", required=True, help="图像文件夹路径")
    index_parser.add_argument("--recursive", action="store_true", default=True, help="递归子文件夹")
    index_parser.add_argument("--batch-size", type=int, default=32, help="批次大小")

    # 图搜图命令
    img_parser = subparsers.add_parser("search-image", help="图搜图")
    img_parser.add_argument("--query", required=True, help="查询图像路径")
    img_parser.add_argument("--top-k", type=int, default=10, help="返回数量")
    img_parser.add_argument("--threshold", type=float, default=0.0, help="相似度阈值")

    # 文搜图命令
    txt_parser = subparsers.add_parser("search-text", help="文搜图")
    txt_parser.add_argument("--query", required=True, help="查询文本")
    txt_parser.add_argument("--top-k", type=int, default=10, help="返回数量")
    txt_parser.add_argument("--threshold", type=float, default=0.0, help="相似度阈值")

    # 图文匹配命令
    match_parser = subparsers.add_parser("match", help="计算图文相似度")
    match_parser.add_argument("--image", required=True, help="图像路径")
    match_parser.add_argument("--text", required=True, help="文本")

    # 通用参数
    parser.add_argument("--model", default="ViT-B-16", help="CN-CLIP 模型名称")
    parser.add_argument("--milvus-uri", default="./milvus_clip.db", help="Milvus Lite 数据库路径")
    parser.add_argument("--collection", default="clip_images", help="Milvus 集合名称")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # 初始化编码器
    print("=" * 60)
    print("CN-CLIP + Milvus 多模态检索系统")
    print("=" * 60)

    encoder = CNClipEncoder(model_name=args.model)

    # 初始化向量库
    vector_store = MilvusVectorStore(
        collection_name=args.collection,
        dim=encoder.embedding_dim,
        uri=args.milvus_uri
    )

    # 创建搜索引擎
    engine = ClipSearchEngine(encoder, vector_store)

    # 执行命令
    if args.command == "index":
        count = engine.index_folder(
            args.folder,
            recursive=args.recursive,
            batch_size=args.batch_size
        )
        print(f"\nIndexed {count} images successfully!")

    elif args.command == "search-image":
        print(f"\nSearching by image: {args.query}")
        results = engine.search_by_image(args.query, args.top_k, args.threshold)

        print(f"\nTop {len(results)} results:")
        print("-" * 60)
        for i, r in enumerate(results, 1):
            print(f"{i}. Score: {r.score:.4f} | {r.path}")

    elif args.command == "search-text":
        print(f"\nSearching by text: {args.query}")
        results = engine.search_by_text(args.query, args.top_k, args.threshold)

        print(f"\nTop {len(results)} results:")
        print("-" * 60)
        for i, r in enumerate(results, 1):
            print(f"{i}. Score: {r.score:.4f} | {r.path}")

    elif args.command == "match":
        similarity = engine.compute_image_text_similarity(args.image, args.text)
        print(f"\nImage: {args.image}")
        print(f"Text: {args.text}")
        print(f"Similarity: {similarity:.4f}")


if __name__ == "__main__":
    main()
