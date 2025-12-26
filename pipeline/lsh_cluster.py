"""
LSH聚类工具模块

提供基于Locality-Sensitive Hashing的文本聚类功能，
用于对推荐意见进行相似度聚类和分桶索引。
"""
from typing import Optional, List, Dict, Set, Tuple
from hashlib import md5
from dataclasses import dataclass

from .models import Provenance, ProvenanceCluster


def compute_shingles(text: str, k: int = 3) -> set:
    """计算文本的k-shingles"""
    text = text.lower()
    return set(text[i:i+k] for i in range(len(text) - k + 1))


def compute_minhash(shingles: set, num_hashes: int = 100) -> List[int]:
    """计算MinHash签名"""
    if not shingles:
        return [0] * num_hashes

    signatures = []
    for i in range(num_hashes):
        min_hash = float('inf')
        for shingle in shingles:
            # 使用不同的哈希种子
            h = int(md5(f"{i}_{shingle}".encode()).hexdigest(), 16)
            min_hash = min(min_hash, h)
        signatures.append(min_hash)
    return signatures


@dataclass
class LSHResult:
    """LSH聚类结果，包含聚类和桶索引信息"""
    clusters: List[ProvenanceCluster]
    bucket_index: Dict[int, List[int]]  # bucket_id -> provenance原始索引列表

    def get_candidate_pairs(self) -> Set[Tuple[int, int]]:
        """
        获取需要进行Z3验证的候选对
        只有同一个bucket内的provenances才需要验证
        """
        pairs = set()
        for indices in self.bucket_index.values():
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        pairs.add((min(indices[i], indices[j]),
                                   max(indices[i], indices[j])))
        return pairs


def lsh_cluster(
    provenances: List[Provenance],
    num_bands: int = 20,
    rows_per_band: int = 5,
    similarity_threshold: float = 0.5,
    max_clusters: Optional[int] = None,
    max_cluster_size: int = 8
) -> LSHResult:
    """
    使用LSH对推荐意见进行聚类

    Args:
        provenances: 推荐意见列表
        num_bands: 带数量
        rows_per_band: 每带行数
        similarity_threshold: 相似度阈值
        max_clusters: 最大聚类数（None 则动态计算：len(provenances) // 5，至少 3）
        max_cluster_size: 单个聚类最大大小（防止输入过长导致 LLM 截断）

    Returns:
        LSHResult: 包含聚类结果和桶索引，可用于后续Z3验证限制范围
    """
    if not provenances:
        return LSHResult(clusters=[], bucket_index={})

    # 计算每个provenance的MinHash签名
    signatures = []
    for p in provenances:
        shingles = compute_shingles(p.quote)
        sig = compute_minhash(shingles, num_bands * rows_per_band)
        signatures.append(sig)

    # LSH: 将签名分成多个带，每个带作为桶的键
    buckets = {}
    for idx, sig in enumerate(signatures):
        for band in range(num_bands):
            start = band * rows_per_band
            end = start + rows_per_band
            band_sig = tuple(sig[start:end])
            key = (band, band_sig)
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(idx)

    # 找到候选对
    candidate_pairs = set()
    for indices in buckets.values():
        if len(indices) > 1:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    candidate_pairs.add((min(indices[i], indices[j]),
                                        max(indices[i], indices[j])))

    # 使用并查集进行聚类
    parent = list(range(len(provenances)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # 合并相似的候选对
    for i, j in candidate_pairs:
        # 计算Jaccard相似度
        sig_i, sig_j = signatures[i], signatures[j]
        similarity = sum(1 for a, b in zip(sig_i, sig_j) if a == b) / len(sig_i)
        if similarity >= similarity_threshold:
            union(i, j)

    # 构建聚类
    cluster_map = {}
    for idx in range(len(provenances)):
        root = find(idx)
        if root not in cluster_map:
            cluster_map[root] = []
        cluster_map[root].append(idx)

    # 动态计算 max_clusters
    if max_clusters is None:
        # 每 5 条推荐大约分 1 个聚类，至少 3 个，最多 10 个
        max_clusters = max(3, min(10, len(provenances) // 5))

    # 收集所有 provenance 到初始聚类列表（保留原始索引）
    initial_clusters = []  # List[List[Tuple[int, Provenance]]] - (原始索引, provenance)
    for _, indices in cluster_map.items():
        cluster_items = [(i, provenances[i]) for i in indices]
        initial_clusters.append(cluster_items)

    # 按大小排序（大的在前，方便后续处理）
    initial_clusters.sort(key=len, reverse=True)

    # 重新分配：确保每个聚类不超过 max_cluster_size，总聚类数不超过 max_clusters
    final_clusters = []  # List[List[Tuple[int, Provenance]]]
    overflow = []  # 存放需要重新分配的 (index, provenance)

    # 第一步：拆分过大的聚类
    for cluster in initial_clusters:
        if len(cluster) > max_cluster_size:
            # 拆分成多个小聚类
            for i in range(0, len(cluster), max_cluster_size):
                chunk = cluster[i:i + max_cluster_size]
                if len(chunk) >= 2:  # 至少 2 条才成为独立聚类
                    final_clusters.append(chunk)
                else:
                    overflow.extend(chunk)
        else:
            final_clusters.append(cluster)

    # 第二步：合并小聚类直到聚类数 <= max_clusters
    while len(final_clusters) > max_clusters:
        # 找到最小的两个聚类合并
        final_clusters.sort(key=len)
        smallest = final_clusters.pop(0)
        second_smallest = final_clusters.pop(0)
        merged = smallest + second_smallest

        # 如果合并后超过 max_cluster_size，放回去（不合并）
        if len(merged) > max_cluster_size:
            final_clusters.append(smallest)
            final_clusters.append(second_smallest)
            break  # 无法继续合并
        else:
            final_clusters.append(merged)

    # 第三步：处理 overflow（加入最小的聚类）
    for item in overflow:
        if final_clusters:
            final_clusters.sort(key=len)
            final_clusters[0].append(item)
        else:
            final_clusters.append([item])

    # 创建最终聚类结果，同时设置bucket_id并构建桶索引
    clusters = []
    bucket_index: Dict[int, List[int]] = {}

    for cluster_id, cluster_items in enumerate(final_clusters):
        # 记录桶索引：cluster_id -> 原始provenance索引列表
        original_indices = [idx for idx, _ in cluster_items]
        bucket_index[cluster_id] = original_indices

        # 给每个provenance设置bucket_id
        cluster_provenances = []
        for orig_idx, p in cluster_items:
            # 创建带bucket_id的新Provenance
            p_with_bucket = Provenance(
                source=p.source,
                quote=p.quote,
                recommendation_grade=p.recommendation_grade,
                evidence_level=p.evidence_level,
                bucket_id=cluster_id
            )
            cluster_provenances.append(p_with_bucket)

        texts_formatted = [f"Quote: {p.quote}" for p in cluster_provenances]
        clusters.append(ProvenanceCluster(
            cluster_id=cluster_id,
            provenances=cluster_provenances,
            texts_formatted=texts_formatted
        ))

    return LSHResult(clusters=clusters, bucket_index=bucket_index)
