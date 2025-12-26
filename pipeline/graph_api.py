"""
Pipeline interfaces built on top of core nodes/graphs.
使用 LangGraph fan-out/fan-in 模式，复用 reducer。
"""
import asyncio
from typing import List, Optional, Dict, Any

from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send

from .models import AgentState, Provenance, ProvenanceCluster, ClinicalRule, Term, MedicationTerm, Predicates, _to_models


from .lsh_cluster import lsh_cluster
from .io_utils import (
    save_provenances,
    load_provenances,
    save_clusters,
    load_clusters,
    get_failed_task_logger,
    save_cluster_cache,
    load_cluster_cache,
)
from .graph import merge_by_id



# ============ 抽取推荐意见 ============

def extract_provenances_stage(
    texts: List[str],
    save_to_file: bool = True,
    filepath: Optional[str] = None,
    max_concurrency: int = 5,
) -> List[Provenance]:
    print(f"[第一阶段] 开始抽取 {len(texts)} 个文本的推荐意见...")

    # 内部导入避免循环导入
    from .graph import extract_recommendation, distribute_texts

    builder = StateGraph(AgentState)
    builder.add_node("extract_recommendation", extract_recommendation)
    builder.add_conditional_edges(START, distribute_texts, ["extract_recommendation"])
    builder.add_edge("extract_recommendation", END)
    graph = builder.compile()

    input_state: AgentState = {
        "input_texts": texts,
    }

    result = graph.invoke(input_state, config={"max_concurrency": max_concurrency})
    provenances = result.get("provenance_buffer", [])

    if save_to_file and provenances:
        save_provenances(provenances, filepath)

    print(f"[第一阶段] 完成，共抽取 {len(provenances)} 条推荐意见")
    return provenances


# ============ LSH 聚类 ============

def cluster_provenances_stage(
    provenances: Optional[List[Provenance]] = None,
    load_from_file: bool = True,
    load_filepath: Optional[str] = None,
    save_to_file: bool = True,
    filepath: Optional[str] = None,
    **lsh_kwargs,
) -> tuple[List[ProvenanceCluster], Dict[int, List[int]]]:
    if provenances is None and load_from_file:
        provenances = load_provenances(load_filepath)

    if not provenances:
        print("[第二阶段] 无推荐意见数据，跳过聚类")
        return [], {}

    print(f"[第二阶段] 开始对 {len(provenances)} 条推荐意见进行LSH聚类...")
    lsh_result = lsh_cluster(provenances, **lsh_kwargs)
    print(f"[第二阶段] 聚类完成：{len(provenances)} 条推荐 -> {len(lsh_result.clusters)} 个聚类")

    if save_to_file:
        save_clusters(lsh_result.clusters, lsh_result.bucket_index, filepath)

    return lsh_result.clusters, lsh_result.bucket_index


def _prepare_cluster_state(
    cluster: ProvenanceCluster,
    cache_entry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """组合 cluster 原始数据与缓存结果，供 worker 复用。"""
    cache_entry = cache_entry or {}
    return {
        "cluster_id": cluster.cluster_id,
        "provenances": getattr(cluster, "provenances", []),
        "texts_formatted": getattr(cluster, "texts_formatted", []),
        "terms": _to_models(cache_entry.get("terms", []), Term),
        "med_terms": _to_models(cache_entry.get("med_terms", []), MedicationTerm),
        "predicates": _to_models(cache_entry.get("predicates", []), Predicates),
        "rules": _to_models(cache_entry.get("rules", []), ClinicalRule),
    }


def _merge_cluster_result(
    base: Dict[str, Any],
    new: Dict[str, Any],
) -> Dict[str, Any]:
    """按 id 合并 cluster 结果，返回更新后的 entry。"""
    return {
        "cluster_id": base.get("cluster_id"),
        "provenances": base.get("provenances", []),
        "texts_formatted": base.get("texts_formatted", []),
        "terms": merge_by_id(base.get("terms", []), new.get("terms", [])),
        "med_terms": merge_by_id(base.get("med_terms", []), new.get("med_terms", [])),
        "predicates": merge_by_id(base.get("predicates", []), new.get("predicates", [])),
        "rules": merge_by_id(base.get("rules", []), new.get("rules", [])),
    }


def save_results_from_cache(
    cluster_cache_path: Optional[str] = None,
    gen_dir: Optional[str] = None,
    save_provenances: bool = False,
) -> None:
    """
    从磁盘上的 cluster_cache.json 聚合并分别保存 `terms`, `med_terms`, `predicates`, `rules` 到 gen 目录。

    参数:
      - cluster_cache_path: 若提供，直接从该路径加载 cluster_cache 文件并使用其内容。
      - gen_dir: 若未提供 cluster_cache_path，则优先尝试读取 `gen_dir/cluster_cache.json`。
      - 否则回退到项目根 `gen/cluster_cache.json`。
    """
    import os

    cluster_cache: Optional[Dict[int, Dict[str, Any]]] = None

    # 1) 如果显式给出路径则优先使用
    if cluster_cache_path:
        try:
            cluster_cache = load_cluster_cache(cluster_cache_path)
        except Exception:
            cluster_cache = None

    # 2) 否则尝试 gen_dir 下的 cluster_cache.json
    if cluster_cache is None and gen_dir:
        candidate = os.path.join(gen_dir, "cluster_cache.json")
        if os.path.exists(candidate):
            try:
                cluster_cache = load_cluster_cache(candidate)
            except Exception:
                cluster_cache = None

    # 3) 回退到项目根的 gen/cluster_cache.json
    if cluster_cache is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        fallback = os.path.join(repo_root, "gen", "cluster_cache.json")
        if os.path.exists(fallback):
            try:
                cluster_cache = load_cluster_cache(fallback)
            except Exception:
                cluster_cache = None

    if not cluster_cache:
        # 没有可用的 cache，直接返回
        return

    aggregated = {"terms": [], "med_terms": [], "predicates": [], "rules": []}
    for entry in (cluster_cache or {}).values():
        # cluster_cache entries are plain dicts; convert to model instances before merging
        aggregated["terms"] = merge_by_id(aggregated["terms"], _to_models(entry.get("terms", []), Term))
        aggregated["med_terms"] = merge_by_id(aggregated["med_terms"], _to_models(entry.get("med_terms", []), MedicationTerm))
        aggregated["predicates"] = merge_by_id(aggregated["predicates"], _to_models(entry.get("predicates", []), Predicates))
        aggregated["rules"] = merge_by_id(aggregated["rules"], _to_models(entry.get("rules", []), ClinicalRule))

    # 构造 AgentState 以复用 save_to_gen 的保存逻辑
    agent_state = AgentState(
        messages=[],
        provenance_buffer=[],
        clusters=[],
        **aggregated,
    )
    from .io_utils import save_to_gen

    save_to_gen(agent_state, gen_dir=gen_dir, save_provenances=save_provenances)

    # 创建 cluster_final.json：将 cluster.json 和 cluster_cache.json 按 cluster_id 合并
    _create_cluster_final(cluster_cache, gen_dir)



async def _fanout_clusters_async(
    clusters: List[ProvenanceCluster],
    worker_fn,
    max_concurrency: int = 5,
    cluster_cache: Optional[Dict[int, Dict[str, Any]]] = None,
) -> tuple[Dict[str, List[Any]], Dict[int, Dict[str, Any]]]:
    """
    异步版本的 fan-out/fan-in，使用 asyncio + to_thread 以便不阻塞事件循环。
    """
    cache: Dict[int, Dict[str, Any]] = {**(cluster_cache or {})}
    global_result = {"terms": [], "med_terms": [], "predicates": [], "rules": []}
    sem = asyncio.Semaphore(max_concurrency)

    async def _run_one(cluster: ProvenanceCluster):
        nonlocal cache
        state = _prepare_cluster_state(cluster, cache.get(cluster.cluster_id))
        try:
            async with sem:
                # Run synchronous worker function in thread without rate limiting
                output = await asyncio.to_thread(worker_fn, state)
            merged_entry = _merge_cluster_result(state, output or {})
            cache[cluster.cluster_id] = merged_entry
            return merged_entry
        except Exception as e:
            print(f"[fanout_async][cluster {cluster.cluster_id}] error: {e}")
            get_failed_task_logger().log_failed_cluster(
                cluster_id=cluster.cluster_id,
                texts=getattr(cluster, "provenances", []),
                error=f"Fanout async execution error: {str(e)}"
            )
            return _prepare_cluster_state(cluster, cache.get(cluster.cluster_id))

    tasks = [asyncio.create_task(_run_one(c)) for c in clusters]

    for coro in asyncio.as_completed(tasks):
        merged_entry = await coro
        global_result["terms"] = merge_by_id(global_result["terms"], merged_entry.get("terms", []))
        global_result["med_terms"] = merge_by_id(global_result["med_terms"], merged_entry.get("med_terms", []))
        global_result["predicates"] = merge_by_id(global_result["predicates"], merged_entry.get("predicates", []))
        global_result["rules"] = merge_by_id(global_result["rules"], merged_entry.get("rules", []))

    return global_result, cache


# ============ 聚类后抽取：术语 / 药物 ============

def extract_terms_stage(
    clusters: Optional[List[ProvenanceCluster]] = None,
    load_from_file: bool = True,
    load_filepath: Optional[str] = None,
    max_concurrency: int = 5,
    cluster_cache_path: Optional[str] = None,
    persist_cluster_cache: bool = True,
) -> Dict[str, List]:
    if clusters is None and load_from_file:
        clusters, _ = load_clusters(load_filepath)
    if not clusters:
        print("[terms_stage] 无聚类数据，跳过处理")
        return {"terms": [], "med_terms": []}

    # 使用 Send API 范式的图
    from .graph import build_terms_extraction_graph
    graph = build_terms_extraction_graph()

    cluster_cache: Dict[int, Dict[str, Any]] = {}
    if cluster_cache_path:
        cluster_cache = load_cluster_cache(cluster_cache_path)

    input_state = {
        "clusters": clusters,
        "cluster_cache": cluster_cache,
    }

    result = graph.invoke(input_state, config={"max_concurrency": max_concurrency})

    # 处理 cluster-specific 缓存更新（由 Send API 自动聚合）
    if persist_cluster_cache and cluster_cache_path:
        cache_updates = result.get("cluster_cache_updates", {})
        # 合并到现有缓存
        updated_cache = cluster_cache.copy()
        updated_cache.update(cache_updates)

        save_cluster_cache(updated_cache, cluster_cache_path)



# ============ 聚类后抽取：谓词 ============

def extract_predicates_stage(
    clusters: Optional[List[ProvenanceCluster]] = None,
    load_from_file: bool = True,
    load_filepath: Optional[str] = None,
    max_concurrency: int = 5,
    cluster_cache_path: Optional[str] = None,
    persist_cluster_cache: bool = True,
) -> Dict[str, List]:
    if clusters is None and load_from_file:
        clusters, _ = load_clusters(load_filepath)
    if not clusters:
        print("[predicates_stage] 无聚类数据，跳过处理")
        return {"predicates": [], "terms": [], "med_terms": []}

    # 使用 Send API 范式的图
    from .graph import build_predicates_extraction_graph
    graph = build_predicates_extraction_graph()

    cluster_cache: Dict[int, Dict[str, Any]] = {}
    if cluster_cache_path:
        cluster_cache = load_cluster_cache(cluster_cache_path)

    input_state = {
        "clusters": clusters,
        "cluster_cache": cluster_cache,
    }

    result = graph.invoke(input_state, config={"max_concurrency": max_concurrency})

    # 处理 cluster-specific 缓存更新（由 Send API 自动聚合）
    if persist_cluster_cache and cluster_cache_path:
        cache_updates = result.get("cluster_cache_updates", {})
        # 合并到现有缓存
        updated_cache = cluster_cache.copy()
        updated_cache.update(cache_updates)

        save_cluster_cache(updated_cache, cluster_cache_path)



# ============ 聚类后抽取：规则 ============

def extract_rules_stage(
    clusters: Optional[List[ProvenanceCluster]] = None,
    load_from_file: bool = True,
    load_filepath: Optional[str] = None,
    max_concurrency: int = 5,
    cluster_cache_path: Optional[str] = None,
    persist_cluster_cache: bool = True,
) -> Dict[str, List]:
    if clusters is None and load_from_file:
        clusters, _ = load_clusters(load_filepath)
    if not clusters:
        print("[rules_stage] 无聚类数据，跳过处理")
        return {"rules": [], "terms": [], "med_terms": [], "predicates": []}

    # 使用 Send API 范式的图
    from .graph import build_rules_extraction_graph
    graph = build_rules_extraction_graph()

    cluster_cache: Dict[int, Dict[str, Any]] = {}
    if cluster_cache_path:
        cluster_cache = load_cluster_cache(cluster_cache_path)

    input_state = {
        "clusters": clusters,
        "cluster_cache": cluster_cache,
    }

    result = graph.invoke(input_state, config={"max_concurrency": max_concurrency})

    # 处理 cluster-specific 缓存更新（由 Send API 自动聚合）
    if persist_cluster_cache and cluster_cache_path:
        cache_updates = result.get("cluster_cache_updates", {})
        # 合并到现有缓存
        updated_cache = cluster_cache.copy()
        updated_cache.update(cache_updates)

        save_cluster_cache(updated_cache, cluster_cache_path)



async def process_clusters_stage_async(
    clusters: Optional[List[ProvenanceCluster]] = None,
    load_from_file: bool = True,
    load_filepath: Optional[str] = None,
    cluster_cache_path: Optional[str] = None,
    persist_cluster_cache: bool = True,
    verbose: bool = True,
) -> Dict[str, List]:
    """
    异步版本的第三阶段处理，便于在 Notebook 中看到进度并不中断事件循环。
    """
    if clusters is None and load_from_file:
        clusters, _ = load_clusters(load_filepath)
    if not clusters:
        print("[第三阶段-async] 无聚类数据，跳过处理")
        return {}

    cluster_cache: Dict[int, Dict[str, Any]] = {}
    if cluster_cache_path:
        cluster_cache = load_cluster_cache(cluster_cache_path)

    from .graph import CLUSTER_SUBGRAPH

    def worker(state):
        return CLUSTER_SUBGRAPH.invoke(state)

    total = len(clusters)
    completed = 0

    async def _progress_worker(cluster: ProvenanceCluster):
        nonlocal completed
        res = await _fanout_clusters_async(
            [cluster],
            worker,
            max_concurrency=1,
            cluster_cache=cluster_cache,
        )
        completed += 1
        if verbose:
            print(f"[第三阶段-async] cluster {cluster.cluster_id} 完成 ({completed}/{total})")
        return res

    tasks = [_progress_worker(c) for c in clusters]
    global_result = {"terms": [], "med_terms": [], "predicates": [], "rules": []}

    for coro in asyncio.as_completed(tasks):
        result_state, returned_cache = await coro
        # 合并返回的 cache 到主 cluster_cache 中，而不是覆盖
        cluster_cache.update(returned_cache)
        global_result["terms"] = merge_by_id(global_result["terms"], result_state.get("terms", []))
        global_result["med_terms"] = merge_by_id(global_result["med_terms"], result_state.get("med_terms", []))
        global_result["predicates"] = merge_by_id(global_result["predicates"], result_state.get("predicates", []))
        global_result["rules"] = merge_by_id(global_result["rules"], result_state.get("rules", []))

    if persist_cluster_cache and cluster_cache_path:
        save_cluster_cache(cluster_cache, cluster_cache_path)

    return global_result


# ============ 临床流水线封装 ============

class ClinicalGuidelinePipeline:
    """
    临床指南解析流水线（同步）
    """

    def __init__(self):
        from .graph import build_pipeline_graph
        self._graph = build_pipeline_graph()

    @property
    def graph(self):
        return self._graph

    def _build_input_state(self, texts: List[str]) -> Dict:
        return {
            "input_texts": texts,
            "messages": [],
            "provenance_buffer": [],
            "clusters": [],
            "terms": [],
            "med_terms": [],
            "predicates": [],
            "rules": [],
            "lsh_bucket_index": {},
            "provenances": [],
            "texts_formatted": [],
        }

    @staticmethod
    def _normalize_input(texts) -> List[str]:
        if isinstance(texts, str):
            return [texts]
        return list(texts)

    def run(self, texts, max_concurrency: int = 5, clear_failed_log: bool = True) -> Dict:
        texts = self._normalize_input(texts)
        input_state = self._build_input_state(texts)
        config = {"max_concurrency": max_concurrency}

        if clear_failed_log:
            get_failed_task_logger().reset()

        print(f"[run] 处理 {len(texts)} 个文本 (max_concurrency={max_concurrency})...")
        result = self._graph.invoke(input_state, config=config)

        failed_count = get_failed_task_logger().get_failed_count()
        if failed_count > 0:
            print(f"\n[WARNING] {failed_count} 个任务失败，已保存到 gen/failed_tasks.json")

        return {
            "terms": result.get("terms", []),
            "med_terms": result.get("med_terms", []),
            "predicates": result.get("predicates", []),
            "rules": result.get("rules", []),
            "provenance_buffer": result.get("provenance_buffer", []),
        }

    # 分阶段接口
    def extract_provenances(self, texts, **kwargs) -> List[Provenance]:
        texts = self._normalize_input(texts)
        return extract_provenances_stage(texts, **kwargs)

    def cluster_provenances(self, **kwargs) -> tuple[List[ProvenanceCluster], Dict[int, List[int]]]:
        return cluster_provenances_stage(**kwargs)

    def extract_terms_stage(self, **kwargs) -> Dict[str, List]:
        return extract_terms_stage(**kwargs)

    def extract_predicates_stage(self, **kwargs) -> Dict[str, List]:
        return extract_predicates_stage(**kwargs)

    def extract_rules_stage(self, **kwargs) -> Dict[str, List]:
        return extract_rules_stage(**kwargs)

    async def process_clusters_async(self, **kwargs) -> Dict[str, List]:
        return await process_clusters_stage_async(**kwargs)


def _create_cluster_final(cluster_cache: Dict[int, Dict[str, Any]], gen_dir: Optional[str] = None):
    """
    创建 cluster_final.json：将 cluster.json 和 cluster_cache.json 按 cluster_id 合并

    Args:
        cluster_cache: 从 cluster_cache.json 加载的缓存数据
        gen_dir: 输出目录，默认为 'gen'
    """
    import os
    import json

    if gen_dir is None:
        gen_dir = "gen"

    # 确保输出目录存在
    os.makedirs(gen_dir, exist_ok=True)

    # 读取 cluster.json
    cluster_file_path = os.path.join(gen_dir, "cluster.json")
    if not os.path.exists(cluster_file_path):
        print(f"警告: {cluster_file_path} 不存在，跳过 cluster_final.json 创建")
        return

    try:
        with open(cluster_file_path, 'r', encoding='utf-8') as f:
            cluster_data = json.load(f)
    except Exception as e:
        print(f"错误: 读取 {cluster_file_path} 失败: {e}")
        return

    clusters = cluster_data.get("clusters", [])
    if not clusters:
        print("警告: cluster.json 中没有 clusters 数据")
        return

    # 创建 cluster_id 到 cluster 数据的映射
    cluster_map = {cluster["cluster_id"]: cluster for cluster in clusters}

    # 合并数据
    final_clusters = []
    for cluster_id, cache_entry in cluster_cache.items():
        if cluster_id in cluster_map:
            # 合并原始 cluster 数据和缓存数据
            final_cluster = {
                **cluster_map[cluster_id],  # 原始数据（包含 provenances）
                **cache_entry,  # 处理后的数据（terms, med_terms, predicates, rules）
            }
            final_clusters.append(final_cluster)
        else:
            print(f"警告: cluster_id {cluster_id} 在 cluster.json 中不存在")
            # 仍然包含只有缓存数据的条目
            final_cluster = {
                "cluster_id": cluster_id,
                **cache_entry,
            }
            final_clusters.append(final_cluster)

    # 按 cluster_id 排序
    final_clusters.sort(key=lambda x: x["cluster_id"])

    # 保存 cluster_final.json
    final_data = {"clusters": final_clusters}
    final_file_path = os.path.join(gen_dir, "cluster_final.json")

    try:
        with open(final_file_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 已创建 {final_file_path}，包含 {len(final_clusters)} 个 cluster")
    except Exception as e:
        print(f"错误: 保存 {final_file_path} 失败: {e}")


# ============ 便捷函数 ============

def create_pipeline(
    api_key: Optional[str] = None,
    base_url: str = "https://openrouter.ai/api/v1",
    model: str = "deepseek/deepseek-v3.1-terminus",
) -> ClinicalGuidelinePipeline:
    if api_key:
        from .config import set_config, PipelineConfig, LLMConfig
        from .graph import _get_default_llm

        llm_config = LLMConfig(api_key=api_key, base_url=base_url, model=model)
        set_config(PipelineConfig(llm=llm_config))
        _get_default_llm.cache_clear()

    return ClinicalGuidelinePipeline()

