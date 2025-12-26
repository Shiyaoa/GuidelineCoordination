"""
Clinical Knowledge Pipeline - 临床指南知识抽取流水线

一个模块化的临床指南解析工具包，用于从临床指南文本中抽取结构化知识。

主要组件:
- models: 数据模型定义
- config: 配置管理
- standard_library: 标准术语库管理
- processors: Fuzzy匹配处理器
- graph: LangGraph 两阶段 Map-Reduce 工作流
- io_utils: 输入输出工具
- z3env: Z3逻辑推理环境

快速开始:
```python
from pipeline import create_pipeline, save_to_gen

# 创建流水线
pipeline = create_pipeline(
    api_key="your-api-key",
    model="deepseek/deepseek-v3.1-terminus"
)

# 运行（单文本或多文本）
result = pipeline.run("单个文本")
result = pipeline.run(["文本1", "文本2", "文本3"])

# 异步运行
result = await pipeline.run_async(["文本1", "文本2"])

# 保存结果
save_to_gen(result)
```
"""

# 版本信息
__version__ = "1.0.0"

# 核心数据模型
from .models import (
    # 枚举
    TermLabel,
    Permission,
    # 核心模型
    Term,
    MedicationTerm,
    Predicates,
    Action,
    Provenance,
    ClinicalRule,
    ProvenanceCluster,
    # 列表包装类
    TermList,
    MedicationTermList,
    PredicatesList,
    ProvenanceList,
    ClinicalRuleList,
    # 状态
    AgentState,
    ClusterState,
)

# 配置
from .config import (
    LLMConfig,
    PathConfig,
    MatchConfig,
    PipelineConfig,
    get_config,
    set_config,
)

# 标准库
from .standard_library import (
    StandardLibrary,
    get_standard_library,
)

# 处理器
from .processors import (
    TermProcessor,
    MedicationProcessor,
    PredicateProcessor,
    process_terms,
    process_med_terms,
    process_predicates,
)

# 图/流水线（底层节点与图）
from .graph import (
    # 节点函数
    extract_recommendation,
    # 图构建
    build_extraction_subgraph,
    build_pipeline_graph,
)

# 接口与组合（基于 fan-out/fan-in 的封装）
from .graph_api import (
    ClinicalGuidelinePipeline,
    create_pipeline,
    save_results_from_cache
)

# LSH聚类
from .lsh_cluster import (
    lsh_cluster,
    LSHResult,
)

# IO工具
from .io_utils import (
    save_to_gen,
    load_from_gen,
    export_state_to_json,
    update_standard_library,
    FailedTaskLogger,
    get_failed_task_logger,
)

# Z3环境
from .z3env import Z3Env


# Z3 分析
from .analyzer import (
    # 枚举
    PredicateRelation,
    ActionRelation,
    RuleRelation,
    # 数据结构
    PredicatePairResult,
    ActionPairResult,
    RulePairResult,
    PredicateLattice,
    ConflictReport,
    # 分析器
    PredicateAnalyzer,
    ActionAnalyzer,
    RuleAnalyzer,
    ConflictResolver,
    # 便捷函数
    analyze_knowledge_base,
)


# 定义公开API
__all__ = [
    # 版本
    "__version__",
    
    # 模型
    "TermLabel",
    "Permission",
    "Term",
    "MedicationTerm",
    "Predicates",
    "Action",
    "Provenance",
    "ClinicalRule",
    "ProvenanceCluster",
    "TermList",
    "MedicationTermList",
    "PredicatesList",
    "ProvenanceList",
    "ClinicalRuleList",
    "AgentState",
    "ClusterState",
    
    # 配置
    "LLMConfig",
    "PathConfig",
    "MatchConfig",
    "PipelineConfig",
    "get_config",
    "set_config",
    
    # 标准库
    "StandardLibrary",
    "get_standard_library",
    
    # 处理器
    "TermProcessor",
    "MedicationProcessor",
    "PredicateProcessor",
    "process_terms",
    "process_med_terms",
    "process_predicates",
    
    # 节点函数
    "extract_recommendation",
    
    # LSH聚类
    "lsh_cluster",
    "LSHResult",
    
    # 图/流水线
    "build_extraction_subgraph",
    "build_pipeline_graph",
    
    # 接口/阶段
    "ClinicalGuidelinePipeline",
    "create_pipeline",
    "extract_provenances_stage",
    "cluster_provenances_stage",
    "extract_terms_stage",
    "extract_predicates_stage",
    "extract_rules_stage",
    "process_clusters_stage",
    
    # 失败任务管理
    "FailedTaskLogger",
    "get_failed_task_logger",
    
    # IO
    "save_to_gen",
    "load_from_gen",
    "export_state_to_json",
    "update_standard_library",
    
    # Z3
    "Z3Env",
    
    # 数据清洗
    "normalize_permission",
    "normalize_action",
    "normalize_rule",
    "clean_rules",
    "clean_predicates",
    "validate_rule_references",
    "run_full_clean",
    "PERMISSION_MAPPING",
    "VALID_PERMISSIONS",
    
    # Z3 分析
    "PredicateRelation",
    "ActionRelation",
    "RuleRelation",
    "PredicatePairResult",
    "ActionPairResult",
    "RulePairResult",
    "PredicateLattice",
    "ConflictReport",
    "PredicateAnalyzer",
    "ActionAnalyzer",
    "RuleAnalyzer",
    "ConflictResolver",
    "analyze_knowledge_base",
]

