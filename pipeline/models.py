"""
数据模型定义 - 包含所有Pydantic模型和AgentState
"""
from enum import Enum
from typing import Annotated, Sequence, TypedDict, List, Optional, TypeVar, Dict, Literal, Any, Union
import operator
from pydantic import BaseModel, Field, root_validator
# LangGraph 相关导入（延迟导入以避免依赖问题）
try:
    from langchain_core.messages import AnyMessage
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    AnyMessage = None
    add_messages = None


# ============ 自定义 Reducer（带去重） ============

T = TypeVar('T')

def merge_by_id(existing: List[T], new: List[T]) -> List[T]:
    """
    自定义 reducer：合并列表并按 id 去重
    后来的项会覆盖先前的同 id 项
    """
    merged = {item.id: item for item in existing}
    for item in new:
        merged[item.id] = item
    return list(merged.values())

def _to_models(items: Any, model_cls):
    """确保列表元素为指定 Pydantic 模型实例，便于 merge_by_id 去重。"""
    if not items:
        return []
    converted = []
    for item in items:
        if isinstance(item, model_cls):
            converted.append(item)
        elif isinstance(item, dict):
            try:
                converted.append(model_cls.model_validate(item))
            except Exception:
                continue
        else:
            continue
    return converted

def merge_cluster_cache_updates(existing: Dict[int, Dict[str, Any]], new: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    自定义 reducer：合并 cluster 缓存更新
    后来的更新会覆盖或合并到先前的缓存中
    """
    merged = existing.copy()
    for cluster_id, updates in new.items():
        if cluster_id not in merged:
            merged[cluster_id] = {}
        # 合并该 cluster 的缓存更新
        merged[cluster_id].update(updates)
    return merged


# ============ 枚举类型 ============

class TermLabel(str, Enum):
    """术语分类标签"""
    MEASURES = "measures"
    CONDITIONS = "conditions"
    PROCEDURES = "procedures"
    OBSERVATIONS = "observations"


class Permission(str, Enum):
    """规则许可类型 - 有明确偏序关系"""
    # 使用类（是否使用）
    ALLOW = "allow"              # 允许使用（可选方案）
    RECOMMEND = "recommend"      # 推荐使用（优先选择）
    REQUIRE = "require"          # 必须使用（强制要求）
    CAUTION = "caution"          # 慎用（需监测）
    AVOID = "avoid"              # 避免使用（有更好替代）
    CONTRAINDICATE = "contraindicate"  # 禁忌（绝对禁止）
    CONTINUE = "continue"        # 继续使用
    STOP = "stop"                # 停止使用
    CONSIDER = "consider"        # 考虑使用
    # 剂量调整类
    REDUCE_DOSE = "reduce_dose"        # 减量（如肾功能下降时）
    INCREASE_DOSE = "increase_dose"    # 加量（如血糖控制不佳时）
    START_LOW_DOSE = "start_low_dose"  # 起始低剂量（逐步滴定）
    MAX_DOSE_LIMIT = "max_dose_limit"  # 限制最大剂量
    TITRATE = "titrate"                # 滴定调整（根据疗效/耐受）
    
    @classmethod
    def usage_permissions(cls) -> list:
        """返回使用类许可"""
        return [cls.CONTRAINDICATE, cls.AVOID, cls.CAUTION, cls.ALLOW, cls.RECOMMEND, cls.REQUIRE]
    
    @classmethod
    def dose_permissions(cls) -> list:
        """返回剂量调整类许可"""
        return [cls.REDUCE_DOSE, cls.INCREASE_DOSE, cls.START_LOW_DOSE, cls.MAX_DOSE_LIMIT, cls.TITRATE]
    
    @classmethod
    def priority_order(cls) -> list:
        """返回使用类优先级顺序（禁忌最高）"""
        return cls.usage_permissions()
    
    @classmethod
    def dose_priority_order(cls) -> list:
        """返回剂量类优先级顺序（减量最高）"""
        return [cls.REDUCE_DOSE, cls.MAX_DOSE_LIMIT, cls.START_LOW_DOSE, cls.TITRATE, cls.INCREASE_DOSE]
    
    def priority(self) -> int:
        """返回优先级数值（越高越限制）"""
        order = self.priority_order()
        return order.index(self) if self in order else -1
    
    def is_restrictive(self) -> bool:
        """是否为限制性许可"""
        return self in [Permission.CONTRAINDICATE, Permission.AVOID, Permission.CAUTION]
    
    def is_permissive(self) -> bool:
        """是否为许可性许可"""
        return self in [Permission.ALLOW, Permission.RECOMMEND, Permission.REQUIRE]
    
    def is_dose_adjustment(self) -> bool:
        """是否为剂量调整类许可"""
        return self in self.dose_permissions()


# ============ 核心数据模型 ============

class Term(BaseModel):
    """非药物术语"""
    id: str
    name: str
    label: TermLabel = Field(..., description="术语分类")
    type: str
    unit: Optional[str] = None


class MedicationTerm(BaseModel):
    """药物术语"""
    id: str
    name: str
    drug_class: Optional[str] = Field(
        None,
        alias="class",
        description="所属的药物大类ID。如果是药物类别本身，则为 null 或缺失。"
    )
    subclass: Optional[str] = Field(
        None,
        description="所属的药物子类ID。如果无子类，则为 null 或缺失。"
    )


class Predicates(BaseModel):
    """逻辑谓词"""
    id: str
    name: str
    formal_definition: str
    dependencies: Optional[List[str]] = None


class Action(BaseModel):
    """规则动作"""
    subjects: List[str] = Field(
        description="药物术语id列表，联合用药时包含多个"
    )
    permission: Permission = Field(
        description="许可类型：使用类(allow/recommend/require/caution/avoid/contraindicate) 或 剂量调整类(reduce_dose/increase_dose/start_low_dose/max_dose_limit/titrate)"
    )
    requirements: List[str] = Field(
        default=[],
        description="额外要求，如 prefer_first_line, monitor_renal_function"
    )


class Provenance(BaseModel):
    """来源信息"""
    source: str
    quote: str
    recommendation_grade: Optional[str] = None
    evidence_level: Optional[str] = None
    bucket_id: Optional[int] = Field(
        None,
        description="LSH桶ID，用于后续Z3验证时限制验证范围（同桶内才验证）"
    )


class ClinicalRule(BaseModel):
    """临床规则"""
    id: str
    label: str
    condition: Optional[str] = Field(
        None,
        description="引用谓词id的布尔运算式"
    )
    action: Action
    provenance: List[Provenance] = []


# ============ 聚类相关模型 ============

class ProvenanceCluster(BaseModel):
    """推荐意见聚类"""
    cluster_id: int
    provenances: List[Provenance]
    texts_formatted: List[str] = []


# ============ 列表包装类（用于LLM结构化输出） ============

class ProvenanceList(BaseModel):
    """包装类,用于接收多个 Provenance 对象"""
    items: List[Provenance] = Field(description="Provenance 对象列表")


class PredicatesList(BaseModel):
    """包装类,用于接收多个 Predicates 对象"""
    items: List[Predicates] = Field(description="Predicates 对象列表")


class TermList(BaseModel):
    """包装类,用于接收多个 Term 对象"""
    items: List[Term] = Field(description="Term 对象列表")


class MedicationTermList(BaseModel):
    """包装类,用于接收多个 MedicationTerm 对象"""
    items: List[MedicationTerm] = Field(description="MedicationTerm 对象列表")


class ClinicalRuleList(BaseModel):
    """包装类,用于接收多个 ClinicalRule 对象"""
    items: List[ClinicalRule] = Field(description="ClinicalRule 对象列表")


# ============ AgentState 定义 ============


class AgentState(TypedDict):
    """Agent状态定义 - 支持两阶段 map-reduce"""
    # 批量输入文本（用于第一阶段 map）
    input_texts: List[str]
    
    # 单条消息（用于单文本处理）
    messages: Annotated[Sequence[AnyMessage], add_messages]
    
    # 推荐意见缓存（reducer: operator.add，保留所有推荐不去重）
    provenance_buffer: Annotated[List[Provenance], operator.add]
    
    # LSH候选对索引（用于Z3验证时限制范围）
    # key: bucket_id, value: 该桶内的provenance索引集合
    lsh_bucket_index: Dict[int, List[int]]
    
    # LSH聚类结果
    clusters: List[ProvenanceCluster]
    
    
    # 最终汇总结果（reducer: merge_by_id，自动按 id 去重）
    terms: Annotated[List[Term], merge_by_id]
    med_terms: Annotated[List[MedicationTerm], merge_by_id]
    predicates: Annotated[List[Predicates], merge_by_id]
    rules: Annotated[List[ClinicalRule], merge_by_id]

class ClusterState(TypedDict):
    """子图状态 - 用于处理单个聚类"""
    cluster_id: int
    provenances: List[Provenance]
    texts_formatted: List[str]
    terms: Annotated[List[Term], merge_by_id]
    med_terms: Annotated[List[MedicationTerm], merge_by_id]
    predicates: Annotated[List[Predicates], merge_by_id]
    rules: Annotated[List[ClinicalRule], merge_by_id]




class OperatorType(str, Enum):
    """
    Symbolic Operator Layer: F(E × O) → L
    
    Maps ontological entities (E) via operators (O) to verifiable logical expressions (L).
    This design bridges the semantic gap between static entities and logical predicates,
    enabling SMT-based reasoning over clinical guidelines.
    
    Operator Classes:
    1. Existential Operators: Map clinical concepts to Boolean satisfiability
    2. Arithmetic Operators: Map measurement entities to Real Arithmetic constraints (LRA)
    3. Categorical Operators: Map categorical entities to String/Int constraints
    """
    # Existential Operators (Bool sort → Propositional Logic)
    HAS = "Has"           # Existential: Disease/diagnosis presence -> Bool
    ON = "On"             # Existential: Medication usage status -> Bool
    HISTORY = "HistoryOf" # Existential: Historical event presence -> Bool
    
    ASSESS = "Assess"     # Existential: Subjective assessment -> Bool/String 
    
    # Categorical Operators (String/Int sort)
    RISK = "Risk"         # Categorical: Risk stratification -> String 
    STAGE = "Stage"       # Categorical: Staging/grading -> String
    
    # Arithmetic Operators (Real sort → Linear Real Arithmetic)
    VALUE = "Value"       # Arithmetic: Measurement value -> Real
    DURATION = "Duration" # Arithmetic: Time duration -> Real
    DELTA = "Delta"       # Arithmetic: Change/trend (increase or decrease) -> Real
    
# 2. 定义比较符枚举
class CompareOp(str, Enum):
    EQ = "=="
    GT = ">"
    LT = "<"
    GE = ">="
    LE = "<="
    NE = "!="

class AtomicCondition(BaseModel):
    """
    Atomic Clinical Condition: A single, independent logical predicate.
    
    Represents one application of the Symbolic Operator Layer: F(E, O) → L
    where E is the term_id (entity), O is the operator, and L is the resulting
    logical expression.
    
    Constraints:
    - NO AND/OR logic allowed (atomicity principle)
    - Each condition maps to a single SMT constraint
    - Operator type determines the Z3 sort (Bool/Real/String/Int)
    """
    # 移除 node_type，因为只有 atomic 一种可能
    # 移除 sub_conditions
    text: str = Field(..., description="text explanation for this condition")
    term_id: str = Field(..., description="Standard Term ID")
    operator: OperatorType = Field(..., description="Semantic Operator")
    comparison: Optional[Literal[">", ">=", "<", "<=", "==", "!="]] = Field(
        None, 
        description="Required for Value, Stage, Risk, DURATION, DELTA. Omit for Has, On, Assess, HISTORY."
    )
    target_value: Optional[Union[str, bool]] = Field(None, description="Threshold value or boolean assessment")
    assess_type: Optional[str] = Field(
        None,
        description="Required ONLY for 'Assess'. Describes the assessment type: 'intolerant'(bool), 'control'(str), 'symptomatic'(bool), etc."
    )

    @root_validator(pre=False, skip_on_failure=True)
    def validate_assess(cls, values):
        operator = values.get("operator")
        if operator == OperatorType.ASSESS:
            assess_type = values.get("assess_type")
            target_value = values.get("target_value")

            if not assess_type:
                # 容错：如果是 Assess 但没给类型，设置默认值
                values["assess_type"] = "status"

            # 根据 assess_type 验证 target_value 类型
            if assess_type in ["intolerant", "symptomatic"]:
                # 这些类型应该是布尔值
                if target_value is not None and not isinstance(target_value, bool):
                    # 尝试转换字符串到布尔
                    if isinstance(target_value, str):
                        values["target_value"] = target_value.lower() in ["true", "yes", "1", "intolerant", "symptomatic"]
                    else:
                        values["target_value"] = bool(target_value)
            elif assess_type == "control":
                # control 类型应该是字符串
                if target_value is not None and not isinstance(target_value, str):
                    values["target_value"] = str(target_value)
        else:
            # 容错：如果不是 Assess 算子，自动清空多余字段，避免校验失败
            values["assess_type"] = None
        return values
    
    
class PredicateExtractionBatch(BaseModel):
    # 强制 LLM 把所有条件拆散
    atoms: List[AtomicCondition] = Field(
        ..., 
        description="A flat list of ALL individual clinical criteria. If text says 'A or B', output [A, B]."
    )


class RuleAction(BaseModel):
    subjects: List[str] = Field(
        description="药物ID列表 (med.xxx)。联合用药包含多个ID，单一用药只包含一个。"
    )
    permission: Literal[
        "recommend", "require", "allow", "consider", "caution", "avoid", 
        "contraindicate", "continue", "stop", 
        "reduce_dose", "increase_dose", "start_low_dose", "max_dose_limit", "titrate"
    ] = Field(description="行动的许可类型或调整方向。")
    requirements: List[str] = Field(
        default_factory=list,
        description="伴随的非药物要求（如监测指标、状态评估、随访评估、转诊等）。"
    )

class SimplifiedRuleItem(BaseModel):
    id: str = Field(description="规则的唯一标识符，必须以 'rule.' 开头。")
    label: str = Field(description="规则的简短人类可读摘要。")
    condition: str = Field(description="逻辑表达式 (使用标准谓词ID)。")
    source_ids: str = Field(
        description="该规则对应的单个quote编号 (如 'q1')。一条规则只能对应一个quote，必须包含此字段用于provenance关联"
    )
    action: RuleAction

class SubmitSimplifiedRules(BaseModel):
    rules: List[SimplifiedRuleItem] = Field(description="提取的原子化规则列表。")