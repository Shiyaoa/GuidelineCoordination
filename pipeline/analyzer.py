# -*- coding: utf-8 -*-
"""
Z3 分析模块 - 谓词和规则的关系分析与冲突消解

核心功能：
1. PredicateAnalyzer: 分析谓词之间的逻辑关系（等价、蕴含、互斥、相交）
2. RuleAnalyzer: 分析规则之间的关系（条件关系 + 动作关系）
3. ConflictResolver: 生成冗余消解和冲突解决建议
"""
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from collections import defaultdict
import json
from pathlib import Path

from z3 import Solver, Not, And, Or, sat, unsat, is_bool

from .z3env import Z3Env
from .models import Permission, Predicates, ClinicalRule
from .lsh_cluster import compute_shingles, compute_minhash


# ============ 关系枚举 ============

class PredicateRelation(Enum):
    """谓词间的逻辑关系"""
    EQUIVALENT = "equivalent"       # P ⇔ Q：同真同假
    P_IMPLIES_Q = "p_implies_q"     # P ⇒ Q：P 蕴含 Q
    Q_IMPLIES_P = "q_implies_p"     # Q ⇒ P：Q 蕴含 P
    MUTEX = "mutex"                 # P ∧ Q 不可满足（互斥）
    INTERSECT = "intersect"         # 可满足但互不蕴含（相交）
    UNKNOWN = "unknown"             # 无法判定


class ActionRelation(Enum):
    """动作间的关系"""
    EQUIVALENT = "equivalent"       # 等价：subjects和permission都完全相同
    CONFLICT = "conflict"           # 冲突：有重叠的subjects，但permission矛盾（使用vs禁用，禁用vs剂量，加剂量vs减剂量）
    DISAGREEMENT = "disagreement"   # 分歧：有重叠的subjects，permission不矛盾但是不相同（使用类内，剂量类内）
    INDEPENDENT = "independent"     # 独立：没有重叠的subjects
    


class RuleRelation(Enum):
    """规则间的综合关系（基于修订后的Problem Formulation）"""
    # 冗余关系
    COMPLETE_REDUNDANCY = "complete_redundancy"      # 完全冗余：PredicateRelation.EQUIVALENT + ActionRelation.EQUIVALENT
    CONTAINED_REDUNDANCY = "contained_redundancy"    # 被包含冗余：PredicateRelation.P_IMPLIES_Q / PredicateRelation.Q_IMPLIES_P + ActionRelation.EQUIVALENT
    
    # 冲突关系
    INTRINSIC_CONFLICT = "intrinsic_conflict"        # 固有冲突：PredicateRelation.EQUIVALENT+ActionRelation.CONFLICT
    INTRINSIC_DISAGREEMENT = "intrinsic_disagreement" # 固有分歧：PredicateRelation.EQUIVALENT+ActionRelation.DISAGREEMENT
    
    IMPLICATION_CONFLICT = "implication_conflict"    # 蕴含冲突：PredicateRelation.P_IMPLIES_Q / PredicateRelation.Q_IMPLIES_P+ActionRelation.CONFLICT
    IMPLICATION_DISAGREEMENT = "implication_disagreement" # 蕴含分歧：PredicateRelation.P_IMPLIES_Q / PredicateRelation.Q_IMPLIES_P+ActionRelation.DISAGREEMENT
    
    # 特殊vs一般（不是冲突）
    # SPECIALIZATION_PRIORITY = "specialization_priority"  # 特殊vs一般：PredicateRelation.P_IMPLIES_Q / PredicateRelation.Q_IMPLIES_P+ActionRelation.CONFLICT
    # context conflict
    LOCAL_CONFLICT = "local_conflict"                # 局部冲突：PredicateRelation.INTERSECT+ActionRelation.CONFLICT

    # 其他关系（others）
    OTHERS = "others"                     # 不属于以上类型


# ============ 数据结构 ============

@dataclass
class PredicatePairResult:
    """谓词对分析结果"""
    pred_p: str
    pred_q: str
    relation: PredicateRelation
    details: str = ""


@dataclass 
class ActionPairResult:
    """动作对分析结果"""
    subjects_overlap: List[str]  # 重叠的药物
    relation: ActionRelation
    dominant_permission: Optional[str] = None  # 哪个 permission 占优
    details: str = ""


@dataclass
class RulePairResult:
    """规则对分析结果"""
    rule_a: str
    rule_b: str
    condition_relation: PredicateRelation
    action_relation: ActionRelation
    overall_relation: RuleRelation
    recommendation: str = ""  # 处理建议


@dataclass
class PredicateLattice:
    """谓词格（有向图表示蕴含关系）"""
    equivalence_classes: Dict[str, Set[str]] = field(default_factory=dict)  # 等价类
    implication_edges: List[Tuple[str, str]] = field(default_factory=list)  # P -> Q 边
    mutex_pairs: List[Tuple[str, str]] = field(default_factory=list)        # 互斥对
    orphan_predicates: Set[str] = field(default_factory=set)               # 孤立谓词


@dataclass
class ConflictReport:
    """冲突分析报告"""
    total_rules: int = 0
    total_predicates: int = 0
    
    # 谓词层
    predicate_equivalences: List[Set[str]] = field(default_factory=list)
    predicate_implications: List[Tuple[str, str]] = field(default_factory=list)
    predicate_mutex: List[Tuple[str, str]] = field(default_factory=list)
    trivial_predicates: List[str] = field(default_factory=list)
    
    # 规则层
    redundant_rules: List[RulePairResult] = field(default_factory=list)
    conflicting_rules: List[RulePairResult] = field(default_factory=list)
    other_rule_relations: List[RulePairResult] = field(default_factory=list)  # 其他关系（OTHERS）
    
    # 建议
    merge_suggestions: List[Dict] = field(default_factory=list)
    deletion_suggestions: List[Dict] = field(default_factory=list)
    priority_suggestions: List[Dict] = field(default_factory=list)


# ============ 谓词分析器 ============

class PredicateAnalyzer:
    """谓词关系分析器"""
    
    def __init__(self, predicates: List[Dict], terms: Optional[List[Dict]] = None):
        """
        Args:
            predicates: 谓词列表 [{"id": "pred.xxx", "formal_definition": "$ ... $", ...}]
            terms: 术语列表（用于初始化 Z3 变量）
        """
        self.predicates = {p["id"]: p for p in predicates}
        self.z3env = Z3Env()
        
        # 预解析所有谓词表达式
        self._parsed_exprs = {}
        self._parse_errors = {}
        self._unsupported_ops = {}  # 记录不支持的谓词及其原因
        self._exclusive_ops = {}    # 记录互斥算子的谓词
        self._preparse_all()
    
    def _preparse_all(self):
        """预解析所有谓词"""
        for pid, pred in self.predicates.items():
            formal_def = pred.get("formal_definition", "")
            if not formal_def or formal_def.strip() in ("$ true $", "$true$"):
                self._parse_errors[pid] = "trivial definition"
                continue
            
            # 【新增】检查算子类型
            category = self.z3env.get_operator_category(formal_def)
            if category == "unsupported":
                self._unsupported_ops[pid] = "Duration/Assess 暂不支持 Z3 运算"
                continue
            if category == "exclusive":
                self._exclusive_ops[pid] = "Stage/Risk 视为互斥条件"
                continue
            if category == "unknown":
                self._parse_errors[pid] = "unknown operator type"
                continue
            
            try:
                self.z3env.ensure_vars_for_expr(formal_def)
                expr = self.z3env.expr(formal_def)
                self._parsed_exprs[pid] = expr
            except Exception as e:
                self._parse_errors[pid] = str(e)
                if len(self._parse_errors) <= 5:
                    print(f"  [DEBUG] 谓词解析失败 {pid}: {e}")
                elif len(self._parse_errors) == 6:
                    print(f"  [DEBUG] ... 更多解析错误已忽略")
    
    def get_parseable_predicates(self) -> List[str]:
        """返回可解析的谓词ID列表"""
        return list(self._parsed_exprs.keys())
    
    def get_parse_errors(self) -> Dict[str, str]:
        """返回解析错误（不包含不支持的算子）"""
        return self._parse_errors.copy()
    
    def get_unsupported_predicates(self) -> Dict[str, str]:
        """返回因不支持的算子而跳过的谓词"""
        return self._unsupported_ops.copy()
    
    def get_exclusive_predicates(self) -> Dict[str, str]:
        """返回互斥算子的谓词（Stage/Risk）"""
        return self._exclusive_ops.copy()
    
    def get_analysis_summary(self) -> Dict[str, int]:
        """获取分析统计摘要"""
        return {
            "parseable": len(self._parsed_exprs),
            "parse_errors": len(self._parse_errors),
            "unsupported_ops": len(self._unsupported_ops),
            "exclusive_ops": len(self._exclusive_ops),
            "total": len(self.predicates)
        }
    
    def analyze_pair(self, p_id: str, q_id: str) -> PredicatePairResult:
        """分析两个谓词的关系（目前仅关注等价和蕴含，其余归为 UNKNOWN）"""
        if p_id not in self._parsed_exprs:
            return PredicatePairResult(p_id, q_id, PredicateRelation.UNKNOWN, 
                                       f"Cannot parse {p_id}")
        if q_id not in self._parsed_exprs:
            return PredicatePairResult(p_id, q_id, PredicateRelation.UNKNOWN,
                                       f"Cannot parse {q_id}")
        
        p_expr = self._parsed_exprs[p_id]
        q_expr = self._parsed_exprs[q_id]
        
        # 确保两个表达式都是布尔类型
        if not is_bool(p_expr) or not is_bool(q_expr):
            return PredicatePairResult(p_id, q_id, PredicateRelation.UNKNOWN,
                                       "Non-boolean expressions")
        
        solver = Solver()
        solver.set("timeout", 5000)  # 5秒超时
        
        try:
            # 检查 P ⇒ Q：P ∧ ¬Q 不可满足
            solver.push()
            solver.add(And(p_expr, Not(q_expr)))
            result = solver.check()
            p_implies_q = (result == unsat)
            solver.pop()
            
            # 检查 Q ⇒ P：Q ∧ ¬P 不可满足
            solver.push()
            solver.add(And(q_expr, Not(p_expr)))
            result = solver.check()
            q_implies_p = (result == unsat)
            solver.pop()
            
            if p_implies_q and q_implies_p:
                return PredicatePairResult(p_id, q_id, PredicateRelation.EQUIVALENT,
                                           "P ⇔ Q: mutual implication")
            elif p_implies_q:
                return PredicatePairResult(p_id, q_id, PredicateRelation.P_IMPLIES_Q,
                                           f"{p_id} ⇒ {q_id}")
            elif q_implies_p:
                return PredicatePairResult(p_id, q_id, PredicateRelation.Q_IMPLIES_P,
                                           f"{q_id} ⇒ {p_id}")

            # 其他关系（相交 / 互斥等）当前不关心，统一标记为 UNKNOWN，避免额外求解开销
            return PredicatePairResult(p_id, q_id, PredicateRelation.UNKNOWN,
                                       "Relation not classified (only equivalence / implication tracked)")
        
        except Exception as e:
            return PredicatePairResult(p_id, q_id, PredicateRelation.UNKNOWN,
                                       f"Z3 error: {str(e)}")
    
    def build_lattice(self, progress_callback=None) -> PredicateLattice:
        """构建谓词格（分析所有谓词对）"""
        lattice = PredicateLattice()
        parseable = self.get_parseable_predicates()
        n = len(parseable)
        total_pairs = n * (n - 1) // 2
        
        equivalences = defaultdict(set)
        implications = []
        mutex_pairs = []  # 目前不再填充互斥关系
        
        pair_count = 0
        for i, p_id in enumerate(parseable):
            for q_id in parseable[i+1:]:
                pair_count += 1
                if progress_callback and pair_count % 100 == 0:
                    progress_callback(pair_count, total_pairs)
                
                result = self.analyze_pair(p_id, q_id)
                
                if result.relation == PredicateRelation.EQUIVALENT:
                    # 合并等价类
                    key = min(p_id, q_id)
                    equivalences[key].add(p_id)
                    equivalences[key].add(q_id)
                elif result.relation == PredicateRelation.P_IMPLIES_Q:
                    implications.append((p_id, q_id))
                elif result.relation == PredicateRelation.Q_IMPLIES_P:
                    implications.append((q_id, p_id))
                # 其它关系（包括互斥 / 相交 / unknown）当前不单独记录
        
        # 合并重叠的等价类
        merged_eq = self._merge_equivalence_classes(equivalences)
        lattice.equivalence_classes = merged_eq
        lattice.implication_edges = implications
        lattice.mutex_pairs = mutex_pairs
        
        # 找出孤立谓词
        all_related = set()
        for eq_class in merged_eq.values():
            all_related.update(eq_class)
        for p, q in implications:
            all_related.add(p)
            all_related.add(q)
        for p, q in mutex_pairs:
            all_related.add(p)
            all_related.add(q)
        
        lattice.orphan_predicates = set(parseable) - all_related
        
        return lattice
    
    def _merge_equivalence_classes(self, equiv: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """合并有交集的等价类"""
        if not equiv:
            return {}
        
        # Union-Find
        parent = {}
        
        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # 合并
        for members in equiv.values():
            members_list = list(members)
            for m in members_list[1:]:
                union(members_list[0], m)
        
        # 收集
        result = defaultdict(set)
        for members in equiv.values():
            for m in members:
                root = find(m)
                result[root].add(m)
        
        return dict(result)


# ============ 动作分析器 ============

class ActionAnalyzer:
    """
    动作关系分析器
    
    Permission 分为两大类，各有偏序：
    
    使用类（是否使用）:
        CONTRAINDICATE > AVOID > CAUTION > ALLOW > RECOMMEND > REQUIRE
        (禁忌)          (避免)   (慎用)   (允许)   (推荐)     (必须)
    
    剂量调整类（怎么用）:
        REDUCE_DOSE ≈ MAX_DOSE_LIMIT > START_LOW_DOSE > TITRATE > INCREASE_DOSE
        (减量)        (限量)          (起始低剂量)     (滴定)    (加量)
    
    两类之间的关系:
        - 禁止类 (contraindicate, avoid) vs 剂量调整类 → 冲突
        - 许可类 (caution, allow, recommend, require) vs 剂量调整类 → 兼容
    """
    
    # 分类
    USAGE_PERMISSIONS = {"contraindicate", "avoid", "caution", "allow", "recommend", "require", "continue", "stop", "consider"}
    DOSE_PERMISSIONS = {"reduce_dose", "max_dose_limit", "start_low_dose", "titrate", "increase_dose"}
    PROHIBIT_PERMISSIONS = {"contraindicate", "avoid","stop"}  # 禁止类
    ALLOW_PERMISSIONS = {"caution", "allow", "recommend", "require", "continue",  "consider"}  # 许可类
    
    @classmethod
    def _get_permission_type(cls, perm: str) -> str:
        """判断 permission 类型"""
        if perm in cls.USAGE_PERMISSIONS:
            return "usage"
        elif perm in cls.DOSE_PERMISSIONS:
            return "dose"
        return "unknown"
    
    @classmethod
    def is_same_permission_category(cls, perm_a: str, perm_b: str) -> bool:
        """
        判断两个 permission 是否属于同一类别
        
        同类别的定义：
        1. 完全相同
        2. 都是推荐类（recommend, require, allow）
        3. 都是禁止类（contraindicate, avoid）
        4. 都是谨慎类（caution）
        5. 都是剂量调整类
        """
        if perm_a == perm_b:
            return True
        
        # 推荐类
        if perm_a in cls.ALLOW_PERMISSIONS and perm_b in cls.ALLOW_PERMISSIONS:
            return True
        
        # 禁止类
        if perm_a in cls.PROHIBIT_PERMISSIONS and perm_b in cls.PROHIBIT_PERMISSIONS:
            return True
        
        # 剂量调整类
        if perm_a in cls.DOSE_PERMISSIONS and perm_b in cls.DOSE_PERMISSIONS:
            return True
        
        return False
    
    @classmethod
    def analyze_pair(cls, action_a: Dict, action_b: Dict) -> ActionPairResult:
        """分析两个 action 的关系"""
        subjects_a = set(action_a.get("subjects", []))
        subjects_b = set(action_b.get("subjects", []))
        
        perm_a = action_a.get("permission", "allow")
        perm_b = action_b.get("permission", "allow")
        
        # 动作等价的含义：subjects和permission都相同
        if subjects_a == subjects_b and perm_a == perm_b:
            return ActionPairResult(
                subjects_overlap=list(subjects_a),
                relation=ActionRelation.EQUIVALENT,
                details="Identical actions: same subjects, permission"
            )

        
        overlap = subjects_a & subjects_b
        
        # 如果没有重叠的subjects，这两个动作是独立的，不是等价的
        if not overlap:
            return ActionPairResult(
                subjects_overlap=[],
                relation=ActionRelation.INDEPENDENT,
                details="No overlapping subjects - actions are independent"
            )
        
        type_a = cls._get_permission_type(perm_a)
        type_b = cls._get_permission_type(perm_b)
        
        # Case 1: 两个都是使用类
        if type_a == "usage" and type_b == "usage":
            return cls._analyze_usage_pair(perm_a, perm_b, overlap, subjects_a, subjects_b, action_a, action_b)
        
        # Case 2: 两个都是剂量类
        if type_a == "dose" and type_b == "dose":
            return cls._analyze_dose_pair(perm_a, perm_b, overlap, subjects_a, subjects_b, action_a, action_b)
        
        # Case 3: 一个使用类，一个剂量类
        if type_a == "usage" and type_b == "dose":
            return cls._analyze_cross_type(perm_a, perm_b, overlap, "a_usage")
        if type_a == "dose" and type_b == "usage":
            return cls._analyze_cross_type(perm_b, perm_a, overlap, "b_usage")
        
        # 未知类型，保守处理
        return ActionPairResult(
            subjects_overlap=list(overlap),
            relation=ActionRelation.INDEPENDENT,
            details=f"Unknown permission type: {perm_a}, {perm_b}"
        )
    
    @classmethod
    def _analyze_usage_pair(cls, perm_a: str, perm_b: str, overlap: set, 
                            subjects_a: set, subjects_b: set,
                            action_a: Dict, action_b: Dict) -> ActionPairResult:
        """分析两个使用类 permission 的关系（在有重叠subjects的情况下，返回CONFLICT、DISAGREEMENT或INDEPENDENT）"""
        a_prohibit = perm_a in cls.PROHIBIT_PERMISSIONS
        b_prohibit = perm_b in cls.PROHIBIT_PERMISSIONS
        a_allow = perm_a in cls.ALLOW_PERMISSIONS
        b_allow = perm_b in cls.ALLOW_PERMISSIONS
        
        # 禁止 vs 许可 → 冲突
        if a_prohibit and b_allow:
            return ActionPairResult(
                subjects_overlap=list(overlap),
                relation=ActionRelation.CONFLICT,
                dominant_permission=perm_a,
                details=f"Conflict: {perm_a} (prohibit) vs {perm_b} (allow) on {overlap}"
            )
        if b_prohibit and a_allow:
            return ActionPairResult(
                subjects_overlap=list(overlap),
                relation=ActionRelation.CONFLICT,
                dominant_permission=perm_b,
                details=f"Conflict: {perm_a} (allow) vs {perm_b} (prohibit) on {overlap}"
            )
        
        # 如果permission相同，已经在analyze_pair中处理过了（会返回EQUIVALENT）
        # 如果permission不同，检查是否为分歧（同类permission但具体不同）
        if cls.is_same_permission_category(perm_a, perm_b):
            # 同类permission但具体不同 → 分歧
            return ActionPairResult(
                subjects_overlap=list(overlap),
                relation=ActionRelation.DISAGREEMENT,
                details=f"Disagreement: {perm_a} vs {perm_b} (same category but different strength)"
            )
        
        # 其他情况（不同类别的permission）→ 独立
        return ActionPairResult(
            subjects_overlap=list(overlap),
            relation=ActionRelation.INDEPENDENT,
            details=f"Different permission categories: {perm_a} vs {perm_b}"
        )
    
    @classmethod
    def is_disagreement(cls, perm_a: str, perm_b: str, action_a: Dict, action_b: Dict) -> bool:
        """
        判断两个permission是否为分歧（同类permission但具体不同）
        用于辅助判断，实际判断应该通过analyze_pair返回的ActionRelation
        """
        # 完全相同不是分歧
        if perm_a == perm_b:
            reqs_a = set(action_a.get("requirements", []))
            reqs_b = set(action_b.get("requirements", []))
            subjects_a = set(action_a.get("subjects", []))
            subjects_b = set(action_b.get("subjects", []))
            if reqs_a == reqs_b and subjects_a == subjects_b:
                return False
        
        # 禁止 vs 许可 → 不是分歧，是冲突
        a_prohibit = perm_a in cls.PROHIBIT_PERMISSIONS
        b_prohibit = perm_b in cls.PROHIBIT_PERMISSIONS
        a_allow = perm_a in cls.ALLOW_PERMISSIONS
        b_allow = perm_b in cls.ALLOW_PERMISSIONS
        if (a_prohibit and b_allow) or (b_prohibit and a_allow):
            return False
        
        # 同类permission但具体不同 → 分歧
        return cls.is_same_permission_category(perm_a, perm_b)
    
    @classmethod
    def _analyze_dose_pair(cls, perm_a: str, perm_b: str, overlap: set,
                           subjects_a: set, subjects_b: set,
                           action_a: Dict, action_b: Dict) -> ActionPairResult:
        """分析两个剂量调整类 permission 的关系（在有重叠subjects的情况下，返回CONFLICT、DISAGREEMENT或INDEPENDENT）"""
        # 明显矛盾（reduce_dose vs increase_dose）→ 冲突
        if {perm_a, perm_b} == {"reduce_dose", "increase_dose"}:
            return ActionPairResult(
                subjects_overlap=list(overlap),
                relation=ActionRelation.CONFLICT,
                details=f"Conflict: {perm_a} vs {perm_b} are contradictory dose adjustments"
            )
        
        # 如果permission相同，已经在analyze_pair中处理过了（会返回EQUIVALENT）
        # 如果permission不同，但都是剂量类 → 分歧（同类permission但具体不同）
        if cls.is_same_permission_category(perm_a, perm_b):
            return ActionPairResult(
                subjects_overlap=list(overlap),
                relation=ActionRelation.DISAGREEMENT,
                details=f"Disagreement: {perm_a} vs {perm_b} (same dose category but different adjustment)"
            )
        
        # 其他情况 → 独立
        return ActionPairResult(
            subjects_overlap=list(overlap),
            relation=ActionRelation.INDEPENDENT,
            details=f"Different dose permissions: {perm_a} vs {perm_b}"
        )
    
    @classmethod
    def _analyze_cross_type(cls, usage_perm: str, dose_perm: str, 
                            overlap: set, which_usage: str) -> ActionPairResult:
        """分析使用类 vs 剂量类的关系（在有重叠subjects的情况下，返回CONFLICT或INDEPENDENT）"""
        # 禁止类 vs 剂量类 → 冲突（不能用就没有剂量问题）
        if usage_perm in cls.PROHIBIT_PERMISSIONS:
            return ActionPairResult(
                subjects_overlap=list(overlap),
                relation=ActionRelation.CONFLICT,
                dominant_permission=usage_perm,
                details=f"Conflict: {usage_perm} prohibits use, {dose_perm} implies use"
            )
        
        # 许可类 vs 剂量类 → 不是等价（不同类型的permission）
        # 虽然不冲突，但这是不同类型的动作，不应该算等价
        return ActionPairResult(
            subjects_overlap=list(overlap),
            relation=ActionRelation.INDEPENDENT,
            details=f"Different permission types: {usage_perm} (usage) vs {dose_perm} (dose) - not equivalent"
        )


# ============ 规则分析器 ============

class RuleAnalyzer:
    """规则关系分析器"""
    
    def __init__(self, rules: List[Dict], predicates: List[Dict]):
        """
        Args:
            rules: 规则列表
            predicates: 谓词列表（用于条件分析）
        """
        from pathlib import Path
        self.rules = {r["id"]: r for r in rules}
        self.pred_analyzer = PredicateAnalyzer(predicates)
        
        # 为规则的 condition 创建虚拟谓词
        self._rule_conditions = {}
        self._prepare_rule_conditions()

        # 谓词关系（等价 / 蕴含），优先复用已导出的结果，避免重复计算
        # 结构来自 export_predicate_relations: {"equivalences": [...], "implications": [[p,q], ...]}
        self._pred_relations: Optional[Dict[str, Any]] = None
        self._equiv_rep: Dict[str, str] = {}          # 谓词 -> 等价类代表
        self._implies_succ: Dict[str, Set[str]] = {}  # 谓词 -> 直接蕴含的后继集合
        self._implies_cache: Dict[Tuple[str, str], bool] = {}

        try:
            rel_path = Path("gen") / "predicate_relations.json"
            if rel_path.exists():
                with open(rel_path, "r", encoding="utf-8") as f:
                    self._pred_relations = json.load(f)
                # 构建等价类代表映射
                for eq_class in self._pred_relations.get("equivalences", []):
                    if not eq_class:
                        continue
                    rep = min(eq_class)
                    for pid in eq_class:
                        self._equiv_rep[pid] = rep
                # 构建蕴含图
                succ: Dict[str, Set[str]] = defaultdict(set)
                for p, q in self._pred_relations.get("implications", []):
                    succ[p].add(q)
                self._implies_succ = dict(succ)
        except Exception:
            # 如果加载失败，不影响后续逻辑，只是无法使用快速路径
            self._pred_relations = None

    def _prepare_rule_conditions(self):
        """将规则的 condition 转换为可分析的形式"""
        for rid, rule in self.rules.items():
            condition = rule.get("condition", "")
            # 空条件也应该存储，表示"无条件/总是成立"
            # condition 是引用谓词的布尔表达式，如 "pred.a AND pred.b"
            # 我们需要展开它
            self._rule_conditions[rid] = condition
    
    def _expand_condition(self, condition: str) -> Optional[str]:
        """
        展开规则条件为 formal_definition
        
        例如：pred.a AND pred.b 展开为实际的逻辑表达式
        """
        if not condition:
            return None
        
        # 替换 AND/OR/NOT
        expr = condition
        expr = expr.replace(" AND ", " & ")
        expr = expr.replace(" OR ", " | ")
        expr = expr.replace("NOT ", "~ ")
        
        # 替换谓词引用为其 formal_definition
        import re
        pred_refs = re.findall(r'pred\.[a-zA-Z0-9_]+', expr)
        
        for ref in pred_refs:
            if ref in self.pred_analyzer.predicates:
                formal_def = self.pred_analyzer.predicates[ref].get("formal_definition", "")
                if formal_def.startswith("$") and formal_def.endswith("$"):
                    inner = formal_def[1:-1].strip()
                    expr = expr.replace(ref, f"({inner})")
                else:
                    return None  # 无法展开
            else:
                return None  # 谓词不存在
        
        return f"$ {expr} $"
    
    # ======== 条件关系分析（优先使用谓词关系的快速路径） ========

    def _canonical_pred(self, pid: str) -> str:
        """将谓词映射到其等价类代表（若存在）"""
        return self._equiv_rep.get(pid, pid)

    def _extract_predicates_from_condition(self, condition: str) -> Optional[Set[str]]:
        """
        从规则的 condition 字符串中提取谓词集合（仅支持纯 AND 形式，含 OR/NOT 则返回 None）
        例如: 'pred.a AND pred.b' / 'pred.a & pred.b'
        """
        if not condition:
            return None
        c = condition.strip()
        up = c.upper()
        # 包含 OR / NOT 或按位 OR / 取反，视为复杂条件，交给 Z3
        if " OR " in up or "NOT " in up or "|" in c or "~" in c:
            return None

        import re
        preds = set(re.findall(r'pred\.[a-zA-Z0-9_]+', c))
        if not preds:
            return None
        # 映射到等价类代表
        return {self._canonical_pred(p) for p in preds}

    def _predicate_implies(self, p: str, q: str) -> bool:
        """利用蕴含图检查谓词级的 p ⇒ q（包含等价类代表）"""
        p = self._canonical_pred(p)
        q = self._canonical_pred(q)
        if p == q:
            return True
        key = (p, q)
        if key in self._implies_cache:
            return self._implies_cache[key]

        visited: Set[str] = set()
        stack = [p]
        while stack:
            cur = stack.pop()
            if cur == q:
                self._implies_cache[key] = True
                return True
            for nxt in self._implies_succ.get(cur, ()):
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        self._implies_cache[key] = False
        return False

    def _analyze_condition_pair_fast(self, rule_a_id: str, rule_b_id: str) -> PredicateRelation:
        """
        使用谓词等价 / 蕴含关系的快速近似：
        仅在规则条件为"纯 AND 的谓词集合"时尝试，
        根据集合包含 + 谓词级蕴含近似判断：
            - 等价
            - A ⇒ B / B ⇒ A
        复杂条件返回 UNKNOWN，交给 Z3 精确判定。
        """
        if not self._pred_relations:
            return PredicateRelation.UNKNOWN

        cond_a = self._rule_conditions.get(rule_a_id, "")
        cond_b = self._rule_conditions.get(rule_b_id, "")
        
        # 两个空条件等价
        if not cond_a and not cond_b:
            return PredicateRelation.EQUIVALENT
        
        # 一个空一个非空，返回UNKNOWN让Z3处理
        if not cond_a or not cond_b:
            return PredicateRelation.UNKNOWN

        preds_a = self._extract_predicates_from_condition(cond_a)
        preds_b = self._extract_predicates_from_condition(cond_b)
        if preds_a is None or preds_b is None:
            return PredicateRelation.UNKNOWN

        # A ⇒ B：对每个 q∈B，存在 p∈A 使得 p ⇒ q
        def cond_implies(P: Set[str], Q: Set[str]) -> bool:
            for q in Q:
                if any(self._predicate_implies(p, q) for p in P):
                    continue
                # 如果没有任何 p ⇒ q，则无法保证 C_P ⇒ C_Q
                return False
            return True

        a_implies_b = cond_implies(preds_a, preds_b)
        b_implies_a = cond_implies(preds_b, preds_a)

        if a_implies_b and b_implies_a:
            return PredicateRelation.EQUIVALENT
        if a_implies_b:
            return PredicateRelation.P_IMPLIES_Q
        if b_implies_a:
            return PredicateRelation.Q_IMPLIES_P
        return PredicateRelation.UNKNOWN

    def _analyze_condition_pair_z3(self, rule_a_id: str, rule_b_id: str) -> PredicateRelation:
        """使用 Z3 精确分析两条规则的条件关系（作为快速路径的回退方案）"""
        cond_a = self._rule_conditions.get(rule_a_id, "")
        cond_b = self._rule_conditions.get(rule_b_id, "")
        
        # 两个空条件等价（都表示"无条件/总是成立"）
        if not cond_a and not cond_b:
            return PredicateRelation.EQUIVALENT
        
        # 一个空一个非空，需要进一步分析（空条件可以视为true，非空条件需要检查是否总是成立）
        # 但为了简化，这里先返回UNKNOWN，后续可以改进
        if not cond_a or not cond_b:
            # 如果字符串相同（都是空），返回等价
            if cond_a == cond_b:
                return PredicateRelation.EQUIVALENT
            return PredicateRelation.UNKNOWN
        
        # 尝试展开
        expanded_a = self._expand_condition(cond_a)
        expanded_b = self._expand_condition(cond_b)
        
        if not expanded_a or not expanded_b:
            # 如果无法展开，尝试直接比较字符串
            if cond_a == cond_b:
                return PredicateRelation.EQUIVALENT
            return PredicateRelation.UNKNOWN
        
        # 使用 Z3 分析
        try:
            self.pred_analyzer.z3env.ensure_vars_for_expr(expanded_a)
            self.pred_analyzer.z3env.ensure_vars_for_expr(expanded_b)
            
            expr_a = self.pred_analyzer.z3env.expr(expanded_a)
            expr_b = self.pred_analyzer.z3env.expr(expanded_b)
            
            solver = Solver()
            
            # 等价检查
            solver.push()
            solver.add(expr_a != expr_b)
            if solver.check() == unsat:
                return PredicateRelation.EQUIVALENT
            solver.pop()
            
            # A ⇒ B
            solver.push()
            solver.add(And(expr_a, Not(expr_b)))
            a_implies_b = (solver.check() == unsat)
            solver.pop()
            
            # B ⇒ A
            solver.push()
            solver.add(And(expr_b, Not(expr_a)))
            b_implies_a = (solver.check() == unsat)
            solver.pop()
            
            if a_implies_b and b_implies_a:
                return PredicateRelation.EQUIVALENT
            elif a_implies_b:
                return PredicateRelation.P_IMPLIES_Q
            elif b_implies_a:
                return PredicateRelation.Q_IMPLIES_P
            
            # 互斥检查
            solver.push()
            solver.add(And(expr_a, expr_b))
            if solver.check() == unsat:
                return PredicateRelation.MUTEX
            solver.pop()
            
            return PredicateRelation.INTERSECT
            
        except Exception:
            return PredicateRelation.UNKNOWN

    def _build_rule_lsh_signature(self, rule_id: str) -> List[int]:
        """
        为规则构建LSH签名，基于condition和action.subjects
        
        Returns:
            MinHash签名列表
        """
        rule = self.rules.get(rule_id)
        if not rule:
            return []
        
        # 构建特征字符串：condition + subjects
        condition = self._rule_conditions.get(rule_id, "")
        action = rule.get("action", {})
        subjects = action.get("subjects", [])
        
        # 组合特征：condition和subjects都作为特征
        features = []
        if condition:
            features.append(f"cond:{condition}")
        for subj in sorted(subjects):
            features.append(f"subj:{subj}")
        
        # 如果没有特征，返回空签名
        if not features:
            return []
        
        # 计算shingles和minhash
        feature_text = " ".join(features)
        shingles = compute_shingles(feature_text, k=5)  # 使用k=5，因为pred和med名称较长
        signature = compute_minhash(shingles, num_hashes=100)
        return signature
    
    def _find_similar_rule_pairs_lsh(self, 
                                     num_bands: int = 20, 
                                     rows_per_band: int = 5,
                                     similarity_threshold: float = 0.3) -> Set[Tuple[str, str]]:
        """
        使用LSH找到相似的规则对（基于condition和action.subjects）
        
        Args:
            num_bands: LSH带数量
            rows_per_band: 每带行数
            similarity_threshold: 相似度阈值（Jaccard）
        
        Returns:
            候选规则对集合
        """
        rule_ids = list(self.rules.keys())
        if len(rule_ids) < 2:
            return set()
        
        # 计算每个规则的MinHash签名
        signatures = {}
        for rid in rule_ids:
            sig = self._build_rule_lsh_signature(rid)
            if sig:  # 只处理有特征的规则
                signatures[rid] = sig
        
        # LSH: 将签名分成多个带，每个带作为桶的键
        buckets = defaultdict(list)
        for rid, sig in signatures.items():
            for band in range(num_bands):
                start = band * rows_per_band
                end = start + rows_per_band
                if end <= len(sig):
                    band_sig = tuple(sig[start:end])
                    key = (band, band_sig)
                    buckets[key].append(rid)
        
        # 找到候选对（同一bucket内的规则对）
        candidate_pairs: Set[Tuple[str, str]] = set()
        for indices in buckets.values():
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        r_a, r_b = indices[i], indices[j]
                        # 确保顺序一致（小的在前）
                        if r_a > r_b:
                            r_a, r_b = r_b, r_a
                        candidate_pairs.add((r_a, r_b))
        
        # 进一步基于 MinHash 估计的 Jaccard 相似度过滤
        # 仅保留估计相似度 >= similarity_threshold 的规则对
        if similarity_threshold is not None and similarity_threshold > 0:
            def est_sim(sig1: List[int], sig2: List[int]) -> float:
                """根据 MinHash 签名估计 Jaccard 相似度（相同位置占比）"""
                n = min(len(sig1), len(sig2))
                if n == 0:
                    return 0.0
                same = sum(1 for k in range(n) if sig1[k] == sig2[k])
                return same / n
            
            filtered_pairs: Set[Tuple[str, str]] = set()
            for r_a, r_b in candidate_pairs:
                sig_a = signatures.get(r_a)
                sig_b = signatures.get(r_b)
                if not sig_a or not sig_b:
                    continue
                if est_sim(sig_a, sig_b) >= similarity_threshold:
                    filtered_pairs.add((r_a, r_b))
            return filtered_pairs
        
        return candidate_pairs
    
    def analyze_condition_pair(self, rule_a_id: str, rule_b_id: str) -> PredicateRelation:
        """分析两条规则的条件关系（优先使用谓词关系的快速判定，失败时回退到 Z3）"""
        # 快速路径：基于 predicate_relations.json 的等价 / 蕴含
        fast_rel = self._analyze_condition_pair_fast(rule_a_id, rule_b_id)
        if fast_rel != PredicateRelation.UNKNOWN:
            return fast_rel
        # 回退到 Z3 精确分析
        return self._analyze_condition_pair_z3(rule_a_id, rule_b_id)
    
    def analyze_pair(self, rule_a_id: str, rule_b_id: str) -> RulePairResult:
        """分析两条规则的综合关系"""
        rule_a = self.rules.get(rule_a_id)
        rule_b = self.rules.get(rule_b_id)
        
        if not rule_a or not rule_b:
            return RulePairResult(
                rule_a=rule_a_id,
                rule_b=rule_b_id,
                condition_relation=PredicateRelation.UNKNOWN,
                action_relation=ActionRelation.INDEPENDENT,
                overall_relation=RuleRelation.OTHERS,
                recommendation="Rule not found"
            )
        
        # 分析条件关系
        cond_rel = self.analyze_condition_pair(rule_a_id, rule_b_id)
        
        # 分析动作关系
        action_a = rule_a.get("action", {})
        action_b = rule_b.get("action", {})
        action_result = ActionAnalyzer.analyze_pair(action_a, action_b)
        action_rel = action_result.relation
        
        # 综合判定
        overall, recommendation = self._determine_overall_relation(
            cond_rel, action_rel, action_result, rule_a_id, rule_b_id
        )
        
        return RulePairResult(
            rule_a=rule_a_id,
            rule_b=rule_b_id,
            condition_relation=cond_rel,
            action_relation=action_rel,
            overall_relation=overall,
            recommendation=recommendation
        )
    
    def _determine_overall_relation(
        self, 
        cond_rel: PredicateRelation, 
        action_rel: ActionRelation,
        action_result: ActionPairResult,
        rule_a: str,
        rule_b: str
    ) -> Tuple[RuleRelation, str]:
        """
        确定规则的综合关系和处理建议
        
        忠实还原 RuleRelation 枚举的定义：
        1. COMPLETE_REDUNDANCY - PredicateRelation.EQUIVALENT + ActionRelation.EQUIVALENT
        2. CONTAINED_REDUNDANCY - PredicateRelation.P_IMPLIES_Q/Q_IMPLIES_P + ActionRelation.EQUIVALENT
        3. INTRINSIC_CONFLICT - PredicateRelation.EQUIVALENT + ActionRelation.CONFLICT
        4. INTRINSIC_DISAGREEMENT - PredicateRelation.EQUIVALENT + ActionRelation.DISAGREEMENT
        5. IMPLICATION_CONFLICT - PredicateRelation.P_IMPLIES_Q/Q_IMPLIES_P + ActionRelation.CONFLICT
        6. IMPLICATION_DISAGREEMENT - PredicateRelation.P_IMPLIES_Q/Q_IMPLIES_P + ActionRelation.DISAGREEMENT
        7. LOCAL_CONFLICT - PredicateRelation.INTERSECT + ActionRelation.CONFLICT
        8. OTHERS - 其他情况
        """
        # 如果动作是独立的（没有重叠的subjects），无论条件关系如何，规则都是独立的
        if action_rel == ActionRelation.INDEPENDENT:
            return RuleRelation.OTHERS, \
                f"INDEPENDENT: Actions have no overlapping subjects, rules are independent"
        
        # 条件等价
        if cond_rel == PredicateRelation.EQUIVALENT:
            if action_rel == ActionRelation.EQUIVALENT:
                # 条件等价 + 动作等价 → 完全冗余
                return RuleRelation.COMPLETE_REDUNDANCY, \
                    f"COMPLETE REDUNDANCY: {rule_a} and {rule_b} have equivalent conditions and actions. Merge and combine provenance."
            elif action_rel == ActionRelation.CONFLICT:
                # 条件等价 + 动作冲突 → 固有冲突
                return RuleRelation.INTRINSIC_CONFLICT, \
                    f"INTRINSIC CONFLICT: {rule_a} and {rule_b} have equivalent conditions but conflicting actions. Review provenance to resolve."
            elif action_rel == ActionRelation.DISAGREEMENT:
                # 条件等价 + 动作分歧 → 固有分歧
                return RuleRelation.INTRINSIC_DISAGREEMENT, \
                    f"INTRINSIC DISAGREEMENT: {rule_a} and {rule_b} have equivalent conditions but different recommendation strength. Requires manual reconciliation."
            else:
                return RuleRelation.OTHERS, f"Same condition, non-conflicting actions (not classified)"
        
        # 条件蕴含：A => B（A更具体）
        if cond_rel == PredicateRelation.P_IMPLIES_Q:
            if action_rel == ActionRelation.EQUIVALENT:
                return RuleRelation.CONTAINED_REDUNDANCY, \
                    f"CONTAINED REDUNDANCY: {rule_a} is a specialization of {rule_b} with equivalent action. Consider merging and keeping provenance."
            elif action_rel == ActionRelation.CONFLICT:
                return RuleRelation.IMPLICATION_CONFLICT, \
                    f"IMPLICATION CONFLICT: {rule_a}'s condition implies {rule_b}, but actions conflict."
            elif action_rel == ActionRelation.DISAGREEMENT:
                return RuleRelation.IMPLICATION_DISAGREEMENT, \
                    f"IMPLICATION DISAGREEMENT: {rule_a}'s condition implies {rule_b}, but actions disagree (same category but different strength)."
            else:
                return RuleRelation.OTHERS, f"Specialization with non-conflicting actions (not classified)"
        
        # 条件蕴含：B => A（B更具体）
        if cond_rel == PredicateRelation.Q_IMPLIES_P:
            if action_rel == ActionRelation.EQUIVALENT:
                return RuleRelation.CONTAINED_REDUNDANCY, \
                    f"CONTAINED REDUNDANCY: {rule_b} is a specialization of {rule_a} with equivalent action. Consider merging and keeping provenance."
            elif action_rel == ActionRelation.CONFLICT:
                return RuleRelation.IMPLICATION_CONFLICT, \
                    f"IMPLICATION CONFLICT: {rule_b}'s condition implies {rule_a}, but actions conflict."
            elif action_rel == ActionRelation.DISAGREEMENT:
                return RuleRelation.IMPLICATION_DISAGREEMENT, \
                    f"IMPLICATION DISAGREEMENT: {rule_b}'s condition implies {rule_a}, but actions disagree (same category but different strength)."
            else:
                return RuleRelation.OTHERS, f"Specialization with non-conflicting actions (not classified)"
        
        # 条件相交
        if cond_rel == PredicateRelation.INTERSECT:
            if action_rel == ActionRelation.CONFLICT:
                return RuleRelation.LOCAL_CONFLICT, \
                    f"LOCAL CONFLICT: Overlapping conditions with conflicting actions on {action_result.subjects_overlap}. " \
                    f"Add priority strategy or refine conditions."
            else:
                return RuleRelation.OTHERS, f"Overlapping conditions with non-conflicting actions (not classified)"
        
        # 条件互斥或未知
        if cond_rel == PredicateRelation.MUTEX:
            return RuleRelation.OTHERS, "Rules are mutually exclusive, no conflict"
        
        # 未知条件关系
        if cond_rel == PredicateRelation.UNKNOWN:
            if action_rel == ActionRelation.CONFLICT:
                return RuleRelation.LOCAL_CONFLICT, \
                    f"LOCAL CONFLICT: Unknown condition relationship but conflicting actions on {action_result.subjects_overlap}. " \
                    f"May need manual review to determine if conditions overlap."
            else:
                return RuleRelation.OTHERS, f"Unknown condition relationship with non-conflicting actions (not classified)"
        
        # 其他情况
        return RuleRelation.OTHERS, "Unable to determine relationship"


# ============ 冲突消解器 ============

class ConflictResolver:
    """冲突消解器 - 生成综合报告和建议"""
    
    def __init__(self, rules: List[Dict], predicates: List[Dict]):
        self.rules = rules
        self.predicates = predicates
        self.pred_analyzer = PredicateAnalyzer(predicates)
        self.rule_analyzer = RuleAnalyzer(rules, predicates)

    
    def analyze_predicates(self, progress_callback=None) -> PredicateLattice:
        """分析谓词层"""
        summary = self.pred_analyzer.get_analysis_summary()
        print(f"[PredicateAnalyzer] 谓词分析统计:")
        print(f"  可解析: {summary['parseable']}")
        print(f"  解析错误: {summary['parse_errors']}")
        print(f"  不支持算子(跳过): {summary['unsupported_ops']}")
        print(f"  互斥算子(Stage/Risk): {summary['exclusive_ops']}")
        
        if summary['parseable'] == 0:
            print(f"  [WARNING] 没有可解析的谓词")
            return PredicateLattice()
        
        lattice = self.pred_analyzer.build_lattice(progress_callback)
        
        print(f"  等价类: {len(lattice.equivalence_classes)}")
        print(f"  蕴含边: {len(lattice.implication_edges)}")
        
        return lattice


        
        return lattice
    
    def export_predicate_relations(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        计算并导出谓词间的等价 / 蕴含关系，便于在规则分析时复用。
        
        结构：
            {
              "equivalences": [  # 等价类列表，每个元素是等价谓词ID列表
                ["pred.a", "pred.b", ...],
                ...
              ],
              "implications": [  # 蕴含边列表 (P ⇒ Q)
                ["pred.p", "pred.q"],
                ...
              ]
            }
        """
        lattice = self.analyze_predicates()
        
        data = {
            "equivalences": [sorted(list(eq)) for eq in lattice.equivalence_classes.values()],
            "implications": [[p, q] for (p, q) in lattice.implication_edges],
        }
        
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        return data
    
    def _build_rule_bucket_map(self) -> Dict[str, Set[int]]:
        """
        构建 rule_id -> bucket_ids 的映射
        一个rule可能关联多个provenance，因此可能属于多个bucket
        """
        rule_bucket_map = {}
        for r in self.rules:
            rid = r["id"]
            bucket_ids = set()
            for prov in r.get("provenance", []):
                bucket_id = prov.get("bucket_id")
                if bucket_id is not None:
                    bucket_ids.add(bucket_id)
            rule_bucket_map[rid] = bucket_ids
        return rule_bucket_map
    
    def _should_compare_rules(
        self, 
        rule_a: str, 
        rule_b: str, 
        rule_bucket_map: Dict[str, Set[int]]
    ) -> bool:
        """
        判断两个规则是否需要进行Z3验证
        只有当它们有共同的bucket_id时才需要验证
        如果某个规则没有bucket信息，则默认需要验证
        """
        buckets_a = rule_bucket_map.get(rule_a, set())
        buckets_b = rule_bucket_map.get(rule_b, set())
        
        # 如果任一规则没有bucket信息，保守起见进行验证
        if not buckets_a or not buckets_b:
            return True
        
        # 只有有交集时才验证
        return bool(buckets_a & buckets_b)

    _PRIMARY_RELATION_LABELS = {
        RuleRelation.COMPLETE_REDUNDANCY: "complete_redundancy",
        RuleRelation.CONTAINED_REDUNDANCY: "contained_redundancy",
        RuleRelation.LOCAL_CONFLICT: "local_conflict",
        RuleRelation.INTRINSIC_CONFLICT: "intrinsic_conflict",
        RuleRelation.IMPLICATION_CONFLICT: "implication_conflict",
        RuleRelation.INTRINSIC_DISAGREEMENT: "intrinsic_disagreement",
        RuleRelation.IMPLICATION_DISAGREEMENT: "implication_disagreement",
    }
    
    def analyze_rules(self, 
                      focus_on: Optional[str] = "conflicts",
                      max_pairs: int = 10000,
                      use_lsh_filter: bool = True,
                      use_bucket_filter: bool = True,
                      lsh_similarity_threshold: float = 0.3,
                      lsh_num_bands: int = 20,
                      lsh_rows_per_band: int = 5,
                      progress_callback=None,
                      candidate_pairs: Optional[Set[Tuple[str, str]]] = None) -> Union[List[RulePairResult], Dict[str, List[RulePairResult]]]:
        """
        分析规则层。
        
        - focus_on=None 时返回按类别聚合的结果（6类核心关系 + others + all）
        - 其他值保持兼容旧接口，返回指定类别的规则对列表
        - lsh_similarity_threshold: LSH 预筛选的 MinHash 估计相似度阈值（0~1），越大越严格
        - lsh_num_bands / lsh_rows_per_band: LSH 带数和每带行数，用于控制候选集规模
        """
        summary, analyzed_pairs = self._collect_rule_relations(
            max_pairs=max_pairs,
            use_lsh_filter=use_lsh_filter,
            use_bucket_filter=use_bucket_filter,
            lsh_similarity_threshold=lsh_similarity_threshold,
            lsh_num_bands=lsh_num_bands,
            lsh_rows_per_band=lsh_rows_per_band,
            progress_callback=progress_callback,
            candidate_pairs=candidate_pairs
        )
        self._apply_transitive_closure(summary, analyzed_pairs)
        
        if focus_on is None:
            return summary
        
        if focus_on == "conflicts":
            return (
                summary["local_conflict"] +
                summary["intrinsic_conflict"] +
                summary["implication_conflict"] +
                summary["intrinsic_disagreement"] +
                summary["implication_disagreement"]
            )
        if focus_on == "redundant":
            return summary["complete_redundancy"] + summary["contained_redundancy"]
        if focus_on == "others":
            return summary["others"]
        if focus_on == "all":
            return summary["all"]
        
        raise ValueError(f"Unknown focus_on value: {focus_on}")
    
    def _collect_rule_relations(self,
                                max_pairs: int,
                                use_lsh_filter: bool,
                                use_bucket_filter: bool,
                                lsh_similarity_threshold: float,
                                lsh_num_bands: int,
                                lsh_rows_per_band: int,
                                progress_callback=None,
                                candidate_pairs: Optional[Set[Tuple[str, str]]] = None) -> Tuple[Dict[str, List[RulePairResult]], Dict[Tuple[str, str], RulePairResult]]:
        """一次遍历所有候选规则对并按关系类别聚合结果。"""
        summary = {
            "complete_redundancy": [],
            "contained_redundancy": [],
            "local_conflict": [],
            "intrinsic_conflict": [],
            "implication_conflict": [],
            "intrinsic_disagreement": [],
            "implication_disagreement": [],
            "others": [],
            "all": [],
        }
        analyzed_pairs: Dict[Tuple[str, str], RulePairResult] = {}
        
        rule_ids = list(self.rule_analyzer.rules.keys())
        n = len(rule_ids)
        total_pairs = n * (n - 1) // 2
        
        if candidate_pairs is None and use_lsh_filter:
            print(f"[RuleAnalyzer] 使用LSH基于condition和subjects找到相似候选对 (阈值={lsh_similarity_threshold})...")
            candidate_pairs = self.rule_analyzer._find_similar_rule_pairs_lsh(
                num_bands=lsh_num_bands,
                rows_per_band=lsh_rows_per_band,
                similarity_threshold=lsh_similarity_threshold,
            )
            reduction = 100 * (1 - len(candidate_pairs) / total_pairs) if total_pairs else 0
            print(f"[RuleAnalyzer] LSH找到 {len(candidate_pairs)} 个候选规则对（原本 {total_pairs} 对，减少 {reduction:.1f}%）")
        
        rule_bucket_map = self._build_rule_bucket_map() if use_bucket_filter and not use_lsh_filter else {}
        skipped_count = 0
        analyzed_count = 0
        
        def record_result(result: RulePairResult):
            key = tuple(sorted((result.rule_a, result.rule_b)))
            analyzed_pairs[key] = result
            label = self._PRIMARY_RELATION_LABELS.get(result.overall_relation)
            if label:
                summary[label].append(result)
            else:
                summary["others"].append(result)
            summary["all"].append(result)
        
        def process_pair(r_a: str, r_b: str):
            nonlocal analyzed_count
            analyzed_count += 1
            result = self.rule_analyzer.analyze_pair(r_a, r_b)
            record_result(result)
            if progress_callback and analyzed_count % 100 == 0:
                expected_total = len(candidate_pairs) if candidate_pairs is not None else min(total_pairs, max_pairs)
                progress_callback(analyzed_count, expected_total)
        
        if candidate_pairs is not None:
            for r_a, r_b in candidate_pairs:
                process_pair(r_a, r_b)
        else:
            for i, r_a in enumerate(rule_ids):
                for r_b in rule_ids[i+1:]:
                    if use_bucket_filter and rule_bucket_map:
                        if not self._should_compare_rules(r_a, r_b, rule_bucket_map):
                            skipped_count += 1
                            continue
                    if analyzed_count >= max_pairs:
                        break
                    process_pair(r_a, r_b)
                if analyzed_count >= max_pairs:
                    break
        
        if skipped_count > 0:
            print(f"[RuleAnalyzer] 跳过 {skipped_count} 对（不同bucket，无需验证）")
        if candidate_pairs is not None:
            print(f"[RuleAnalyzer] 实际验证 {analyzed_count} 对（LSH候选对）")
        else:
            print(f"[RuleAnalyzer] 实际验证 {analyzed_count} 对（遍历模式）")
        
        return summary, analyzed_pairs

    def _apply_transitive_closure(self,
                                  summary: Dict[str, List[RulePairResult]],
                                  analyzed_pairs: Dict[Tuple[str, str], RulePairResult]):
        """在每个核心关系内执行传递闭包，补充遗漏的规则对。"""
        core_labels = [
            "complete_redundancy",
            "contained_redundancy",
            "local_conflict",
            "intrinsic_conflict",
            "implication_conflict",
            "intrinsic_disagreement",
            "implication_disagreement",
        ]
        
        for label in core_labels:
            pairs = summary[label]
            if not pairs:
                continue
            components = self._build_components(pairs)
            for comp_nodes in components:
                nodes = list(comp_nodes)
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        key = tuple(sorted((nodes[i], nodes[j])))
                        if key in analyzed_pairs:
                            continue
                        result = self.rule_analyzer.analyze_pair(*key)
                        analyzed_pairs[key] = result
                        mapped_label = self._PRIMARY_RELATION_LABELS.get(result.overall_relation)
                        if mapped_label:
                            summary[mapped_label].append(result)
                        else:
                            summary["others"].append(result)
                        summary["all"].append(result)

    def _build_components(self, relations: List[RulePairResult]) -> List[Set[str]]:
        """根据规则对构建无向图分量。"""
        parent: Dict[str, str] = {}
        
        def find(x: str) -> str:
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(a: str, b: str):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        
        for r in relations:
            union(r.rule_a, r.rule_b)
        
        components: Dict[str, Set[str]] = {}
        for node in parent.keys():
            root = find(node)
            components.setdefault(root, set()).add(node)
        
        return list(components.values())
    
    def find_string_redundancies(self) -> List[Dict]:
        """
        基于字符串匹配查找潜在冗余（补充 Z3 分析）
        
        检测：
        1. 相同的 condition 字符串
        2. 相同的 subjects + permission
        3. 相似的 label/quote（可能是重复抽取）
        """
        from collections import defaultdict
        
        # 按 condition 分组
        by_condition = defaultdict(list)
        # 按 (subjects, permission) 分组
        by_action = defaultdict(list)
        # 按 quote 分组
        by_quote = defaultdict(list)
        
        for r in self.rules:
            rid = r["id"]
            
            # condition 分组
            cond = r.get("condition", "")
            if cond:
                # 标准化：去空格，大写
                cond_norm = cond.replace(" ", "").upper()
                by_condition[cond_norm].append(rid)
            
            # action 分组
            action = r.get("action", {})
            subjects = tuple(sorted(action.get("subjects", [])))
            perm = action.get("permission", "")
            if subjects:
                by_action[(subjects, perm)].append(rid)
            
            # quote 分组
            for prov in r.get("provenance", []):
                quote = prov.get("quote", "")
                if quote and len(quote) > 20:
                    by_quote[quote].append(rid)
        
        redundancies = []
        
        # 报告相同 condition 的规则
        for cond, rule_ids in by_condition.items():
            if len(rule_ids) > 1:
                redundancies.append({
                    "type": "same_condition",
                    "rules": rule_ids,
                    "condition": cond[:100] + "..." if len(cond) > 100 else cond,
                    "reason": f"{len(rule_ids)} rules have identical condition"
                })
        
        # 报告相同 action 的规则
        for (subjects, perm), rule_ids in by_action.items():
            if len(rule_ids) > 1:
                redundancies.append({
                    "type": "same_action",
                    "rules": rule_ids,
                    "subjects": list(subjects),
                    "permission": perm,
                    "reason": f"{len(rule_ids)} rules have same subjects and permission"
                })
        
        # 报告相同 quote 的规则
        for quote, rule_ids in by_quote.items():
            if len(rule_ids) > 1:
                redundancies.append({
                    "type": "same_quote",
                    "rules": rule_ids,
                    "quote": quote[:80] + "..." if len(quote) > 80 else quote,
                    "reason": f"{len(rule_ids)} rules derived from same quote"
                })
        
        return redundancies
    
    def generate_report(self, 
                        analyze_predicates: bool = True,
                        analyze_rules: bool = True,
                        max_rule_pairs: int = 10000,
                        lsh_similarity_threshold: float = 0.3,
                        lsh_num_bands: int = 20,
                        lsh_rows_per_band: int = 5,
                        use_lsh_filter: bool = True,
                        use_bucket_filter: bool = True) -> ConflictReport:
        """生成完整的冲突分析报告"""
        report = ConflictReport(
            total_rules=len(self.rules),
            total_predicates=len(self.predicates)
        )
        
        # 谓词分析
        if analyze_predicates:
            lattice = self.analyze_predicates()
            report.predicate_equivalences = list(lattice.equivalence_classes.values())
            report.predicate_implications = lattice.implication_edges
            report.predicate_mutex = lattice.mutex_pairs
            report.trivial_predicates = list(self.pred_analyzer.get_parse_errors().keys())
            
            # 【新增】记录不支持的谓词
            unsupported = self.pred_analyzer.get_unsupported_predicates()
            exclusive = self.pred_analyzer.get_exclusive_predicates()
            if unsupported:
                print(f"  [INFO] 跳过 {len(unsupported)} 个不支持算子的谓词 (Duration/Assess)")
            if exclusive:
                print(f"  [INFO] 跳过 {len(exclusive)} 个互斥算子谓词 (Stage/Risk)")
        
        # 规则分析
        if analyze_rules:
            print(f"\n[RuleAnalyzer] 分析 {len(self.rules)} 条规则...")
            summary = self.analyze_rules(
                focus_on=None,
                max_pairs=max_rule_pairs,
                use_lsh_filter=use_lsh_filter,
                use_bucket_filter=use_bucket_filter,
                lsh_similarity_threshold=lsh_similarity_threshold,
                lsh_num_bands=lsh_num_bands,
                lsh_rows_per_band=lsh_rows_per_band,
            )
            report.redundant_rules = summary["complete_redundancy"] + summary["contained_redundancy"]
            report.conflicting_rules = (
                summary["local_conflict"] +
                summary["intrinsic_conflict"] +
                summary["implication_conflict"] +
                summary["intrinsic_disagreement"] +
                summary["implication_disagreement"]
            )
            report.other_rule_relations = summary["others"]
            
            print(f"  完全冗余: {len(summary['complete_redundancy'])}")
            print(f"  被包含冗余: {len(summary['contained_redundancy'])}")
            print(f"  局部冲突: {len(summary['local_conflict'])}")
            print(f"  固有冲突: {len(summary['intrinsic_conflict'])}")
            print(f"  蕴含冲突: {len(summary['implication_conflict'])}")
            print(f"  固有分歧: {len(summary['intrinsic_disagreement'])}")
            print(f"  蕴含分歧: {len(summary['implication_disagreement'])}")
            print(f"  其他关系: {len(summary['others'])}")
            
            # 字符串级别的冗余检测（补充）
            string_redundancies = self.find_string_redundancies()
            print(f"  字符串匹配发现潜在冗余: {len(string_redundancies)}")
            
            # 将字符串冗余添加到建议中
            for sr in string_redundancies:
                report.merge_suggestions.append({
                    "type": f"potential_{sr['type']}",
                    "rules": sr["rules"],
                    "reason": sr["reason"]
                })
        
        # 生成建议
        self._generate_suggestions(report)
        
        return report
    
    def _generate_suggestions(self, report: ConflictReport):
        """生成处理建议"""
        # 合并建议（等价规则）
        for result in report.redundant_rules:
            if result.overall_relation == RuleRelation.COMPLETE_REDUNDANCY:
                report.merge_suggestions.append({
                    "type": "merge",
                    "rules": [result.rule_a, result.rule_b],
                    "reason": result.recommendation
                })
            elif result.overall_relation == RuleRelation.CONTAINED_REDUNDANCY:
                report.deletion_suggestions.append({
                    "type": "delete",
                    "rule": result.rule_a if "remove" in result.recommendation and result.rule_a in result.recommendation else result.rule_b,
                    "reason": result.recommendation
                })
        
        # 优先级建议（冲突规则）
        for result in report.conflicting_rules:
            report.priority_suggestions.append({
                "type": "add_priority",
                "rules": [result.rule_a, result.rule_b],
                "conflict_type": result.overall_relation.value,
                "reason": result.recommendation
            })
        
        # 特化优先级不需要建议（这是正常的关系，不是问题）
    
    def export_report(self, report: ConflictReport, output_path: str):
        """导出报告为 JSON"""
        def serialize(pairs: List[RulePairResult]) -> List[Dict[str, Any]]:
            """
            序列化规则对为字典列表。
            改动：不再输出 `recommendation`，而输出每条规则对应的 `provenance` 列表，
            字段名为 `provenance_a` / `provenance_b`。
            """
            serialized: List[Dict[str, Any]] = []
            for r in pairs:
                prov_a = []
                prov_b = []
                # 从原始规则中取 provenance（如果存在）
                rule_a_obj = next((rule for rule in self.rules if rule["id"] == r.rule_a), None)
                if rule_a_obj:
                    prov_a = rule_a_obj.get("provenance", [])
                rule_b_obj = next((rule for rule in self.rules if rule["id"] == r.rule_b), None)
                if rule_b_obj:
                    prov_b = rule_b_obj.get("provenance", [])

                serialized.append({
                    "rule_a": r.rule_a,
                    "rule_b": r.rule_b,
                    "type": r.overall_relation.value,
                    "provenance_a": prov_a,
                    "provenance_b": prov_b
                })
            return serialized
        
        complete = [r for r in report.redundant_rules if r.overall_relation == RuleRelation.COMPLETE_REDUNDANCY]
        contained = [r for r in report.redundant_rules if r.overall_relation == RuleRelation.CONTAINED_REDUNDANCY]
        local_conflict = [r for r in report.conflicting_rules if r.overall_relation == RuleRelation.LOCAL_CONFLICT]
        intrinsic_conflict = [r for r in report.conflicting_rules if r.overall_relation == RuleRelation.INTRINSIC_CONFLICT]
        implication_conflict = [r for r in report.conflicting_rules if r.overall_relation == RuleRelation.IMPLICATION_CONFLICT]
        intrinsic_disagreement = [r for r in report.conflicting_rules if r.overall_relation == RuleRelation.INTRINSIC_DISAGREEMENT]
        implication_disagreement = [r for r in report.conflicting_rules if r.overall_relation == RuleRelation.IMPLICATION_DISAGREEMENT]
        others = report.other_rule_relations
        
        data = {
            "summary": {
                "total_rules": report.total_rules,
                "total_predicates": report.total_predicates,
                "complete_redundancy": len(complete),
                "contained_redundancy": len(contained),
                "local_conflict": len(local_conflict),
                "intrinsic_conflict": len(intrinsic_conflict),
                "implication_conflict": len(implication_conflict),
                "intrinsic_disagreement": len(intrinsic_disagreement),
                "implication_disagreement": len(implication_disagreement),
                "other_relations": len(others),
            },
            "rule_analysis": {
                "complete_redundancy": serialize(complete),
                "contained_redundancy": serialize(contained),
                "local_conflict": serialize(local_conflict),
                "intrinsic_conflict": serialize(intrinsic_conflict),
                "implication_conflict": serialize(implication_conflict),
                "intrinsic_disagreement": serialize(intrinsic_disagreement),
                "implication_disagreement": serialize(implication_disagreement),
            },
            "suggestions": {
                "merge": report.merge_suggestions,
                "delete": report.deletion_suggestions,
                "priority": report.priority_suggestions,
            }
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n报告已导出到: {output_path}")


# ============ 便捷函数 ============

def analyze_knowledge_base(gen_dir: str, output_path: Optional[str] = None) -> ConflictReport:
    """
    分析知识库的一站式函数
    
    Args:
        gen_dir: gen 目录路径
        output_path: 报告输出路径（可选）
        
    Returns:
        ConflictReport
    """
    gen_path = Path(gen_dir)
    
    rules = json.load(open(gen_path / "rules.json", "r", encoding="utf-8"))
    predicates = json.load(open(gen_path / "predicates.json", "r", encoding="utf-8"))
    
    resolver = ConflictResolver(rules, predicates)
    report = resolver.generate_report()
    
    if output_path:
        resolver.export_report(report, output_path)
    
    return report


def export_predicate_relations(gen_dir: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    计算并导出谓词间的等价 / 蕴含关系，便于在规则分析时复用。
    底层复用 ConflictResolver.analyze_predicates / export_predicate_relations，避免重复实现。
    """
    gen_path = Path(gen_dir)
    predicates = json.load(open(gen_path / "predicates.json", "r", encoding="utf-8"))
    # rules 在这里并不直接参与谓词格计算，但 ConflictResolver 需要完整上下文;
    # 如果没有 rules.json，则回退到仅使用 PredicateAnalyzer。
    try:
        rules = json.load(open(gen_path / "rules.json", "r", encoding="utf-8"))
        resolver = ConflictResolver(rules, predicates)
        return resolver.export_predicate_relations(output_path=output_path)
    except FileNotFoundError:
        pa = PredicateAnalyzer(predicates)
        lattice = pa.build_lattice()
        data = {
            "equivalences": [sorted(list(eq)) for eq in lattice.equivalence_classes.values()],
            "implications": [[p, q] for (p, q) in lattice.implication_edges],
        }
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        return data

