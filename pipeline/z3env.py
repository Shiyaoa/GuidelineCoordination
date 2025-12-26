# pipeline/z3env.py
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from z3 import Real, Bool, String, RealVal, BoolVal, StringVal, Solver, Not, And, Or, sat, unsat, is_bool
import re
import json


@dataclass
class Term:
    id: str
    type: str  # number/enum/bool


@dataclass
class Predicate:
    id: str
    label: str
    formal_definition: str
    dependencies: List[str]


class Z3Env:
    """
    Z3 环境 - 支持谓词的 SMT 约束求解
    
    算子支持策略：
    - Z3_SUPPORTED_OPERATORS: 支持完整 Z3 运算（Has, On, HistoryOf, Value, Delta）
    - EXCLUSIVE_OPERATORS: 视为互斥条件，不进行偏序比较（Stage, Risk）
    - 其他算子（Duration, Assess）: 暂不支持 Z3 运算
    """
    
    # 支持完整 Z3 运算的算子
    Z3_SUPPORTED_OPERATORS = {
        "Has",       # 诊断存在性
        "On",        # 用药状态
        "HistoryOf", # 历史事件
        "Value",     # 测量值
        "Delta",     # 变化量
    }
    
    # 视为互斥条件的算子（不进行偏序比较）
    EXCLUSIVE_OPERATORS = {
        "Stage",     # 分期
        "Risk",      # 风险
    }
    
    # 暂不支持的算子
    UNSUPPORTED_OPERATORS = {
        "Duration",  # 时间持续
        "Assess",    # 临床评估
    }
    
    def __init__(self, terms: Dict[str, Term] = None):
        self.terms = terms or {}
        self.vars = {}
        self.var_types: Dict[str, str] = {}
        self._init_vars()
    
    def _init_vars(self):
        """初始化变量映射"""
        for tid, t in self.terms.items():
            safe = tid.replace(".", "_")
            if t.type == "number":
                self.vars[tid] = Real(safe)
                self.var_types[tid] = "number"
            elif t.type in ("bool", "boolean"):
                self.vars[tid] = Bool(safe)
                self.var_types[tid] = "bool"
            elif t.type == "string":
                self.vars[tid] = String(safe)
                self.var_types[tid] = "string"
            else:
                self.vars[tid] = Real(safe)
                self.var_types[tid] = "number"

    def add_term(self, term_id: str, term_type: str):
        """动态添加新的term变量"""
        safe = term_id.replace(".", "_")
        current = self.var_types.get(term_id)

        if current is not None:
            if current == term_type:
                return
            if current == "bool" and term_type in ("number", "string"):
                if term_type == "number":
                    self.vars[term_id] = Real(safe)
                elif term_type == "string":
                    self.vars[term_id] = String(safe)
                self.var_types[term_id] = term_type
            return

        if term_type == "number":
            self.vars[term_id] = Real(safe)
            self.var_types[term_id] = "number"
        elif term_type in ("bool", "boolean"):
            self.vars[term_id] = Bool(safe)
            self.var_types[term_id] = "bool"
        elif term_type == "string":
            self.vars[term_id] = String(safe)
            self.var_types[term_id] = "string"
        else:
            self.vars[term_id] = Real(safe)
            self.var_types[term_id] = "number"
    
    def _extract_operator(self, formal_definition: str) -> Optional[str]:
        """
        从 formal_definition 中提取算子类型
        
        Examples:
            "$ Has(cond.t2dm) $" -> "Has"
            "$ Value(meas.egfr) >= 45 $" -> "Value"
            "$ Stage(cond.ckd) == 3 $" -> "Stage"
        """
        for op in self.Z3_SUPPORTED_OPERATORS:
            if f"{op}(" in formal_definition:
                return op
        for op in self.EXCLUSIVE_OPERATORS:
            if f"{op}(" in formal_definition:
                return op
        for op in self.UNSUPPORTED_OPERATORS:
            if f"{op}(" in formal_definition:
                return op
        return None
    
    def get_operator_category(self, formal_definition: str) -> str:
        """
        获取算子类别：'supported' | 'exclusive' | 'unsupported' | 'unknown'
        """
        op = self._extract_operator(formal_definition)
        if op is None:
            return "unknown"
        if op in self.Z3_SUPPORTED_OPERATORS:
            return "supported"
        if op in self.EXCLUSIVE_OPERATORS:
            return "exclusive"
        if op in self.UNSUPPORTED_OPERATORS:
            return "unsupported"
        return "unknown"
    
    def is_z3_supported(self, formal_definition: str) -> bool:
        """检查 formal_definition 是否支持 Z3 运算"""
        return self.get_operator_category(formal_definition) == "supported"
    
    def is_exclusive(self, formal_definition: str) -> bool:
        """检查 formal_definition 是否是互斥条件（Stage/Risk）"""
        return self.get_operator_category(formal_definition) == "exclusive"
    
    def is_unsupported(self, formal_definition: str) -> bool:
        """检查 formal_definition 是否暂不支持 Z3 运算"""
        return self.get_operator_category(formal_definition) == "unsupported"
    
    def is_operator_compatible(self, formal_definition1: str, formal_definition2: str) -> Tuple[bool, str]:
        """
        检查两个 formal_definition 是否具有兼容的算子类型（可用于比较）
        
        Returns:
            (is_compatible, reason): (True/False, 原因说明)
        """
        cat1 = self.get_operator_category(formal_definition1)
        cat2 = self.get_operator_category(formal_definition2)
        
        # 如果任一无法识别
        if cat1 == "unknown" or cat2 == "unknown":
            return (False, "无法识别的算子类型")
        
        # 如果任一不支持
        if cat1 == "unsupported" or cat2 == "unsupported":
            return (False, "Duration/Assess 暂不支持 Z3 运算")
        
        # 如果任一是互斥算子
        if cat1 == "exclusive" or cat2 == "exclusive":
            return (False, "Stage/Risk 视为互斥条件，无法进行比较")
        
        # 算子类型必须相同
        op1 = self._extract_operator(formal_definition1)
        op2 = self._extract_operator(formal_definition2)
        if op1 != op2:
            return (False, f"算子类型不同: {op1} vs {op2}")
        
        return (True, "算子兼容")
    
    def ensure_vars_for_expr(self, formal_definition: str):
        """确保表达式中所有变量都已定义，自动推断类型"""
        s = formal_definition.strip()
        if s.startswith("$") and s.endswith("$"):
            s = s[1:-1]
        
        # 标准化逻辑运算符
        s = s.replace("\\land", " & ").replace("\\lor", " | ").replace("\\lnot", " ~ ")
        s = re.sub(r"(?i)\band\b", " & ", s)
        s = re.sub(r"(?i)\bor\b", " | ", s)
        s = re.sub(r"(?i)\bnot\b", " ~ ", s)
        
        # 查找所有变量标识符
        var_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)\b')
        potential_vars = var_pattern.findall(s)
        
        # 识别数值比较的变量（包含比较符和数字的模式）
        # 支持：var >= -30%, var < 45, var == 30, var >= 2周 等
        numeric_vars = set()
        numeric_pattern = re.compile(
            r'\b([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z0-9_]+)+)\s*(?:>=|<=|>|<|==|!=)\s*[−\-]?[\d.]+%?'
        )
        for match in numeric_pattern.finditer(s):
            numeric_vars.add(match.group(1))
        
        for var_id in potential_vars:
            if var_id in self.vars:
                continue
            
            term_def = self.terms.get(var_id)
            if term_def is not None:
                declared_type = term_def.type
                if declared_type == "number":
                    self.add_term(var_id, "number")
                elif declared_type in ("bool", "boolean"):
                    self.add_term(var_id, "bool")
                else:
                    self.add_term(var_id, "string")
                continue
            
            # 默认策略
            if var_id in numeric_vars:
                self.add_term(var_id, "number")
            elif var_id.startswith("meas.") or var_id.startswith("delta."):
                self.add_term(var_id, "number")
            else:
                self.add_term(var_id, "bool")

    def _normalize_math(self, s: str) -> str:
        """数值标准化 - 处理 LaTeX、Unicode、布尔字面量"""
        # 1) LaTeX/Unicode -> ASCII
        repl = [
            (r"\\ge", ">="), (r"\\le", "<="), (r"\\gt", ">"), (r"\\lt", "<"),
            (r"\\neq", "!="),
            ("≥", ">="), ("≤", "<="), ("≠", "!=")
        ]
        for pat, rep in repl:
            s = re.sub(pat, rep, s)
        
        # 2) true/false 标准化（大小写不敏感）
        s = re.sub(r"(?i)\btrue\b", "True", s)
        s = re.sub(r"(?i)\bfalse\b", "False", s)
        
        # 3) 孤立 '=' -> '=='
        s = re.sub(r"(?<![!<>=])=(?![=])", "==", s)
        
        # 4) 布尔变量与 True/False 比较
        var_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*(?:[._][a-zA-Z0-9_]+)*)'
        s = re.sub(var_pattern + r'\s*==\s*True', r'(\1)', s)
        s = re.sub(var_pattern + r'\s*==\s*False', r'(~ \1)', s)
        s = re.sub(r'True\s*==\s*' + var_pattern, r'(\1)', s)
        s = re.sub(r'False\s*==\s*' + var_pattern, r'(\1)', s)
        
        return s
    
    def _register_enum_constants(self, s: str) -> str:
        """
        提取并注册表达式中的枚举常量（仅处理 supported 算子）
        Stage/Risk 的枚举值视为互斥，不进行 Z3 运算
        """
        # 如果包含互斥算子，跳过枚举注册
        if "Stage(" in s or "Risk(" in s:
            self._current_enum_constants = {}
            return s
        
        defined_vars = set(self.vars.keys())
        safe_vars = {tid.replace(".", "_") for tid in defined_vars}
        
        python_keywords = {
            'and', 'or', 'not', 'if', 'else', 'elif', 'for', 'while', 'def', 'class',
            'return', 'import', 'from', 'as', 'try', 'except', 'finally', 'with',
            'yield', 'pass', 'break', 'continue', 'global', 'nonlocal', 'lambda',
            'True', 'False', 'None', 'in', 'is', 'del', 'assert', 'raise'
        }
        
        enum_pattern = re.compile(
            r'==\s*([A-Z][a-zA-Z0-9_]*)\b'
        )
        
        found_enums = set()
        for match in enum_pattern.finditer(s):
            enum_name = match.group(1)
            if enum_name in ('True', 'False'):
                continue
            found_enums.add(enum_name)
        
        enum_constants: Dict[str, Any] = {}
        
        for enum_name in found_enums:
            safe_name = f"enum_{enum_name}"
            counter = 0
            while safe_name in safe_vars or safe_name in python_keywords:
                counter += 1
                safe_name = f"enum_{enum_name}_{counter}"
            
            enum_constants[enum_name] = StringVal(enum_name)
        
        def replace_enum(match):
            enum_name = match.group(1)
            if enum_name in enum_constants:
                return f"== enum_{enum_name}"
            return match.group(0)
        
        s = enum_pattern.sub(replace_enum, s)
        
        self._current_enum_constants = enum_constants
        
        return s
    
    def _wrap_comparisons_in_parens(self, s: str) -> str:
        """给比较表达式加括号，解决运算符优先级问题"""
        comparison_pattern = re.compile(
            r'(?<![(\w])' +
            r'(' +
            r'[a-zA-Z_][a-zA-Z0-9_]*\s*(?:>=|<=|>|<|==|!=)\s*[−\-]?[\d.]+%?' +
            r'|' +
            r'[−\-]?[\d.]+%?\s*(?:>=|<=|>|<|==|!=)\s*[a-zA-Z_][a-zA-Z0-9_]*' +
            r')' +
            r'(?![)\w])'
        )
        
        prev = None
        while prev != s:
            prev = s
            s = comparison_pattern.sub(r'(\1)', s)
        return s

    def _expand_chained_comparisons(self, s: str) -> str:
        """把 a <= x < b 改为 (a <= x) & (x < b)"""
        pattern = re.compile(
            r'(?P<a>(?:\([^()]*\)|[A-Za-z0-9_\.]+))\s*'
            r'(?P<op1><=|<|>=|>)\s*'
            r'(?P<x>(?:\([^()]*\)|[A-Za-z0-9_\.]+))\s*'
            r'(?P<op2><=|<|>=|>)\s*'
            r'(?P<b>(?:\([^()]*\)|[A-Za-z0-9_\.]+))'
        )
        while True:
            m = pattern.search(s)
            if not m:
                break
            a, op1, x, op2, b = m.group('a', 'op1', 'x', 'op2', 'b')
            repl = f"( {a} {op1} {x} ) & ( {x} {op2} {b} )"
            s = s[:m.start()] + repl + s[m.end():]
        return s

    def expr(self, formal_definition: str):
        """解析 formal_definition 为 Z3 表达式"""
        s = formal_definition.strip()
        assert s.startswith("$") and s.endswith("$"), "Expression must be wrapped in $...$"
        s = s[1:-1]

        # 逻辑记号 -> 位运算
        s = s.replace("\\land", " & ").replace("\\lor", " | ").replace("\\lnot", " ~ ")
        s = re.sub(r"(?i)\band\b", " & ", s)
        s = re.sub(r"(?i)\bor\b", " | ", s)
        s = re.sub(r"(?i)\bnot\b", " ~ ", s)

        # 数学与布尔字面量标准化
        s = self._normalize_math(s)

        # 提取并注册枚举常量
        s = self._register_enum_constants(s)

        # 变量替换为安全标识符
        ns: Dict[str, Any] = {
            "Has": lambda x: x,
            "On": lambda x: x,
            "HistoryOf": lambda x: x,
            "Value": lambda x: x,
            "Duration": lambda x: x,
            "Delta": lambda x: x,
            "Stage": lambda x: x,
            "Risk": lambda x: x,
            "Assess": lambda type_str, x: x,
        }
        for tid, sym in self.vars.items():
            safe = tid.replace(".", "_")
            ns[safe] = sym
        
        if hasattr(self, '_current_enum_constants'):
            for enum_name, const in self._current_enum_constants.items():
                ns[f"enum_{enum_name}"] = const

        var_regex = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)\b')
        
        def _replace_match(match):
            tid = match.group(0)
            if tid in self.vars:
                return tid.replace(".", "_")
            return tid
            
        s = var_regex.sub(_replace_match, s)

        # 给比较表达式加括号
        s = self._wrap_comparisons_in_parens(s)
        
        # 链式比较改写
        s = self._expand_chained_comparisons(s)

        try:
            ast = eval(s, {"__builtins__": {}}, ns)
        except Exception as e:
            raise ValueError(f"Failed to parse expression: {formal_definition}. \nNormalized: {s}. \nError: {e}")
        return ast

    def assignment_constraints(self, patient_data: Dict[str, Any]):
        cons = []
        for tid, sym in self.vars.items():
            if tid in patient_data:
                val = patient_data[tid]
                if isinstance(val, bool):
                    cons.append(sym == BoolVal(val))
                else:
                    cons.append(sym == RealVal(val))
        return cons

    def check_equivalence(self, expr1_def: str, expr2_def: str) -> Tuple[bool, str]:
        """
        检查两个谓词是否等价
        
        Returns:
            (result, reason): (True/False, 原因说明)
        """
        # 1. 检查算子支持
        category1 = self.get_operator_category(expr1_def)
        category2 = self.get_operator_category(expr2_def)
        
        # 如果任一是不支持的算子
        if category1 == "unsupported" or category2 == "unsupported":
            return (False, "Duration/Assess 暂不支持 Z3 运算")
        
        # 如果任一是互斥算子
        if category1 == "exclusive" or category2 == "exclusive":
            return (False, "Stage/Risk 视为互斥条件，无法比较等价性")
        
        # 如果无法识别算子
        if category1 == "unknown" or category2 == "unknown":
            return (False, "无法识别的算子类型")
        
        # 算子类型必须相同
        op1 = self._extract_operator(expr1_def)
        op2 = self._extract_operator(expr2_def)
        if op1 != op2:
            return (False, f"算子类型不同: {op1} vs {op2}")
        
        try:
            self.ensure_vars_for_expr(expr1_def)
            self.ensure_vars_for_expr(expr2_def)
            
            expr1 = self.expr(expr1_def)
            expr2 = self.expr(expr2_def)
            
            # 确保都是布尔类型
            if not is_bool(expr1) or not is_bool(expr2):
                return (False, "表达式不是布尔类型")
            
            solver = Solver()
            xor_expr = (expr1 != expr2)
            solver.add(xor_expr)
            
            result = solver.check()
            return (result == unsat, "XOR 不可满足则等价")
        except Exception as e:
            return (False, f"解析错误: {str(e)}")

    def check_implication(self, premise_def: str, conclusion_def: str) -> Tuple[bool, str]:
        """
        检查 premise -> conclusion 是否恒成立
        
        Returns:
            (result, reason): (True/False, 原因说明)
        """
        # 1. 检查算子支持
        category_premise = self.get_operator_category(premise_def)
        category_conclusion = self.get_operator_category(conclusion_def)
        
        # 如果任一是不支持的算子
        if category_premise == "unsupported" or category_conclusion == "unsupported":
            return (False, "Duration/Assess 暂不支持 Z3 运算")
        
        # 如果任一是互斥算子
        if category_premise == "exclusive" or category_conclusion == "exclusive":
            return (False, "Stage/Risk 视为互斥条件，无法比较蕴含关系")
        
        # 如果无法识别算子
        if category_premise == "unknown" or category_conclusion == "unknown":
            return (False, "无法识别的算子类型")
        
        # 算子类型必须相同
        op_premise = self._extract_operator(premise_def)
        op_conclusion = self._extract_operator(conclusion_def)
        if op_premise != op_conclusion:
            return (False, f"算子类型不同: {op_premise} vs {op_conclusion}")
        
        try:
            self.ensure_vars_for_expr(premise_def)
            self.ensure_vars_for_expr(conclusion_def)
            
            premise = self.expr(premise_def)
            conclusion = self.expr(conclusion_def)
            
            # 确保都是布尔类型
            if not is_bool(premise) or not is_bool(conclusion):
                return (False, "表达式不是布尔类型")
            
            solver = Solver()
            solver.add(premise)
            solver.add(Not(conclusion))
            
            result = solver.check()
            return (result == unsat, "premise ∧ ¬conclusion 不可满足则蕴含成立")
        except Exception as e:
            return (False, f"解析错误: {str(e)}")
    
    def check_mutex(self, expr1_def: str, expr2_def: str) -> Tuple[bool, str]:
        """
        检查两个谓词是否互斥（P ∧ Q 不可满足）
        
        Returns:
            (result, reason): (True/False, 原因说明)
        """
        try:
            self.ensure_vars_for_expr(expr1_def)
            self.ensure_vars_for_expr(expr2_def)
            
            expr1 = self.expr(expr1_def)
            expr2 = self.expr(expr2_def)
            
            # 确保都是布尔类型
            if not is_bool(expr1) or not is_bool(expr2):
                return (False, "表达式不是布尔类型")
            
            solver = Solver()
            solver.add(expr1)
            solver.add(expr2)
            
            result = solver.check()
            return (result == unsat, "P ∧ Q 不可满足则互斥")
        except Exception as e:
            return (False, f"解析错误: {str(e)}")
