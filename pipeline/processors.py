"""
Fuzzy匹配处理器 - 用于标准化术语、药物和谓词
"""
from typing import List, Type, Callable, TypeVar, Optional

from rapidfuzz import fuzz, process as rf_process
from pydantic import BaseModel

from .models import (
    Term, TermLabel, MedicationTerm, Predicates,
    TermList, MedicationTermList, PredicatesList
)
from .standard_library import StandardLibrary, get_standard_library
from .config import get_config, MatchConfig


T = TypeVar('T', bound=BaseModel)


def process_by_fuzzy_match(
    items: List[T],
    std_by_id: dict,
    std_by_name: dict,
    name_field: str,
    model_class: Type[T],
    field_mapping: Callable[[dict], dict],
    threshold: float = 80.0
) -> List[T]:
    """
    通用的fuzzy匹配处理函数
    
    Args:
        items: 待处理的项目列表
        std_by_id: 标准库按id索引的字典
        std_by_name: 标准库按name索引的字典
        name_field: 输入项用于匹配的字段名
        model_class: 输出的Pydantic模型类
        field_mapping: 将标准库dict映射为model字段的函数
        threshold: fuzzy匹配阈值
    
    Returns:
        处理后的项目列表
    """
    std_names = list(std_by_name.keys())
    processed = []
    
    for item in items:
        # 1. ID精确匹配
        if item.id in std_by_id:
            std = std_by_id[item.id]
            processed.append(model_class(**field_mapping(std)))
            continue
        
        # 2. Fuzzy匹配
        item_name = getattr(item, name_field, None)
        if item_name:
            match_result = rf_process.extractOne(
                item_name.lower(),
                std_names,
                scorer=fuzz.token_sort_ratio
            )
            
            if match_result and match_result[1] >= threshold:
                matched_name, score, _ = match_result
                std = std_by_name[matched_name]
                processed.append(model_class(**field_mapping(std)))
                continue
        
        # 3. 无匹配，保留原项
        processed.append(item)
    
    return processed


class TermProcessor:
    """术语处理器"""
    
    def __init__(
        self, 
        std_lib: Optional[StandardLibrary] = None,
        config: Optional[MatchConfig] = None
    ):
        self._std_lib = std_lib or get_standard_library()
        self._config = config or get_config().match
    
    @staticmethod
    def _term_mapping(std: dict) -> dict:
        """标准库dict -> Term字段映射"""
        return {
            'id': std['id'],
            'name': std['name'],
            'label': TermLabel(std['label']),
            'type': std['type'],
            'unit': std.get('unit')
        }
    
    def process(self, response: TermList) -> TermList:
        """
        处理术语列表，通过fuzzy匹配标准化
        
        Args:
            response: LLM返回的术语列表
            
        Returns:
            标准化后的术语列表
        """
        processed = process_by_fuzzy_match(
            items=response.items,
            std_by_id=self._std_lib.terms_by_id,
            std_by_name=self._std_lib.terms_by_name,
            name_field='name',
            model_class=Term,
            field_mapping=self._term_mapping,
            threshold=self._config.term_threshold
        )
        return TermList(items=processed)


class MedicationProcessor:
    """药物术语处理器"""
    
    def __init__(
        self,
        std_lib: Optional[StandardLibrary] = None,
        config: Optional[MatchConfig] = None
    ):
        self._std_lib = std_lib or get_standard_library()
        self._config = config or get_config().match
    
    @staticmethod
    def _med_mapping(std: dict) -> dict:
        """标准库dict -> MedicationTerm字段映射"""
        return {
            'id': std['id'],
            'name': std['name'],
            'drug_class': std.get('class'),
            'subclass': std.get('subclass')
        }
    
    def process(self, response: MedicationTermList) -> MedicationTermList:
        """
        处理药物术语列表，通过fuzzy匹配标准化
        
        Args:
            response: LLM返回的药物术语列表
            
        Returns:
            标准化后的药物术语列表
        """
        processed = process_by_fuzzy_match(
            items=response.items,
            std_by_id=self._std_lib.meds_by_id,
            std_by_name=self._std_lib.meds_by_name,
            name_field='name',
            model_class=MedicationTerm,
            field_mapping=self._med_mapping,
            threshold=self._config.med_threshold
        )
        return MedicationTermList(items=processed)


class PredicateProcessor:
    """谓词处理器 - 结合fuzzy匹配和Z3等价性判定"""
    
    def __init__(
        self,
        std_lib: Optional[StandardLibrary] = None,
        config: Optional[MatchConfig] = None
    ):
        self._std_lib = std_lib or get_standard_library()
        self._config = config or get_config().match
    
    def process(self, response: PredicatesList) -> PredicatesList:
        """
        处理谓词列表
        匹配策略:
        1. 首先通过id精确匹配
        2. 然后通过label进行fuzzy匹配，筛选出候选谓词
        3. 对候选谓词使用Z3进行等价性判定
        
        Args:
            response: LLM返回的谓词列表
            
        Returns:
            标准化后的谓词列表
        """
        std_labels = self._std_lib.get_predicate_labels()
        processed_items = []
        
        for pred in response.items:
            # 1. ID精确匹配
            if pred.id in self._std_lib.predicates_by_id:
                std = self._std_lib.predicates_by_id[pred.id]
                processed_items.append(self._create_predicate(std))
                continue
            
            # 2. Fuzzy匹配筛选候选
            candidates = rf_process.extract(
                pred.name.lower() if pred.name else "",
                std_labels,
                scorer=fuzz.token_sort_ratio,
                limit=5
            )
            
            matched = False
            # 3. Z3等价性判定
            for matched_label, score, _ in candidates:
                if score < self._config.predicate_fuzzy_threshold:
                    continue
                
                std = self._std_lib.predicates_by_label[matched_label]
                
                # 尝试Z3等价判定
                if pred.formal_definition and std.get('formal_definition'):
                    try:
                        if self._std_lib.z3_env.check_equivalence(
                            pred.formal_definition,
                            std['formal_definition']
                        ):
                            processed_items.append(self._create_predicate(std))
                            matched = True
                            break
                    except Exception:
                        pass
                
                # 高分fuzzy匹配直接接受
                if score >= self._config.predicate_high_confidence:
                    processed_items.append(self._create_predicate(std))
                    matched = True
                    break
            
            if not matched:
                processed_items.append(pred)
        
        return PredicatesList(items=processed_items)
    
    @staticmethod
    def _create_predicate(std: dict) -> Predicates:
        """从标准库dict创建Predicate对象"""
        return Predicates(
            id=std['id'],
            name=std.get('label', ''),
            formal_definition=std['formal_definition'],
            dependencies=std.get('dependencies')
        )


# 便捷函数
def process_terms(response: TermList) -> TermList:
    """处理术语列表"""
    return TermProcessor().process(response)


def process_med_terms(response: MedicationTermList) -> MedicationTermList:
    """处理药物术语列表"""
    return MedicationProcessor().process(response)


def process_predicates(response: PredicatesList) -> PredicatesList:
    """处理谓词列表"""
    return PredicateProcessor().process(response)

