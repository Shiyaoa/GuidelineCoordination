"""
标准库管理器 - 启动时一次性加载所有标准库到内存
"""
import os
import json
from typing import Dict, List, Optional

from .config import get_config, PipelineConfig
from .z3env import Z3Env


class StandardLibrary:
    """
    标准库内存管理器 - 单例模式
    启动时一次性加载所有标准库到内存，避免重复IO操作
    """
    _instance: Optional["StandardLibrary"] = None
    
    def __new__(cls, config: Optional[PipelineConfig] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        if self._initialized:
            return
        
        self._config = config or get_config()
        self._standard_dir = self._config.paths.standard_dir
        
        # 一次性加载所有标准库到内存
        self.terms: List[dict] = self._load_json("terms.json")
        self.meds: List[dict] = self._load_json("meds.json")
        self.predicates: List[dict] = self._load_json("predicates.json")
        
        # 构建索引（用于快速查找）
        self._build_indices()
        
        # Z3环境（懒加载）
        self._z3_env: Optional[Z3Env] = None
        
        self._initialized = True
        print(f"[StandardLibrary] 已加载: {len(self.terms)} terms, "
              f"{len(self.meds)} meds, {len(self.predicates)} predicates")
    
    def _load_json(self, filename: str) -> List[dict]:
        """加载JSON文件"""
        filepath = os.path.join(self._standard_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[StandardLibrary] 警告: 文件不存在 {filepath}")
            return []
        except json.JSONDecodeError as e:
            print(f"[StandardLibrary] 警告: JSON解析错误 {filepath}: {e}")
            return []
    
    def _build_indices(self):
        """构建索引以支持快速查找"""
        # Terms 索引
        self.terms_by_id: Dict[str, dict] = {t['id']: t for t in self.terms}
        self.terms_by_name: Dict[str, dict] = {t['name'].lower(): t for t in self.terms}
        
        # 添加别名索引
        for t in self.terms:
            for alias in t.get('aliases', []):
                self.terms_by_name[alias.lower()] = t
        
        # Meds 索引
        self.meds_by_id: Dict[str, dict] = {m['id']: m for m in self.meds}
        self.meds_by_name: Dict[str, dict] = {m['name'].lower(): m for m in self.meds}
        
        # Predicates 索引
        self.predicates_by_id: Dict[str, dict] = {p['id']: p for p in self.predicates}
        self.predicates_by_label: Dict[str, dict] = {
            p.get('label', '').lower(): p for p in self.predicates if p.get('label')
        }
    
    @property
    def z3_env(self) -> Z3Env:
        """获取Z3环境（懒加载）"""
        if self._z3_env is None:
            self._z3_env = Z3Env()
        return self._z3_env
    
    def get_term_names(self) -> List[str]:
        """获取所有术语名称（小写）"""
        return list(self.terms_by_name.keys())
    
    def get_med_names(self) -> List[str]:
        """获取所有药物名称（小写）"""
        return list(self.meds_by_name.keys())
    
    def get_predicate_labels(self) -> List[str]:
        """获取所有谓词标签（小写）"""
        return list(self.predicates_by_label.keys())
    
    @classmethod
    def reset(cls):
        """重置单例（主要用于测试）"""
        cls._instance = None


# 便捷函数
def get_standard_library(config: Optional[PipelineConfig] = None) -> StandardLibrary:
    """获取标准库单例"""
    return StandardLibrary(config)

