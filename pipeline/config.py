"""
配置管理模块 - 集中管理所有配置项
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """LLM配置"""
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "deepseek/deepseek-v3.1-terminus"
    temperature: float = 0.1
    max_tokens: int = 16384  # 增加输出长度限制
    timeout: float = 120.0
    max_retries: int = 3


@dataclass
class PathConfig:
    """路径配置"""
    base_dir: str = field(default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    standard_dir: str = field(default="")
    gen_dir: str = field(default="")
    
    def __post_init__(self):
        if not self.standard_dir:
            self.standard_dir = os.path.join(self.base_dir, "standard")
        if not self.gen_dir:
            self.gen_dir = os.path.join(self.base_dir, "gen")


@dataclass
class MatchConfig:
    """匹配配置"""
    term_threshold: float = 80.0
    med_threshold: float = 80.0
    predicate_fuzzy_threshold: float = 70.0
    predicate_high_confidence: float = 90.0


@dataclass
class PipelineConfig:
    """流水线总配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    
    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """从环境变量创建配置"""
        llm_config = LLMConfig(
            api_key=os.getenv("LLM_API_KEY", ""),
            base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
            model=os.getenv("LLM_MODEL", "deepseek/deepseek-v3.1-terminus"),
        )
        return cls(llm=llm_config)


# 默认配置单例
_default_config: Optional[PipelineConfig] = None


def get_config() -> PipelineConfig:
    """获取默认配置"""
    global _default_config
    if _default_config is None:
        _default_config = PipelineConfig()
    return _default_config


def set_config(config: PipelineConfig):
    """设置默认配置"""
    global _default_config
    _default_config = config

