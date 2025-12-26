"""
临床指南知识抽取工作流 - 支持多轮输入 + LSH聚类 + 并行子图

架构设计:
- 子图 (ClusterState): 输入 cluster_id, texts, texts_formatted
- 主图 (AgentState): 输出 terms, med_terms, predicates, rules (带 operator.add reducer)
- 使用 LangGraph Send API 实现并行分发，reducer 自动完成合并
"""
from typing import Optional, Any, List, Dict, TypedDict, Annotated
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph, START
from langgraph.constants import Send

from .models import (
    AgentState, ClusterState, Provenance,
    ProvenanceList, TermList, MedicationTermList,SubmitSimplifiedRules,Predicates, ClinicalRule, PredicateExtractionBatch,ProvenanceCluster,
    merge_by_id, merge_cluster_cache_updates, _to_models, Term, MedicationTerm
)
from .processors import process_terms, process_med_terms
from .config import get_config, LLMConfig
from .lsh_cluster import lsh_cluster
from .io_utils import get_failed_task_logger


# ============ LLM 工厂 ============

from functools import lru_cache

@lru_cache(maxsize=1)
def _get_default_llm() -> Any:
    """获取默认 LLM 实例（单例，缓存复用）"""
    from langchain_openai import ChatOpenAI
    cfg = get_config().llm
    return ChatOpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
    )

def _create_llm(config: Optional[LLMConfig] = None) -> Any:
    """创建 LLM 实例（传入 config 时创建新实例，否则返回默认单例）"""
    if config is None:
        return _get_default_llm()
    
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )


def _extract_raw_output(res: Any) -> Optional[str]:
    """从 with_structured_output 的返回结果中提取原始输出文本"""
    if not isinstance(res, dict):
        return f"Type: {type(res)}, Value: {str(res)}" if res else "Empty response"
    
    raw = res.get("raw")
    if raw is None:
        return f"Raw output is None. Available keys in response: {list(res.keys())}"
    
    try:
        # 1. 如果是 AIMessage 对象 (LangChain 标准输出)
        if hasattr(raw, "content"):
            content = raw.content
            # 处理多模态或非字符串 content
            if isinstance(content, list):
                return json.dumps(content, ensure_ascii=False)
            
            # 如果 content 为空，检查是否有 tool_calls (结构化输出通常通过工具调用实现)
            if not content and hasattr(raw, "tool_calls") and raw.tool_calls:
                return json.dumps(raw.tool_calls, ensure_ascii=False)
            
            # 如果 content 为空且无工具调用，尝试获取 additional_kwargs
            if not content and hasattr(raw, "additional_kwargs"):
                return json.dumps(raw.additional_kwargs, ensure_ascii=False)
                
            return str(content) if content else "Empty message content"
        
        # 2. 如果是字典格式 (某些 provider 或转换后的格式)
        if isinstance(raw, dict):
            if "content" in raw and raw["content"]:
                return str(raw["content"])
            return json.dumps(raw, ensure_ascii=False)
            
        # 3. 兜底处理：直接转字符串
        return str(raw)
    except Exception as e:
        return f"Extraction error: {str(e)}. Raw object type: {type(raw)}"


def _invoke_structured_list(
    prompt: str,
    schema_cls,
    content: str,
    error_tag: str,
    task_info: Optional[Dict] = None,
):
    """
    通用的结构化调用辅助：给定系统prompt、输出schema和文本内容，返回schema实例或None。
    支持记录失败任务及其返回信息。
    """
    # Backwards-compatible enhanced wrapper:
    # - messages: Optional pre-built message list (if provided, prompt/content ignored)
    # - return_raw: if True, return dict {"parsed": parsed, "raw": raw, "parsing_error": ...}
    # - retry_on_parsing_error/retry_attempts: attempt a debug plain invoke when parsing fails
    def _default_messages():
        return [SystemMessage(content=prompt), HumanMessage(content=content)]

    if not content and not prompt:
        return None

    llm = _get_default_llm()
    messages = _default_messages()

    res = None
    parsed = None
    raw_output = None
    parsing_error = None
    try:
        # 使用 include_raw=True 以便在解析失败时获取原始输出
        res = llm.with_structured_output(schema_cls, include_raw=True).invoke(messages)

        parsing_error = res.get("parsing_error")
        if parsing_error:
            # 记录并且尝试抓取 raw 输出以供调试
            raw_output = _extract_raw_output(res)
            error_msg = f"解析错误: {parsing_error}"
            print(f"[{error_tag}] {error_msg}")
            if task_info:
                get_failed_task_logger().log_generic_failure(
                    stage=error_tag,
                    error=error_msg,
                    task_info=task_info,
                    node_output=res
                )
            return {"parsed": None, "raw": raw_output, "parsing_error": parsing_error}

        parsed = res.get("parsed")
        raw_output = _extract_raw_output(res)
        return {"parsed": parsed, "raw": raw_output, "parsing_error": None, "error": None}

    except Exception as e:
        error_msg = str(e)
        print(f"[{error_tag}] Error: {error_msg}")

        # 兜底方案：如果 invoke 直接抛出异常，尝试捕获 LLM 的原始反馈（plain invoke）
        node_output = "Invoke failed."
        try:
            debug_res = llm.invoke(messages)
            raw_output = _extract_raw_output({"raw": debug_res})
        except Exception as de:
            raw_output = f"Could not capture raw output: {str(de)}"

        if task_info:
            get_failed_task_logger().log_generic_failure(
                stage=error_tag,
                error=error_msg,
                task_info=task_info,
                node_output=raw_output
            )
        return {"parsed": None, "raw": raw_output, "parsing_error": error_msg, "error": error_msg}

def _safe_items(obj) -> List:
    """安全获取 .items 属性，返回空列表如果为 None"""
    if obj is None:
        return []
    items = getattr(obj, 'items', None)
    return items if items is not None else []


# ============ 系统提示词 ============

RECOMMENDATION_PROMPT = """
你是一名临床专家，给定以下的临床指南文本，你的核心任务是：**从中精确识别并提取出关于药物治疗的、可作为临床决策指令的“行动建议”条目。**

### 一、 核心识别标准：什么是“行动建议”？
一条合格的“行动建议”必须同时满足以下三个条件：
1.  **有明确的目标人群**：清晰定义了“谁”（例如：T2DM患者、CKD 3-4期患者、使用SGLT2i后eGFR下降≥30%的患者）。
2.  **有明确的临床情境或前提条件**：说明了“在什么情况下”（例如：单药治疗3个月未达标、起始胰岛素治疗时、eGFR<30 ml/min时）。
3.  **有明确的、可执行的行动指令**：包含一个表示“做什么”的动词，通常是**建议、推荐、考虑、应、可、需、避免、禁忌、起始、加用、停用、调整剂量至...** 等。

### 二、 关键区分：什么**不是**“行动建议”？（应被过滤）
请特别注意过滤以下类型的陈述，**它们不应被提取**：
-   **背景信息/事实陈述**：描述疾病、药物特性或现状，但没有给出针对性的行动指令。
    -   *示例*：“β受体阻滞剂在该类患者中耐受性更好。”（描述特性）
    -   *示例*：“SGLT2i可降低HFrEF患者的心血管死亡风险。”（陈述获益）
    -   *示例*：“老年T2DM患者常存在多种并发症。”（描述现状）
-   **治疗目标/原则声明**：说明了治疗目标或一般原则，但没有给出具体如何实现的操作指令。
    -   *示例*：“治疗需个体化。”
    -   *示例*：“应综合管理血糖、血压、血脂。”
-   **不完整的建议片段**：只说了“什么药”，没说“给谁用”或“在什么情况下用”。
    -   *修正方法*：必须从上下文中补充完整的目标人群和条件，否则不提取。

### 三、 提取与格式化要求
1.  **完整性**：一个完整的`quote`必须是一个能独立传达临床决策信息的句子或句群，**明确包含“目标人群+情境条件+行动指令”**。如果原文语句缺少其中一部分，你必须从紧邻的上下文中补充，使其完整。
2.  **原子性**：一条`quote`应对应一个独立的临床决策。如果原文是复合句（如“如果A，则做X；如果B，则做Y”），应拆分为两条独立的`quote`。
3.  **忠实性**：保持原文措辞的核心语义。补充上下文时，仅添加使建议完整所必需的最小信息。
4.  **表格处理**：如为表格，视每一行为一条独立的推荐意见进行提取，确保每行信息都构成一个完整的“行动建议”。

### 四、 输出格式
对于每一个识别出的“行动建议”，按以下格式输出：
```json
{
    "source": "指南名称（年份版）",
    "quote": "完整且包含目标人群、条件和行动指令的推荐意见文本。",
    "recommendation_grade": "推荐等级（如I, IIa, IIb, III，若无则填‘None‘）",
    "evidence_level": "证据等级（如A, B, C，若无则填‘None‘）"
}
"""

TERMS_PROMPT = """你是一名临床专家。从当前的临床指南文本中抽取非药物类型的标准术语。
## 术语类型
- measures: 实验室检验结果和生命体征（如eGFR、HbA1c、血压、BMI）
- conditions: 临床诊断和疾病（如T2DM、CKD、ASCVD、Heart failure）
- procedures: 医疗程序（如手术、造影检查）
- observations: 其他临床观察（如吸烟史、家族病史）
## 命名规范
1. id: 英文小写+下划线，格式为 `label.descriptive_name`
   - measures: meas.egfr, meas.hba1c, meas.bmi, meas.alt
   - conditions: cond.t2dm, cond.ckd, cond.hf, cond.ascvd
   - procedures: proc.contrast_imaging, proc.surgery_major
   - observations: obs.smoking_history, obs.lactate_acidosis_risk
2. name: 必须使用英文医学术语（如 "Estimated GFR", "Type 2 diabetes mellitus"）
## TermList工具的参数格式
**必须调用 TermList 工具**，提供以下格式的参数：
{
  "items": [
    {"id": "meas.egfr", "name": "Estimated GFR", "label": "measures", "type": "number"},
    {"id": "cond.renal_impairment", "name": "Renal Impairment", "label": "conditions", "type": "string"}
  ]
}

## 标准术语规范
肝肾功能不全/受损以及分级相关的标准术语只有如下选择，直接复用统一的术语，严禁根据具体分级或者分类创造新的术语：
- cond.hepatic_impairment: Hepatic Functional Severity (Child-Pugh)/Liver Disease Diagnosis/History
- cond.renal_impairment: Renal Impairment
"""

MED_TERMS_PROMPT = """你是一名临床专家。从当前的临床指南文本中抽取药物术语。
## 命名规范
1. id: 英文小写，格式为:
- 药物类别: med.class.xxx (如 med.class.sglt2i, med.class.biguanide)
- 具体药物: med.xxx (如 med.metformin, med.empagliflozin)
2. name: 必须使用英文药物通用名（如 "Metformin", "Empagliflozin", "SGLT2 inhibitors"）
## MedicationTermList工具的参数格式
**必须调用 MedicationTermList 工具**，提供以下格式的参数：
{
  "items": [
    {"id": "med.metformin", "name": "Metformin", "drug_class": "med.class.biguanide", "subclass": null}
  ]
}
"""

PREDICATES_PROMPT = """
Identify all preconditions regarding medication use in clinical guideline recommendations, including target patient populations and explicit clinical contexts. Break them down into atomic conditions and use the function to parse the logical expressions of atomic conditions. Classify each criteria into one of the following categories to select the correct Operator.

**IMPORTANT: output in English **

### Disease and Risk (Diagnosis, Risk, Staging, Stratification, Classification)
- **Presence of Disease**: If the text states a diagnosis exists.
  - **Operator**: `Has`
  - *Example*: "Patients with T2DM" -> `{"term_id": "cond.t2dm", "operator": "Has"}`
- **Staging/Grading**: If the text specifies a clinical stage, class, or grade.
  - **Operator**: `Stage`
  - *Example*: "CKD Stage 3" -> `{"term_id": "cond.ckd", "operator": "Stage", "comparison": "==", "target_value": "3"}`
  - *Example*: "liver impairment Child-Pugh Class B" -> `{"term_id": "cond.liver_impairment", "operator": "Stage", "comparison": "==", "target_value": "B"}`
- **Risk Stratification**: If the text describes a risk level category.
  - **Operator**: `Risk`
  - *Example*: "High ASCVD risk" -> `{"term_id": "cond.ascvd", "operator": "Risk", "comparison": "==", "target_value": "High"}`

### Measurements & Thresholds (Lab Values, Vital Signs, and Thresholds)
- **Lab/Vital Values**: Comparisons of physiological measurements against a number.
  - **Operator**: `Value`
  - *Example*: "eGFR < 30" -> `{"term_id": "meas.egfr", "operator": "Value", "comparison": "<", "target_value": "30"}`
- **Change/Trend**: Changes in measurements over time (increase or decrease).
  - **Operator**: `Delta`
  - *Example*: "eGFR decreased by 10" -> `{"term_id": "meas.egfr", "operator": "Delta", "comparison": "<=", "target_value": "-10"}`
  - *Example*: "Serum creatinine increased by 0.5" -> `{"term_id": "meas.serum_creatinine", "operator": "Delta", "comparison": ">=", "target_value": "0.5"}`
  - **Note**: Use positive `target_value` for increases and negative for decreases.

### Medication Context (Medication Use Background)
- **Current Use / Continue?**: Patient is already on a medication; assess whether to continue/adjust (not initial start).
  - **Operator**: `On`
  - *Example*: "On Insulin" -> `{"term_id": "med.insulin", "operator": "On"}`
  - *Example*: "Already taking ACEI, consider continue if tolerated" -> `{"term_id": "med.class.acei", "operator": "On"}`

### Event (History/Prior Events)
- **History/Prior Events**: Events happened right now or in the past.
  - **Operator**: `HistoryOf`
  - *Example*: "History of stroke" -> `{"term_id": "cond.stroke", "operator": "HistoryOf"}`

### Subjective Evaluations (Clinical Assessments)
  - **Operator**: `Assess`, typically used for subjective evaluations such as symptoms, tolerability, control status, etc. Use only when the objective operators above are not applicable.
  - *Example*: "Symptomatic Heart Failure" -> `{"term_id": "cond.hf", "operator": "Assess", "assess_type": "symptomatic", "target_value": true}`
  - *Example*: "Intolerant to Statin" -> `{"term_id": "med.class.statin", "operator": "Assess", "assess_type": "intolerant", "target_value": true}`
  - *Example*: "Poor glycemic control" -> `{"term_id": "cond.diabetes", "operator": "Assess", "assess_type": "control", "target_value": "Poor"}`
  - *Example*: "Insufficient glycemic control" -> `{"term_id": "cond.diabetes", "operator": "Assess", "assess_type": "Control", "target_value": "Insufficient"}`


**必须调用 PredicateExtractionBatch 工具**，提供以下格式的参数：
{
  "atoms": [
    {"term_id": "cond.t2dm", "operator": "Has", "text": "Patients with T2DM"},
    {"term_id": "meas.egfr", "operator": "Value", "comparison": "<", "target_value": "30", "text": "eGFR < 30"},
    {"term_id": "cond.hf", "operator": "Assess", "assess_type": "symptomatic", "target_value": true, "text": "Symptomatic Heart Failure"},
    {"term_id": "med.drug", "operator": "Assess", "assess_type": "intolerant", "target_value": true, "text": "Drug intolerance"}
  ]
}
"""


RULES_PROMPT = """你是一名临床专家，负责将临床指南中的quote行动指令转化为原子化的、形式化的决策规则。
你的核心任务是：正确引用提供的原子谓词来构建条件，并精确映射行动，确保规则的正确性和完整性。

## 规则原子化原则
- 一条规则对应一个决策
- 复合推荐应拆分为多条原子规则（不同条件或不同决策）
- 联合用药是一个决策，subjects 包含多个药物

输入包括：多个临床指南引用文本（每条都有唯一的ID）。相关原子谓词列表，相关药物术语列表。

### 条件完整性
规则的 `condition` 必须完整包含原文中所有限定患者群体和情境的关键信息，引用提供的标准谓词id，通过逻辑运算符组合。

### 行动完整性
`action` 必须完整反映原文操作：
1.  `subjects`：必须与原文提到的药物/方案完全一致，不得添加或减少。使用提供的标准化药物ID。
2.  `requirements`：必须包含原文明确要求的所有伴随操作，如监测、筛查、转诊、剂量目标。

### 处理药物列表中的“或”关系
当行动涉及一个由“或”、“包括”、“如”连接的**药物选项列表**时，必须拆分为多条原子规则：
- 保持 `condition` 和 `permission` 不变。
- 将 `subjects` 列表中的每个药物拆分成独立规则。
- 共享原有的 `requirements`。

## Permission 值规范（必须使用以下值之一）

### 是否使用
| 值 | 语义 | 适用场景 |
|---| --- | --- |
| recommend | 推荐使用 | 优先选择，有循证支持 |
| require | 必须使用 | 强制要求，无替代 |
| allow | 允许使用 | 可选方案，无特别推荐 |
| consider | 考虑使用 | 考虑使用，无特别推荐 |
| caution | 慎用 | 可用但需监测，有风险 |
| avoid | 避免使用 | 有更好替代，不推荐 |
| contraindicate | 禁忌 | 绝对禁止，有严重风险 |

### 使用状态调整
| 值 | 语义 | 适用场景 |
|---| --- | --- |
| continue | 继续使用 | 已有治疗，继续使用 |
| stop | 停止使用 | 已有治疗，停止使用 |

### 剂量调整类
| 值 | 语义 | 适用场景 |
|---| --- | --- |
| reduce_dose | 减量 | 需要降低剂量，如肾功能下降时 |
| increase_dose | 加量 | 需要增加剂量，如血糖控制不佳时 |
| start_low_dose | 起始低剂量 | 从低剂量开始，逐步滴定 |
| max_dose_limit | 限制最大剂量 | 不超过某一剂量上限 |
| titrate | 滴定调整 | 根据疗效/耐受逐步调整 |

**注意**：`permission` 映射是规则正确性的核心。必须仔细辨别原文语气强度。

## 格式要求
1. id 必须以 'rule.' 开头
2. condition 忠实地引用标准谓词id（用 AND/OR/NOT 连接）,不能遗漏
3. action.subjects 是药物id列表（必须用复数形式 subjects）
4. action.permission 必须是上述Permission值规范值之一
5. 每条规则必须有 condition（不能为空）
6. **必须**包含 source_ids字段，值为单个输入中对应的 `quote_id`（如 "q1"），这是必需字段，用于关联provenance信息。每条规则只能引用一个quote，如果需要引用多个quote的内容，请拆分为多条规则。缺少此字段将导致规则无效。


**必须调用 SubmitSimplifiedRules 工具**，提供以下格式的参数：
{
  "rules": [
    {
      "id": "rule.egfr_ge_20_hf_or_ckd_empagliflozin_no_dose_adjustment",
      "label": "HF或CKD患者eGFR≥20时恩格列净无需调整剂量",
      "condition": "(pred.Has.cond.hf OR pred.Has.cond.ckd) AND pred.Value.meas.egfr.ge.20",
      "action": {
        "subjects": ["med.empagliflozin"],
        "permission": "allow",
        "requirements": []
      },
      "source_ids": "q1"
    }
  ]
}
  
"""



# ============ 主图节点函数 ============

def extract_recommendation(state: AgentState) -> Dict:
    """
    主图节点：从指南文本中抽取推荐意见
    兼容两种输入：
    - 直接传入 text/text_idx
    - 传入 messages（旧接口）
    """
    text = state.get("text")
    text_idx = state.get("text_idx", 0)
    messages = list(state.get("messages", []))

    if not messages and text:
        messages = [HumanMessage(content=text)]

    if not messages:
        print(f"[extract {text_idx}] 无可用文本")
        return {"provenance_buffer": []}

    llm = _get_default_llm()
    system_message = SystemMessage(content=RECOMMENDATION_PROMPT)
    message_with_system = [system_message] + messages

    res = None
    try:
        # 使用 include_raw=True 以便记录错误和原始响应
        res = llm.with_structured_output(ProvenanceList, include_raw=True).invoke(message_with_system)
        
        if res.get("parsing_error"):
            error_msg = f"解析错误: {res['parsing_error']}"
            raw_output = _extract_raw_output(res)
            get_failed_task_logger().log_failed_extraction(
                text_idx=text_idx, 
                text=text or str(messages[-1]), 
                error=error_msg,
                node_output=raw_output
            )
            return {"provenance_buffer": []}
            
        provenances = res.get("parsed").items if res.get("parsed") else []
        print(f"  [extract {text_idx}] 抽取到 {len(provenances)} 条推荐意见")
        return {"provenance_buffer": provenances}

    except Exception as e:
        error_msg = str(e)
        raw_output = "Invoke failed."
        if res:
            raw_output = _extract_raw_output(res)
        else:
            try:
                debug_res = llm.invoke(message_with_system)
                raw_output = _extract_raw_output({"raw": debug_res})
            except Exception as de:
                raw_output = f"Could not capture raw output: {str(de)}"
                
        get_failed_task_logger().log_failed_extraction(
            text_idx=text_idx, 
            text=text or str(messages[-1]), 
            error=error_msg,
            node_output=raw_output
        )
        return {"provenance_buffer": []}

def distribute_texts(state: AgentState) -> List[Send]:
    """
    第一阶段的分发节点
    将多个输入文本分发到并行的 extract_recommendation 节点
    """
    input_texts = state.get("input_texts", [])
    
    # 如果只有单条消息（兼容旧接口）
    if not input_texts:
        messages = state.get("messages", [])
        if messages:
            # 从消息中提取文本
            text = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
            input_texts = [text]
    
    if not input_texts:
        print("[distribute_texts] 无输入文本")
        return []
    
    sends = []
    for idx, text in enumerate(input_texts):
        sends.append(Send("extract_recommendation", {"text": text, "text_idx": idx}))
    
    print(f"[distribute_texts] 分发 {len(sends)} 个文本到并行抽取")
    return sends


# ============ 主图节点函数，LSH聚类 ============

def do_lsh_clustering(state: AgentState) -> AgentState:
    """对推荐意见进行 LSH 聚类"""
    provenances = state.get("provenance_buffer", [])
    if not provenances:
        print("[lsh_clustering] 无推荐意见，跳过聚类")
        return {"clusters": [], "lsh_bucket_index": {}}
    
    lsh_result = lsh_cluster(provenances)
    print(f"[lsh_clustering] {len(provenances)} 条推荐 -> {len(lsh_result.clusters)} 个聚类")
    # 保存bucket索引，后续Z3验证时可用于限制验证范围
    return {
        "clusters": lsh_result.clusters,
        "lsh_bucket_index": lsh_result.bucket_index
    }



# ============ 子图节点函数 ============

def subgraph_extract_terms(state: ClusterState) -> dict:
    """Extract standard terms from cluster texts using LLM."""
    cluster_id = state.get("cluster_id")
    content = "\n\n".join(state.get("texts_formatted", []))
    if not content:
        return {"terms": []}

    response = _invoke_structured_list(
        prompt=TERMS_PROMPT,
        schema_cls=TermList,
        content=content,
        error_tag="extract_terms",
        task_info={"cluster_id": cluster_id}
    )
    if not response:
        return {"terms": []}

    parsed = response.get("parsed")
    if not parsed:
        return {"terms": []}

    standard_terms = process_terms(parsed)
    return {"terms": _safe_items(standard_terms)}


def subgraph_extract_med_terms(state: ClusterState) -> dict:
    """Extract medication terms from cluster texts using LLM."""
    cluster_id = state.get("cluster_id")
    content = "\n\n".join(state.get("texts_formatted", []))
    if not content:
        return {"med_terms": []}

    response = _invoke_structured_list(
        prompt=MED_TERMS_PROMPT,
        schema_cls=MedicationTermList,
        content=content,
        error_tag="extract_med_terms",
        task_info={"cluster_id": cluster_id}
    )
    if not response:
        return {"med_terms": []}

    parsed = response.get("parsed")
    if not parsed:
        return {"med_terms": []}

    standard_med_terms = process_med_terms(parsed)
    return {"med_terms": _safe_items(standard_med_terms)}


# ============ Predicate Agent ============

class PredicateAgentState(TypedDict):
    """
    Predicate Agent State: Implements the Symbolic Operator Layer F(E × O) → L
    
    This agent extracts atomic conditions from clinical text and formalizes them
    as logical predicates using the operator layer, which are then compiled into
    SMT constraints for Z3 verification.
    """
    cluster_id: Optional[int]  # For logging
    content: str  # Clinical text input
    terms_context: str  # Available ontological entities (conditions, measures)
    med_terms_context: str  # Available medication entities
    atoms: List[dict]  # Atomic conditions (E × O pairs before formalization)
    predicates: Annotated[List[Predicates], merge_by_id]  # Formalized predicates (L)


class RuleAgentState(TypedDict):
    """
    Rule Agent State: cluster-level rule extraction subgraph state
    """
    cluster_id: Optional[int]
    content: str
    predicates_context: str
    med_terms_context: str
    quotes_map: Dict[str, Dict]
    fragments: List[dict]
    rules: Annotated[List[ClinicalRule], merge_by_id]


def extract_predicate_atoms(state: PredicateAgentState) -> dict:
    """
    Extract predicate atoms (cluster-level).

    Args:
        state: PredicateAgentState containing cluster_id, content, terms_context, med_terms_context.

    Returns:
        dict with key "atoms" -> list of atomic condition dicts.
    """
    content = state.get("content", "")
    terms_context = state.get("terms_context", "")
    med_terms_context = state.get("med_terms_context", "")
    cluster_id = state.get("cluster_id")
    
    if not content:
        return {"atoms": []}
    
    # 构建包含术语上下文的完整内容
    full_content = content
    if terms_context:
        full_content += f"\n\nAvailable standard terms:\n{terms_context}"
    if med_terms_context:
        full_content += f"\n\nAvailable standard medications:\n{med_terms_context}"
    
    system_msg = SystemMessage(content=PREDICATES_PROMPT)
    
    
    # Use centralized structured invocation helper
    res = _invoke_structured_list(
        prompt=PREDICATES_PROMPT,
        schema_cls=PredicateExtractionBatch,
        content=full_content,
        error_tag="predicate_agent_extract_atoms",
        task_info={"cluster_id": cluster_id}
    )

    if not res:
        return {"atoms": []}

    parsed = res.get("parsed")
    if not parsed:
        return {"atoms": []}

    atoms = getattr(parsed, "atoms", []) or []
    out_atoms = []
    for atom in atoms:
        out_atoms.append(atom.model_dump() if hasattr(atom, "model_dump") else atom)
    return {"atoms": out_atoms}


def _normalize_term(term: str) -> str:
    """
    规范化 term ID，转换为 LLM 友好的点号分隔格式
    
    Args:
        term: 原始 term ID，如 "meas.egfr", "cond.hf__", "med.class.beta_blocker"
    
    Returns:
        规范化后的 term，如 "meas.egfr", "cond.hf", "med.class.beta_blocker"
    """
    if not term:
        return ""
    # 清理末尾的下划线（常见于 Has/On 类型的 term）
    term = term.rstrip("_")
    # term 通常已经有点号分隔（如 meas.egfr），只需确保格式一致
    # 如果 term 中有下划线但没有点号，将下划线转换为点号
    if "." not in term and "_" in term:
        term = term.replace("_", ".")
    return term


def _normalize_value(val: Any) -> str:
    """
    规范化值，转换为 LLM 友好的 ID 格式
    
    Args:
        val: 原始值，可能是数字、字符串、布尔值等
    
    Returns:
        规范化后的值字符串，如 "20", "-25%", "TargetOrMaxTolerated", "true"
    """
    if isinstance(val, bool):
        return "true" if val else "false"
    
    val_str = str(val)
    # 处理特殊字符：空格转为下划线，其他特殊字符保持不变（如 %, -, + 等）
    val_str = val_str.replace(" ", "_")
    # 移除可能存在的引号
    val_str = val_str.strip("'\"")
    return val_str


def _normalize_comparison(cmp: str, default: str = "==") -> str:
    """
    规范化比较符，转换为 LLM 友好的缩写形式
    
    Args:
        cmp: 原始比较符，如 ">=", "<=", "==" 等
        default: 默认比较符（当 cmp 为空时）
    
    Returns:
        规范化后的比较符缩写，如 "ge", "le", "eq" 等
    """
    cmp_map = {
        ">=": "ge",  # greater or equal
        "<=": "le",  # less or equal
        ">": "gt",   # greater than
        "<": "lt",   # less than
        "==": "eq",  # equal
        "!=": "ne"   # not equal
    }
    cmp = cmp or default
    return cmp_map.get(cmp, cmp)


def assemble_predicate(atom: dict) -> dict:
    """
    Assemble a single atomic condition into a Predicates model.

    Args:
        atom: dict describing an atomic condition (text, operator, term_id, comparison, target_value, etc.)

    Returns:
        dict with key "predicates" -> list containing one Predicates instance.
    """
    text = atom.get("text", "")
    op_type = atom.get("operator", "")
    term = atom.get("term_id", "")
    cmp = atom.get("comparison", "")
    val = atom.get("target_value", "")
    
    # 统一算子名称：History -> HistoryOf（与 models.py 保持一致）
    if op_type == "History":
        op_type = "HistoryOf"
    
    # 规范化 term
    term_normalized = _normalize_term(term)
    
    # 分发逻辑 (Dispatcher)
    if op_type in ["Has", "On", "HistoryOf"]:
        # Bool 开关类，无需比较符
        # 格式: pred.Has.cond.hf
        formal_def = f"$ {op_type}({term}) $"
        pred_id = f"pred.{op_type}.{term_normalized}"
        
    elif op_type in ["Value", "Duration", "Delta", "Stage", "Risk"]:
        # 需要比较符和值，缺失则容错处理
        # 格式: pred.Value.meas.egfr.ge.20 或 pred.Delta.meas.egfr.le.-25%
        # 如果值缺失，不应盲目默认为 True（会改变数值比较语义）。
        # 将缺省值置为 None（由后续修复 agent 补全）；在构造 ID 时使用占位符 "UNSPECIFIED"。
        if val == "":
            val = None

        cmp_normalized = _normalize_comparison(cmp, default="==")
        if val is None:
            val_normalized = "UNSPECIFIED"
            formal_def = f"$ {op_type}({term}) {cmp or '=='} UNSPECIFIED $"
        else:
            val_normalized = _normalize_value(val)
            formal_def = f"$ {op_type}({term}) {cmp or '=='} {val} $"

        pred_id = f"pred.{op_type}.{term_normalized}.{cmp_normalized}.{val_normalized}"
        
    elif op_type == "Assess":
        # Assess("Intolerance", med.acei) == True
        # 格式: pred.Assess.DosageStatus.med.class.beta_blocker.eq.TargetOrMaxTolerated
        assess_type = atom.get("assess_type", "Status")
        # Assess 可能期待字符串标签或枚举（非布尔）。若缺失则设为 None，等待后续 agent 修复。
        if val == "":
            val = None

        cmp_normalized = _normalize_comparison(cmp, default="==")
        if val is None:
            val_normalized = "UNSPECIFIED"
            formal_def = f"$ Assess('{assess_type}', {term}) {cmp or '=='} UNSPECIFIED $"
        else:
            val_normalized = _normalize_value(val)
            formal_def = f"$ Assess('{assess_type}', {term}) {cmp or '=='} {val} $"

        pred_id = f"pred.Assess.{assess_type}.{term_normalized}.{cmp_normalized}.{val_normalized}"
        
    else:
        # 兜底：直接输出
        if cmp and val:
            cmp_normalized = _normalize_comparison(cmp, default="==")
            val_normalized = _normalize_value(val)
            formal_def = f"$ {term} {cmp} {val} $"
            pred_id = f"pred.{op_type}.{term_normalized}.{cmp_normalized}.{val_normalized}"
        else:
            formal_def = f"$ {term} $"
            pred_id = f"pred.{op_type}.{term_normalized}" 
    
    pred = Predicates(
        id=pred_id,
        name=text,
        formal_definition=formal_def,
        dependencies=[term] if term else None,
    )
    
    return {"predicates": [pred]}


def distribute_predicates(state: PredicateAgentState) -> List[Send]:
    """Distribute predicate atoms to assemble nodes in parallel."""
    atoms = state.get("atoms", [])
    sends = []
    for atom in atoms:
        sends.append(Send("predicate_agent_assemble_predicate", atom))
    return sends


def build_predicate_subgraph():
    """
    Build the predicate extraction subgraph.
    
    Workflow:
    START → extract_atoms (E × O extraction)
         → distribute (parallelization)
         → assemble_predicate (F(E, O) → L formalization)
         → END
    
    This subgraph implements the Symbolic Operator Layer, bridging the semantic gap
    between ontological entities and logical predicates for SMT-based verification.
    """
    builder = StateGraph(PredicateAgentState)

    builder.add_node("predicate_agent_extract_atoms", extract_predicate_atoms)
    builder.add_node("predicate_agent_assemble_predicate", assemble_predicate)

    builder.add_edge(START, "predicate_agent_extract_atoms")
    builder.add_conditional_edges(
        "predicate_agent_extract_atoms",
        distribute_predicates,
        ["predicate_agent_assemble_predicate"]
    )
    builder.add_edge("predicate_agent_assemble_predicate", END)

    return builder.compile()


def extract_predicates_subgraph(state: ClusterState) -> dict:
    """Run the predicate extraction subgraph for a cluster and return predicates."""
    content = "\n\n".join(state.get("texts_formatted", []))
    terms = state.get("terms", []) or []
    med_terms = state.get("med_terms", []) or []
    cluster_id = state.get("cluster_id")

    if not content:
        return {"predicates": []}

    # Build terms/med terms context
    terms_content = "\n".join([f"id: {term.id} name: {term.name}" for term in terms])
    med_terms_content = "\n".join([f"id: {med.id} name: {med.name}" for med in med_terms])

    predicate_state = {
        "cluster_id": cluster_id,
        "content": content,
        "terms_context": terms_content,
        "med_terms_context": med_terms_content,
        "atoms": [],
        "predicates": []
    }

    try:
        predicate_agent = build_predicate_subgraph()
        result = predicate_agent.invoke(predicate_state)
        predicates = result.get("predicates", [])
        return {"predicates": predicates}
    except Exception as e:
        print(f"[extract_predicates] Error: {e}")
        get_failed_task_logger().log_generic_failure(
            stage="extract_predicates",
            error=str(e),
            task_info={"cluster_id": cluster_id}
        )
        return {"predicates": []}

# ============ rules node ============

def extract_rule_fragments(state: RuleAgentState) -> dict:
    """Extract simplified rule fragments for a cluster using LLM structured output."""
    
    content = state.get("content", "")
    cluster_id = state.get("cluster_id")

    if not content:
        return {"fragments": []}

    # content 已经包含了完整的上下文
    full_content = content

    # Use centralized structured invocation helper
    res = _invoke_structured_list(
        prompt=RULES_PROMPT,
        schema_cls=SubmitSimplifiedRules,
        content=full_content,
        error_tag="rule_agent_extract_fragments",
        task_info={"cluster_id": cluster_id}
    )

    if not res:
        return {"fragments": []}

    parsed = res.get("parsed")
    if not parsed:
        return {"fragments": []}

    items = getattr(parsed, "rules", []) or []
    fragments = [item.model_dump() if hasattr(item, "model_dump") else item for item in items]
    return {"fragments": fragments}


def distribute_rule_fragments(state: RuleAgentState) -> List[Send]:
    """Distribute simplified fragments to assemble nodes in parallel."""
    fragments = state.get("fragments", []) or []
    quotes_map = state.get("quotes_map", {}) or {}
    cluster_id = state.get("cluster_id")
    sends = []
    for frag in fragments:
        payload = {"fragment": frag, "quotes_map": quotes_map, "cluster_id": cluster_id}
        sends.append(Send("rule_agent_assemble_rule", payload))
    return sends


def assemble_rule_fragment(payload: dict) -> dict:
    """Assemble a single simplified fragment into a ClinicalRule with provenance backfilled."""
    fragment = payload.get("fragment", {}) if isinstance(payload, dict) else payload
    quotes_map = payload.get("quotes_map", {}) if isinstance(payload, dict) else {}
    cluster_id = payload.get("cluster_id")


    try:
        rule_id = fragment.get("id") or fragment.get("rule_id")
        label = fragment.get("label", "")
        condition = fragment.get("condition", "")
        action = fragment.get("action", {})
        # 处理source_ids字段 - 必须是有效的字符串key
        source_id = fragment.get("source_ids")

        if not source_id or not isinstance(source_id, str) or not source_id.strip():
            # source_ids字段缺失、无效或为空，这是系统错误
            error_msg = f"[assemble_rule_fragment] CRITICAL ERROR: Rule {rule_id} has invalid source_ids field: {source_id}"
            print(error_msg)
            print(f"[assemble_rule_fragment] Fragment: {fragment}")
            print(f"[assemble_rule_fragment] Available quotes_map keys: {list(quotes_map.keys())}")

            # 记录失败任务
            
            get_failed_task_logger().log_generic_failure(
                stage="assemble_rule_fragment",
                error=f"Invalid source_ids '{source_id}' for rule {rule_id}",
                task_info={"cluster_id": cluster_id, "rule_id": rule_id, "source_id": source_id, "fragment": fragment}
            )
            return {"rules": []}

        # 验证source_id(s)是否存在于quotes_map中，支持单个或组合的source_ids（如'q5,q7'）
        source_ids_list = [s.strip() for s in source_id.split(',') if s.strip()]

        # 验证所有source_ids都存在
        missing_ids = [sid for sid in source_ids_list if sid not in quotes_map]
        if missing_ids:
            error_msg = f"[assemble_rule_fragment] CRITICAL ERROR: source_id(s) '{','.join(missing_ids)}' not found in quotes_map for rule {rule_id}"
            print(error_msg)
            print(f"[assemble_rule_fragment] Available keys: {list(quotes_map.keys())}")


            get_failed_task_logger().log_generic_failure(
                stage="assemble_rule_fragment",
                error=f"source_id(s) '{','.join(missing_ids)}' not in quotes_map for rule {rule_id}",
                task_info={"cluster_id": cluster_id, "rule_id": rule_id, "source_id": source_id, "missing_ids": missing_ids, "available_keys": list(quotes_map.keys())}
            )
            return {"rules": []}

        # 为每个source_id创建provenance
        provenance_list = []
        for sid in source_ids_list:
            prov_dict = quotes_map[sid]
            provenance_list.append(Provenance(**prov_dict))

        cr = ClinicalRule(
            id=rule_id,
            label=label,
            condition=condition,
            action=action,
            provenance=provenance_list,
        )
        return {"rules": [cr]}
    except Exception as e:
        print(f"[rule_agent_assemble_rule] Failed to assemble fragment {fragment}: {e}")
        return {"rules": []}


def build_rule_subgraph():
    """Build the rule extraction subgraph (extract -> distribute -> assemble)."""
    builder = StateGraph(RuleAgentState)
    builder.add_node("rule_agent_extract_fragments", extract_rule_fragments)
    builder.add_node("rule_agent_assemble_rule", assemble_rule_fragment)

    builder.add_edge(START, "rule_agent_extract_fragments")
    builder.add_conditional_edges(
        "rule_agent_extract_fragments",
        distribute_rule_fragments,
        ["rule_agent_assemble_rule"]
    )
    builder.add_edge("rule_agent_assemble_rule", END)
    return builder.compile()



def extract_rules_subgraph(state: ClusterState) -> dict:
    """Cluster-level rule extraction: build quotes map, call rule subgraph and return rules."""
    cluster_id = state.get("cluster_id")
    provenances = state.get('provenances', []) or []
    predicates = state.get("predicates", []) or []
    med_terms = state.get("med_terms", []) or []
    
    # 动态为每个 provenance 生成 quote_id（q1, q2, ...），并构建引用映射用于回填
    quotes_map = {}
    quotes_content_lines = []
    for idx, p in enumerate(provenances):
        qid = f"q{idx+1}"
        quotes_map[qid] = {
            "source": getattr(p, "source", None) if hasattr(p, "source") else p.get("source"),
            "quote": getattr(p, "quote", None) if hasattr(p, "quote") else p.get("quote"),
            "recommendation_grade": getattr(p, "recommendation_grade", None) if hasattr(p, "recommendation_grade") else p.get("recommendation_grade"),
            "evidence_level": getattr(p, "evidence_level", None) if hasattr(p, "evidence_level") else p.get("evidence_level"),
        }

        quotes_content_lines.append(
            f"id: {qid}\nQuote: {quotes_map[qid]['quote']}\n"
        )
    # cluster-level content (原始拼接文本)

    predicates_content = "\n".join([f"id: {pred.id} formal_def: {pred.formal_definition}" for pred in predicates])
    med_terms_content = "\n".join([f"id: {med.id} name: {med.name}" for med in med_terms])

    # 在 prompt 中显式传入 quote_id 映射，指示 LLM 在输出规则时只使用 quote_id 作为 provenance 引用
    quotes_content = "\n\n".join(quotes_content_lines)
    prompt_body = (
        f"Quote IDs and texts:\n{quotes_content}\n\n"
        f"Available predicates:\n{predicates_content}\n\n"
        f"Available medications:\n{med_terms_content}\n\n"
    )

    # 使用 Rule Agent 子图（extract_fragments -> distribute -> assemble_rule）并行化处理
    rule_agent = build_rule_subgraph()
    # 把 prompt_body 放入 content 以便 LLM 看到 quote_id 映射和上下文（子节点会再次构建需要的上下文）
    rule_state = {
        "cluster_id": cluster_id,
        "content": prompt_body,
        "quotes_map": quotes_map,
        "fragments": [],
        "rules": []
    }
    try:
        result = rule_agent.invoke(rule_state)
        rules = result.get("rules", [])
        return {"rules": rules}
    except Exception as e:
        print(f"[extract_rules_subgraph] Error invoking rule agent subgraph: {e}")
        get_failed_task_logger().log_generic_failure(
            stage="extract_rules",
            error=str(e),
            task_info={"cluster_id": cluster_id}
        )
        return {"rules": []}



# ============ 图构建 ============

def build_extraction_subgraph():
    builder = StateGraph(ClusterState)
    builder.add_node("extract_terms", subgraph_extract_terms)
    builder.add_node("extract_med_terms", subgraph_extract_med_terms)
    builder.add_node("extract_predicates", extract_predicates_subgraph)
    builder.add_node("extract_rules", extract_rules_subgraph)
    builder.add_edge(START, "extract_terms")
    builder.add_edge(START, "extract_med_terms")
    builder.add_edge(["extract_terms","extract_med_terms"], "extract_predicates")
    builder.add_edge(["extract_predicates","extract_med_terms"], "extract_rules")
    builder.add_edge("extract_rules", END)
    return builder.compile()

CLUSTER_SUBGRAPH = build_extraction_subgraph()

# ============ 智能缓存复用的处理节点 ============
# 节点函数现在直接实现缓存复用逻辑，不再需要预定义子图

def process_cluster(state: dict) -> dict:
    # state 是 Send 传入的 cluster 局部 dict
    result = CLUSTER_SUBGRAPH.invoke(state)

    return {
        "terms": result.get("terms", []),
        "med_terms": result.get("med_terms", []),
        "predicates": result.get("predicates", []),
        "rules": result.get("rules", []),
    }

def process_cluster_terms(state: dict) -> dict:
    """处理单个 cluster 的术语抽取，智能复用缓存"""
    cluster_id = state.get("cluster_id")

    # 检查缓存中是否已有数据
    terms = state.get("terms", [])
    med_terms = state.get("med_terms", [])

    # 只有在没有缓存数据时才重新抽取
    if not terms:
        terms_result = subgraph_extract_terms(state)
        terms = terms_result.get("terms", [])

    if not med_terms:
        med_terms_result = subgraph_extract_med_terms(state)
        med_terms = med_terms_result.get("med_terms", [])

    print(f"[process_cluster_terms][cluster {cluster_id}] terms={len(terms)} med_terms={len(med_terms)}")

    # 返回全局聚合结果 + cluster-specific 缓存更新
    return {

        "cluster_cache_updates": {
            cluster_id: {
                "terms": terms,
                "med_terms": med_terms,
            }
        }
    }

def process_cluster_predicates(state: dict) -> dict:
    """处理单个 cluster 的谓词抽取，复用已有的术语"""
    cluster_id = state.get("cluster_id")

    # 复用已有的 terms/med_terms
    terms = state.get("terms", [])
    med_terms = state.get("med_terms", [])

    # 如果没有缓存，才重新抽取
    if not terms:
        terms_result = subgraph_extract_terms(state)
        terms = terms_result.get("terms", [])

    if not med_terms:
        med_terms_result = subgraph_extract_med_terms(state)
        med_terms = med_terms_result.get("med_terms", [])

    # 抽取谓词
    predicates_result = extract_predicates_subgraph({**state, "terms": terms, "med_terms": med_terms})
    predicates = predicates_result.get("predicates", [])

    print(f"[process_cluster_predicates][cluster {cluster_id}] preds={len(predicates)} (复用缓存: terms={bool(state.get('terms'))}, meds={bool(state.get('med_terms'))})")

    # 返回全局聚合结果 + cluster-specific 缓存更新
    return {

        "cluster_cache_updates": {
            cluster_id: {
                "terms": terms,
                "med_terms": med_terms,
                "predicates": predicates,
            }
        }
    }

def process_cluster_rules(state: dict) -> dict:
    """处理单个 cluster 的规则抽取，复用所有已有数据"""
    # 复用已有的所有数据
    terms = state.get("terms", [])
    med_terms = state.get("med_terms", [])
    predicates = state.get("predicates", [])
    cluster_id = state.get("cluster_id")
    # 如果没有缓存，才重新抽取
    if not terms:
        terms_result = subgraph_extract_terms(state)
        terms = terms_result.get("terms", [])

    if not med_terms:
        med_terms_result = subgraph_extract_med_terms(state)
        med_terms = med_terms_result.get("med_terms", [])

    if not predicates:
        predicates_result = extract_predicates_subgraph({**state, "terms": terms, "med_terms": med_terms})
        predicates = predicates_result.get("predicates", [])

    # 抽取规则
    rules_result = extract_rules_subgraph({**state, "med_terms": med_terms, "predicates": predicates})
    rules = rules_result.get("rules", [])

    print(f"[process_cluster_rules][cluster {state.get('cluster_id')}] rules={len(rules)} (复用缓存: terms={bool(state.get('terms'))}, meds={bool(state.get('med_terms'))}, preds={bool(state.get('predicates'))})")

    # 返回全局聚合结果 + cluster-specific 缓存更新
    return {

        "cluster_cache_updates": {
            cluster_id: {
                "terms": terms,
                "med_terms": med_terms,
                "predicates": predicates,
                "rules": rules,
            }
        }
    }

def route_to_clusters(state: AgentState) -> List[Send]:
    sends = []
    for cluster in state.get("clusters", []):
        sends.append(
            Send(
                "process_cluster",
                {
                    "cluster_id": cluster.cluster_id,
                    "texts": cluster.provenances,
                    "texts_formatted": cluster.texts_formatted,
                    "terms": [],
                    "med_terms": [],
                    "predicates": [],
                    "rules": [],
                },
            )
        )
    return sends

def distribute_clusters_for_terms(state: dict) -> List[Send]:
    """分发 clusters 到术语抽取节点"""
    sends = []
    clusters = state.get("clusters", [])
    cluster_cache = state.get("cluster_cache", {})

    for cluster in clusters:
        # 从 cluster-specific 缓存准备状态
        cache_entry = cluster_cache.get(cluster.cluster_id, {})
        sends.append(
            Send(
                "process_cluster_terms",
                {
                    "cluster_id": cluster.cluster_id,
                    "provenances": cluster.provenances,
                    "texts_formatted": cluster.texts_formatted,

                },
            )
        )
    return sends

def distribute_clusters_for_predicates(state: dict) -> List[Send]:
    """分发 clusters 到谓词抽取节点"""
    sends = []
    clusters = state.get("clusters", [])
    cluster_cache = state.get("cluster_cache", {})

    for cluster in clusters:
        # 从 cluster-specific 缓存准备状态
        cache_entry = cluster_cache.get(cluster.cluster_id, {})
        sends.append(
            Send(
                "process_cluster_predicates",
                {
                    "cluster_id": cluster.cluster_id,
                    "provenances": cluster.provenances,
                    "texts_formatted": cluster.texts_formatted,
                    "terms": _to_models(cache_entry.get("terms", []), Term),
                    "med_terms": _to_models(cache_entry.get("med_terms", []), MedicationTerm),

                },
            )
        )
    return sends

def distribute_clusters_for_rules(state: dict) -> List[Send]:
    """分发 clusters 到规则抽取节点"""
    sends = []
    clusters = state.get("clusters", [])
    cluster_cache = state.get("cluster_cache", {})

    for cluster in clusters:
        # 从 cluster-specific 缓存准备状态
        cache_entry = cluster_cache.get(cluster.cluster_id, {})
        sends.append(
            Send(
                "process_cluster_rules",
                {
                    "cluster_id": cluster.cluster_id,
                    "provenances": cluster.provenances,
                    "texts_formatted": cluster.texts_formatted,
                    "terms": _to_models(cache_entry.get("terms", []), Term),
                    "med_terms": _to_models(cache_entry.get("med_terms", []), MedicationTerm),
                    "predicates": _to_models(cache_entry.get("predicates", []), Predicates)

                },
            )
        )
    return sends

# ============ 独立阶段图 ============

def build_terms_extraction_graph():
    """术语抽取专用图"""
    from .models import TermList, MedicationTermList

    class TermsState(TypedDict, total=False):
        clusters: List[ProvenanceCluster]
        cluster_cache: Dict[int, Dict[str, Any]]

        cluster_cache_updates: Annotated[Dict[int, Dict[str, Any]], merge_cluster_cache_updates]

    builder = StateGraph(TermsState)
    builder.add_node("process_cluster_terms", process_cluster_terms)

    builder.add_conditional_edges(START, distribute_clusters_for_terms, ["process_cluster_terms"])
    builder.add_edge("process_cluster_terms", END)

    return builder.compile()

def build_predicates_extraction_graph():
    """谓词抽取专用图"""
    from .models import TermList, MedicationTermList, PredicatesList

    class PredicatesState(TypedDict, total=False):
        clusters: List[ProvenanceCluster]
        cluster_cache: Dict[int, Dict[str, Any]]

        cluster_cache_updates: Annotated[Dict[int, Dict[str, Any]], merge_cluster_cache_updates]

    builder = StateGraph(PredicatesState)
    builder.add_node("process_cluster_predicates", process_cluster_predicates)

    builder.add_conditional_edges(START, distribute_clusters_for_predicates, ["process_cluster_predicates"])
    builder.add_edge("process_cluster_predicates", END)

    return builder.compile()

def build_rules_extraction_graph():
    """规则抽取专用图"""
    from .models import TermList, MedicationTermList, PredicatesList, ClinicalRuleList

    class RulesState(TypedDict, total=False):
        clusters: List[ProvenanceCluster]
        cluster_cache: Dict[int, Dict[str, Any]]

        cluster_cache_updates: Annotated[Dict[int, Dict[str, Any]], merge_cluster_cache_updates]

    builder = StateGraph(RulesState)
    builder.add_node("process_cluster_rules", process_cluster_rules)

    builder.add_conditional_edges(START, distribute_clusters_for_rules, ["process_cluster_rules"])
    builder.add_edge("process_cluster_rules", END)

    return builder.compile()

def build_pipeline_graph():
    """
    完整流水线图：两阶段 Map-Reduce 架构
    
    阶段1 (Map-Reduce): 并行抽取推荐意见
    START -> distribute_texts (Send) -> extract_recommendation (并行 N 个)
                                              ↓
                                     [reducer: operator.add 自动聚合到 provenance_buffer]
    
    阶段2 (Map-Reduce): 聚类后并行抽取知识
                                              ↓
                                       lsh_clustering
                                              ↓
                               route_to_clusters (Send) -> process_cluster (并行 M 个)
                                                                  ↓
                                                    [reducer: merge_by_id 自动聚合去重]
                                                                  ↓
                                                                 END
    """
    builder = StateGraph(AgentState)
    builder.add_node("extract_recommendation", extract_recommendation)
    builder.add_node("lsh_clustering", do_lsh_clustering)
    builder.add_node("process_cluster", process_cluster)

    builder.add_conditional_edges(START, distribute_texts, ["extract_recommendation"])
    builder.add_edge("extract_recommendation", "lsh_clustering")
    builder.add_conditional_edges("lsh_clustering", route_to_clusters, ["process_cluster"])
    builder.add_edge("process_cluster", END)
    return builder.compile()


