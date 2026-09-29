# RAG 检索优化指南

## 概述

本文档记录了 SageMatch 项目中 RAG 检索系统的优化方案和使用指南。

## 已实现的优化

### 1. 查询理解模块 (Query Understanding)

**文件位置**: `app/services/materials/rag/query_understanding.py`

**功能**:
- **意图识别**: 识别查询类型（factual, procedural, comparative, conceptual）
- **实体抽取**: 提取关键技术术语和概念
- **复杂度评估**: 判断查询是简单、中等还是复杂
- **综合需求判断**: 检测是否需要多来源信息综合

**使用示例**:
```python
from app.services.materials.rag.query_understanding import analyze_query

# 分析用户查询
analysis = analyze_query("FastAPI 和 Django 在性能上有什么区别？")

print(analysis.intent)              # "comparative"
print(analysis.complexity)          # "complex"
print(analysis.entities)            # ["FastAPI", "Django"]
print(analysis.requires_synthesis)  # True
```

**关键参数**:
- `intent`: 查询意图（用于选择不同的检索策略）
- `complexity`: 复杂度（simple/moderate/complex，影响 top_k 大小）
- `requires_synthesis`: 是否需要综合多个来源（触发补充召回）

---

### 2. 自适应多阶段检索流水线 (Adaptive Retrieval Pipeline)

**文件位置**: `app/services/materials/rag/adaptive_retrieval.py`

**核心流程**:

```
查询理解
    ↓
策略选择（根据复杂度）
    ↓
Stage 1: 宽召回（broad_recall）
    - 词法检索 + 向量检索
    - 动态 top_k（30-80）
    ↓
Stage 2: 精排 + 过滤（rerank_and_filter）
    - 使用 reranker 重排序
    - 应用相关性阈值过滤
    ↓
Stage 3: 补充召回（supplement_recall，条件触发）
    - 查询扩展（实体增强）
    - 去重后补充新结果
    ↓
Stage 4: 多样性治理（govern_candidates）
    - 限制单材料的块数量
    - 合并相邻块
```

**策略类型**:

| 策略名称 | 触发条件 | broad_top_k | relevance_threshold | final_top_k |
|---------|---------|-------------|---------------------|-------------|
| `efficient` | 简单查询 | 30 | 0.0 | 8 |
| `standard` | 默认 | 40 | 0.0 | 10 |
| `wide_recall` | 复杂查询 | 60 | 0.5 | 15 |
| `synthesis` | 需要综合 | 80 | 0.4 | 20 |

**使用示例**:
```python
from app.services.materials.rag.adaptive_retrieval import AdaptiveRetrievalPipeline

# 初始化流水线
pipeline = AdaptiveRetrievalPipeline(db_session)

# 执行检索
result = await pipeline.retrieve(
    query="如何使用 FastAPI 实现异步接口？",
    packed_chunks=all_chunks,
    query_vector=query_embedding,
    reranker=reranker_model,
)

# 获取结果
candidates = result["candidates"]
diagnostics = result["diagnostics"]

# 诊断信息
print(f"检索策略: {diagnostics['retrieval_strategy']}")
print(f"执行阶段: {diagnostics['stages_executed']}")
print(f"总延迟: {diagnostics['total_latency_ms']}ms")
```

**诊断信息说明**:
- `query_intent`: 查询意图
- `query_complexity`: 复杂度评估
- `retrieval_strategy`: 使用的检索策略
- `stages_executed`: 执行的阶段列表
- `stage1_candidates`: 宽召回的候选数量
- `stage2_candidates`: 精排后的候选数量
- `stage3_supplemented`: 补充召回的数量（如果触发）
- `final_candidates`: 最终返回的候选数量
- `total_latency_ms`: 总检索延迟（毫秒）

---

### 3. Agent 自愈能力 (Resilience)

**文件位置**: `app/agents/orchestration/resilience.py`

**核心组件**:

#### a) 重试策略 (RetryPolicy)
```python
from app.agents.orchestration.resilience import RetryPolicy, RetryableError

policy = RetryPolicy(
    max_retries=3,
    initial_backoff=1.0,  # 1秒
    max_backoff=10.0,     # 最大10秒
    backoff_multiplier=2.0  # 指数退避
)

# 判断是否应该重试
should_retry = policy.should_retry(TimeoutError())  # True
should_retry = policy.should_retry(ValueError())     # False（不可重试的错误）

# 计算退避时间
wait_time = policy.calculate_backoff(attempt=1)  # 2.0秒
```

**可重试的错误类型**:
- `TimeoutError`: 超时
- `RateLimitError`: 速率限制
- `TemporaryError`: 临时错误
- `ConnectionError`: 连接错误

#### b) 断路器 (CircuitBreaker)
```python
from app.agents.orchestration.resilience import CircuitBreaker

breaker = CircuitBreaker(
    failure_threshold=5,  # 5次失败后打开
    timeout_seconds=60,   # 60秒后进入半开状态
)

# 记录调用结果
await breaker.record_success()
await breaker.record_failure()

# 检查是否允许调用
if await breaker.allow_request():
    # 执行操作
    pass
else:
    # 断路器已打开，拒绝请求
    raise CircuitBreakerOpenError()
```

**断路器状态**:
- `CLOSED`: 正常状态，允许所有请求
- `OPEN`: 断开状态，拒绝所有请求（失败次数超过阈值）
- `HALF_OPEN`: 半开状态，允许少量请求测试（超时后自动进入）

#### c) 弹性 Agent (ResilientAgent)
```python
from app.agents.orchestration.resilience import ResilientAgent

agent = ResilientAgent(
    base_agent=my_agent,
    retry_policy=retry_policy,
    circuit_breaker=breaker,
    context_degradation_enabled=True,
)

# 自动重试 + 降级调用
result = await agent.invoke_with_resilience(
    input_data={"query": "用户查询", "context": long_context},
    max_retries=3,
)
```

**自动降级策略**:
1. 第一次失败：重试（完整上下文）
2. 第二次失败：截断上下文到 70%，重试
3. 第三次失败：截断上下文到 49%，重试
4. 超过重试次数：抛出异常

**上下文截断逻辑**:
```python
def truncate_context(context: str, ratio: float) -> str:
    """保留前 ratio 比例的上下文"""
    target_length = int(len(context) * ratio)
    return context[:target_length]
```

---

## 集成示例

### 完整的 RAG + 弹性调用流程

```python
from app.services.materials.rag.adaptive_retrieval import AdaptiveRetrievalPipeline
from app.agents.orchestration.resilience import ResilientAgent, RetryPolicy, CircuitBreaker

# 1. 初始化组件
pipeline = AdaptiveRetrievalPipeline(db_session)
retry_policy = RetryPolicy(max_retries=3)
breaker = CircuitBreaker(failure_threshold=5)

# 2. 执行 RAG 检索
rag_result = await pipeline.retrieve(
    query=user_query,
    packed_chunks=chunks,
    query_vector=embedding,
    reranker=reranker,
)

# 3. 构建 Agent 上下文
context = "\n\n".join([c["content"] for c in rag_result["candidates"]])

# 4. 使用弹性 Agent 调用
resilient_agent = ResilientAgent(
    base_agent=interviewer_agent,
    retry_policy=retry_policy,
    circuit_breaker=breaker,
)

response = await resilient_agent.invoke_with_resilience(
    input_data={
        "query": user_query,
        "context": context,
        "diagnostics": rag_result["diagnostics"],
    },
    max_retries=3,
)

# 5. 记录诊断信息
logger.info(f"RAG 延迟: {rag_result['diagnostics']['total_latency_ms']}ms")
logger.info(f"检索策略: {rag_result['diagnostics']['retrieval_strategy']}")
logger.info(f"最终候选数: {rag_result['diagnostics']['final_candidates']}")
```

---

## 性能调优建议

### 1. 根据材料规模调整参数

**小规模材料库（< 10 个材料）**:
```python
# 使用 efficient 策略，减少不必要的召回
# 在 rag_config.yaml 中设置:
recall:
  score_floor: 0.1  # 提高分数下限
  max_materials: 3  # 限制材料数量
```

**大规模材料库（> 50 个材料）**:
```python
# 增加召回范围，启用更严格的过滤
recall:
  score_floor: 0.05  # 降低分数下限
  max_materials: 8   # 允许更多材料
rerank:
  enabled: true      # 必须启用 reranker
```

### 2. 根据查询类型优化

**事实性查询（What is X?）**:
- 倾向于 `efficient` 策略
- 不需要补充召回
- rerank threshold 可以设高一些（0.6+）

**对比性查询（X vs Y）**:
- 使用 `synthesis` 策略
- 确保至少覆盖两个实体的材料
- max_per_material 设为 4-6

**过程性查询（How to do X?）**:
- 使用 `wide_recall` 策略
- 可能需要多轮补充召回
- 合并相邻块以保持连贯性

### 3. 降低延迟

**启用缓存**:
```python
# 缓存向量检索结果
from functools import lru_cache

@lru_cache(maxsize=128)
def get_query_embedding(query: str) -> list[float]:
    return embedding_model.embed(query)
```

**并行化独立操作**:
```python
import asyncio

# 并行执行词法和向量检索
lexical_task = asyncio.create_task(lexical_retrieve(query))
vector_task = asyncio.create_task(vector_retrieve(query))

lexical_results = await lexical_task
vector_results = await vector_task
```

**批量 rerank**:
```python
# 使用批量处理减少网络往返
rerank_candidates(
    query,
    candidates,
    batch_size=32,  # 增加批次大小
)
```

---

## 监控指标

### 关键指标

1. **召回率（Recall）**: 相关文档被检索出的比例
2. **精确率（Precision）**: 检索结果中相关文档的比例
3. **MRR@k**: 第一个相关文档的平均倒数排名
4. **延迟（Latency）**: 端到端检索时间

### 如何评估

```python
# 使用诊断信息评估
diagnostics = result["diagnostics"]

# 延迟监控
if diagnostics["total_latency_ms"] > 1000:
    logger.warning("检索延迟过高")

# 召回量监控
if diagnostics["final_candidates"] < 3:
    logger.warning("召回结果不足")

# 策略使用统计
strategy_counter[diagnostics["retrieval_strategy"]] += 1
```

---

## 故障排查

### 问题 1: 检索结果为空

**可能原因**:
- `score_floor` 设置过高
- rerank threshold 过严格
- 查询与材料内容差异过大

**解决方案**:
```python
# 检查诊断信息
if result["retrieval_status"] == "empty":
    print(f"Stage 1 候选数: {diagnostics['stage1_candidates']}")
    print(f"Stage 2 候选数: {diagnostics['stage2_candidates']}")
    
    # 如果 stage1 就为空，是基础召回问题
    # 如果 stage2 过滤太多，调整 threshold
```

### 问题 2: Agent 调用频繁失败

**可能原因**:
- 上下文过长导致超时
- 速率限制
- 模型服务不稳定

**解决方案**:
```python
# 启用上下文降级
agent = ResilientAgent(
    base_agent=my_agent,
    context_degradation_enabled=True,  # 自动截断上下文
)

# 调整断路器阈值
breaker = CircuitBreaker(
    failure_threshold=3,  # 降低阈值，更早断开
    timeout_seconds=30,   # 减少恢复时间
)
```

### 问题 3: 延迟过高

**排查步骤**:
1. 检查各阶段延迟：`diagnostics['stages_executed']`
2. 识别瓶颈：
   - Stage 1 慢 → 词法/向量检索优化
   - Stage 2 慢 → reranker 批次大小优化
   - Stage 3 触发频繁 → 调整 `min_required` 参数

---

## 未来优化方向

1. **混合搜索权重自适应**: 根据查询类型动态调整词法/向量权重
2. **查询扩展 LLM 集成**: 使用 LLM 进行智能查询改写
3. **负反馈学习**: 记录用户反馈，调整检索策略
4. **增量索引**: 支持材料增量更新而不重建全量索引

---

## 测试覆盖

- ✅ 查询理解测试：`tests/unit/services/test_query_understanding.py`
- ✅ 自适应检索测试：`tests/unit/services/test_adaptive_retrieval.py`
- ✅ 弹性 Agent 测试：`tests/unit/agents/test_resilience.py`

运行全部测试：
```bash
cd backend
python -m pytest tests/unit/services/test_query_understanding.py \
                 tests/unit/services/test_adaptive_retrieval.py \
                 tests/unit/agents/test_resilience.py -v
```

---

**文档维护**: 本文档随代码演进持续更新
**最后更新**: 2026-09-30
