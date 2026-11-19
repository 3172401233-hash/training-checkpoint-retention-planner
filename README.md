# training-checkpoint-retention-planner
规划训练检查点保留、继承关系和存储清理窗口

## 概述

`checkpoint_planner.py` 根据检查点清单（父检查点、优化器状态、文件大小、
上传状态、校验摘要、验证分数、存储层）和保留预算，生成保留计划：

- **必须保留**：恢复链上每个任务的最近有效检查点、回滚窗口内的祖先、
  最佳验证指标、未过期的固定归档、周期性归档、正在上传的检查点，
  以及仍被多个任务引用的记录。
- **可删除**：不满足任何保留规则、且删除后能逼近预算的记录（最旧优先）。
- **不可用版本**：上传中或校验摘要不完整的检查点不计为可用版本，
  不会成为恢复链的锚点。
- **重复内容**：相同摘要的记录共享存储成本；删除其中一条不一定释放空间，
  但多任务引用的记录不会因为内容相同而被删除。
- **孤立检查点**：父记录缺失的检查点会在 `plan.orphans` 中报告。
- **预算不足**：删除全部可删记录后仍超预算时，差额体现在
  `plan.unmet_bytes` 并附说明。

## 重要说明

清理计划**不会直接删除任何存储对象**。`plan_retention()` 只返回纯数据的
`RetentionPlan`（保留集合、删除集合、释放空间、保留原因、孤立列表、
未满足预算部分），实际删除由调用方审计后自行执行。

## 用法

```python
from checkpoint_planner import CheckpointRecord, RetentionPolicy, plan_retention

records = [
    CheckpointRecord(checkpoint_id="c1", task_id="train-1", created_at=1,
                     size_bytes=100, digest="sha256:..."),
    CheckpointRecord(checkpoint_id="c2", task_id="train-1", parent_id="c1",
                     created_at=2, size_bytes=100, digest="sha256:..."),
]
plan = plan_retention(records, RetentionPolicy(budget_bytes=150, rollback_window=1))
print(plan.keep, plan.delete, plan.freed_bytes, plan.unmet_bytes)
```

## 测试

兼容 Python 3.8.2+：

```
python -m unittest
```

覆盖线性恢复链、分叉链、重复摘要、半上传、最佳指标保护、到期归档与
预算不足等场景。
