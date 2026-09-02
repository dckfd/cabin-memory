# TencentDB Temporal Cockpit V14：相对时间端到端优化

日期：2026-08-25  
分支：`experiment/cockpit-memory-v13-20260824`  
数据：KVRET、SLURP 各固定 300 条；每套包含 yesterday、day-before-yesterday、last-weekday 各 100 条。答案、evidence 与源显式日期样本保持不变。

## 结论

- 两套数据的 evidence 均为 300/300 完整召回。
- KVRET：Exact 253/300（84.33%），Contains 293/300（97.67%）。
- SLURP：Exact 288/300（96.00%），Contains 299/300（99.67%）。
- 相对同 ID 的显式日期基线，平均检索延迟下降 82.76% / 81.56%，动态搜索调用下降 96.08% / 89.87%。
- 与修复前相对时间版本相比，最终 600 条回答 token 从 260,062 降至 239,692（-7.83%），质量无回退。
- 没有修改 MemoryCore 源码或提示词，没有修改数据集、回答 prompt 或 Judge。实现只位于 framework-eval 适配层、运行配置和测试。

## 实现

1. 从可信的 `query_time`、`question_time`、`request_time`、`asked_at`、`captured_at` 或 `question_date` 读取提问时刻和时区；没有锚点时不猜测相对日期。
2. 将中英文“昨天/前天/昨晚/last Tuesday”等解析为半开时间区间，同时保留 latest/earliest 排序意图。
3. 在 ANN Top-K 之前扫描已缓存的规范 L0 历史，按 source time 与语义 event time 融合时间候选。
4. 高置信时间候选直接走 L0 short circuit；低置信、更新、多事实问题仍自适应回退到 Top-3/L1/L2/L3。
5. 短会话保留发起指令、助手澄清和用户最终槽值；长会话维持 8 条消息硬上限。
6. 只为已解析的相对时间问题向 user message 注入请求时间和绝对区间。普通问题与显式日期问题保持字节级不变。
7. 引号内的 `"today"`、`"tomorrow"` 等视为历史话语/槽位内容，不再错误地按当前提问时间解析。
8. conversation-batched L1 在独立检索进程中必须携带 construction trace 才扩展相邻窗口；缺少 source-session provenance 时安全地跳过扩窗，避免把不同座舱指令静默拼接。

## 检索结果

显式日期基线与相对时间版本使用相同 question ID、store、Top-K 和最终上下文预算。

| 数据集 | 配置 | Evidence 100% | L0 short circuit | 动态搜索调用 | 平均 / P50 / P95 延迟 | 平均 / P95 上下文字符 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| KVRET | 显式日期基线 | 300/300 | 0 | 306 | 192.59 / 178.72 / 291.80 ms | 716.41 / 900 |
| KVRET | Temporal V14 | 300/300 | 294 | 12 | 33.21 / 22.42 / 40.30 ms | 718.31 / 900 |
| SLURP | 显式日期基线 | 300/300 | 0 | 316 | 195.18 / 179.07 / 338.00 ms | 393.61 / 402 |
| SLURP | Temporal V14 | 300/300 | 284 | 32 | 35.99 / 15.99 / 290.58 ms | 419.47 / 673 |

相对显式日期基线，KVRET 平均上下文仅增加 0.27%，SLURP 增加 6.57%。时间规范化不调用模型；时间候选扫描平均耗时分别为 17.66 ms 和 11.25 ms。两个 300 条只读检索进程分别用时 9.99 秒和 10.83 秒。

引号去歧义修复本身将 KVRET 快路由从 268 提升到 294，动态搜索从 64 降至 12，平均检索从 60.95 ms 降至 33.21 ms；SLURP 快路由从 278 提升到 284，动态搜索从 44 降至 32，平均检索从 45.41 ms 降至 35.99 ms。典型天气 bad case 从 2 次搜索、2,200 字符干扰上下文变为 0 次搜索、677 字符单一正确会话。

## 回答、Token 与评分

正式结果固定 DeepSeek V4 Flash 非思考模式、8 并发、64 输出 token 上限，并使用既有默认回答 prompt。较长 prompt 的审计跑分不纳入主结果。

| 数据集 | 版本 | Prompt / Completion / 总 token | Exact | Contains |
| --- | --- | ---: | ---: | ---: |
| KVRET | 修复前相对时间 | 153,506 / 1,432 / 154,938 | 251/300（83.67%） | 292/300（97.33%） |
| KVRET | Temporal V14 | 135,334 / 1,439 / 136,773 | 253/300（84.33%） | 293/300（97.67%） |
| SLURP | 修复前相对时间 | 104,383 / 741 / 105,124 | 288/300（96.00%） | 299/300（99.67%） |
| SLURP | Temporal V14 | 102,178 / 741 / 102,919 | 288/300（96.00%） | 299/300（99.67%） |
| 合计 | 修复前 → V14 | 257,889 → 237,512 / 2,173 → 2,180 / 260,062 → 239,692 |  |  |

总 token 减少 20,370（7.83%）。按此前实验固定的保守估算口径——全部 prompt 均按 ¥1/百万、completion 按 ¥2/百万——本轮回答约 ¥0.242；实际 usage 含缓存命中，不能把该上界当作账单。

相对时间 envelope 平均增加 146.71（KVRET）和 146.75（SLURP）个字符；它只作用于相对时间问题。忽略引号内历史槽位后，KVRET 的 envelope 平均长度比修复前减少 7.47 字符。

## Bad case 边界

最终 Contains 未通过 8 条，但全部检索 evidence 完整：

- KVRET 3 条将地点别名 `home` 回答为已存家庭地址，语义正确但字符串 Judge 不认别名。
- KVRET 2 条源对话与 gold 自相矛盾或名称拼写不一致：实际选择 Westin 但 gold 为 Clement Hotel；Palo Alto Coffee/Cafe 混用。
- KVRET 1 条问题问 time，但 gold 为 `Next Monday`；模型回答证据中的 `10am`。
- KVRET 1 条在 900 字符预算边界把 `Webster Garage` 截为 `Webster G`。后续适合做“完整消息优先”的自适应字符预算，不应全局放大上下文。
- SLURP 1 条把口语化 `macs@gmail dot com` 正规化为 `macs@gmail.com`。

这些结果不通过修改 Judge、gold 或回答 prompt 修饰。严格分数与语义分析分开报告。

## 构建与可复现边界

本轮复用已完成的 TencentDB L0/L1/L2/L3 store，只读检索没有新增构建调用或构建 token。因此 239,692 仅是本轮 600 条回答 token，不能代表从零构建总成本。原 KVRET 构建数据见 `kvret-v13-deepseek-flash-end-to-end.md`，原 SLURP 构建数据见 `slurp-r12-end-to-end.md`。

SLURP 旧 store 采用 conversation-batched transport；最终只读运行显式传入原 `traces/` 目录。一次漏传 trace 的审计运行暴露了跨 source-session 扩窗风险，现已增加 fail-closed 保护。两份最终 retrieval 产物均经 `validate --expected-count 300` 校验，无缺行和告警。

## 相关提交与验证

- `15a1b8b`：相对时间规范化。
- `8d512b9`：Top-K 前时间候选融合。
- `41c01bf`：回答时注入可信时间锚点。
- `d04a45b`：有界时间问题快路由。
- `9a9a154`：高置信时间候选 short circuit。
- `2beb3ff`：忽略引号内历史相对时间槽位。
- `b7329f5`：缺少 source-session provenance 时禁止不安全扩窗。

最终 framework-eval 测试：137 项全部通过。

机器可读产物位于：

```text
benchmarks/framework_eval_runs/cockpit-e2e-20260825-v14/temporal-cockpit/
  kvret-300/retrieval-v5.jsonl
  kvret-300/predictions-v3-default.jsonl
  kvret-300/score-v3-default/
  slurp-300/retrieval-v3.jsonl
  slurp-300/predictions-v3-default.jsonl
  slurp-300/score-v3-default/
```
