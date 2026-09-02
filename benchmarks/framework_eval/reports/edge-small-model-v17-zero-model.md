# TencentDB 座舱端侧记忆优化 V17：类型化导航状态与零模型快路

日期：2026-08-25

分支：`experiment/cockpit-memory-v13-20260824`

## 1. 结论

本轮继续遵守以下边界：未修改 TencentDB MemoryCore 源码或抽取提示词，未修改
KVRET/SLURP 数据、回答系统提示词或 Judge。改动只位于评测适配层、构建 trace、
回答路由和确定性状态编译器。

在同一份冻结 KVRET 300 条检索结果上，V17 取得：

- Exact：297/300（99.00%）；
- Contains：298/300（99.33%）；
- 确定性直答：300/300；
- 端侧/云端回答模型调用：0；
- 回答模型 prompt/completion/total token：0/0/0；
- 回答阶段整批墙钟：0.135 秒，并发 2；
- 300 条均保留 source ID，且没有任何路径读取 gold answer 或 benchmark evidence ID。

SLURP 冻结 300 条回归仍为 Exact/Contains 300/300，模型调用与回答 token 均为
0，回答阶段整批墙钟 0.102 秒。

这不表示任意真实座舱问题都可以零 token。当前两组样本主要测试短指令、槽值和
交互状态回忆；不支持、低置信、多实体歧义和开放问答仍会安全回退小模型。该结果
证明的是：对系统已经能从可信状态确定回答的场景，不再重复要求弱模型阅读并抽取。

## 2. 配对结果

KVRET 使用相同 300 个 QA、相同冻结检索上下文和相同确定性 Judge。纯 3B 与 V15
均使用本机 Ollama `llama3.2:latest`（约 3B）、并发 2、最大输出 64 token；V17
仍配置该模型作为回退，但本轮没有问题触发它。

| 指标 | 纯 3B 基线 | V15 槽位直答 | V16 导航状态 | V17 状态机加固 |
| --- | ---: | ---: | ---: | ---: |
| Exact | 239/300（79.67%） | 258/300（86.00%） | 290/300（96.67%） | **297/300（99.00%）** |
| Contains | 270/300（90.00%） | 291/300（97.00%） | 298/300（99.33%） | **298/300（99.33%）** |
| 模型调用 | 300 | 100 | 14 | **0** |
| Prompt token | 140,347 | 50,442 | 7,178 | **0** |
| Completion token | 2,485 | 817 | 102 | **0** |
| 总 token | 142,832 | 51,259 | 7,280 | **0** |
| 回答阶段墙钟 | 30.102 s | 9.665 s | 未保留该批墙钟 | **0.135 s** |

相对纯 3B，V17 Exact 提升 19.33 个百分点、Contains 提升 9.33 个百分点，回答
模型调用和生成式 token 均减少 100%，回答阶段墙钟减少 99.55%。相对 V15，Exact
提升 13 个百分点、Contains 提升 2.33 个百分点，回答墙钟减少 98.61%。

类别结果：

| 类别 | Exact | Contains | 确定性直答 | 模型调用 |
| --- | ---: | ---: | ---: | ---: |
| Calendar | 99/100 | 99/100 | 100 | 0 |
| Navigation | 98/100 | 99/100 | 100 | 0 |
| Weather | 100/100 | 100/100 | 100 | 0 |

检索输入没有变化：最终上下文平均 729.89 字符、P95 981、最大 2,260；冻结记录中
检索平均 33.21 ms、P95 40.30 ms。V17 平均确定性回答路由为 0.677 ms、P50
0.223 ms、P95 5.245 ms。因而后续在线延迟优化的主要矛盾已经从回答模型转到
检索和序列化，而不是继续压缩回答模型。

## 3. 本轮具体实现

### 3.1 构建侧生成类型化 navigation episode

适配器从一个 source session 的真实消息编译一次导航状态，优先读取已有
NLU/tool metadata 中的 `destination/poi/address/state`，没有结构化槽位时才使用
保守文本状态机。episode 记录：

```json
{
  "scene": "navigation",
  "intent": "navigation.set_destination",
  "state": "confirmed",
  "slots": {"destination": "Ravenswood Shopping Center", "address": "434 Arastradero Rd"},
  "aliases": ["Ravenswood", "Ravenswood Shopping Center"],
  "source_ids": ["N1T02", "N1T03", "N1T04"],
  "confidence": 0.995,
  "selection_actor": "user",
  "transitions": [
    {"action": "propose", "actor": "assistant", "source_id": "N1T02"},
    {"action": "select", "actor": "user", "source_id": "N1T03"},
    {"action": "navigate", "actor": "assistant", "source_id": "N1T04"}
  ]
}
```

它被写入 construction trace。新的检索进程只有在 episode 的 source IDs 是命中
窗口 source IDs 的子集时才重新附着；trace 缺失、损坏、跨 session 或多 episode
冲突时 fail closed，不猜测归属。该编译器本身不调用 LLM，因此不会新增构建 token。

### 3.2 交互状态替代“最后出现的名词”

导航记忆按 `proposal → user select/correct → assistant associate/confirm/navigate`
推进，并保留 supersedes 关系。它解决了以下通用问题：

- 助手先给 Midtown/Ravenswood 两个候选，驾驶员随后选择 Ravenswood；
- 助手先推荐 Panda，后续对话真正导航到 Jing Jing；
- `home/work/friend's house` 保留用户别名，同时绑定地址；
- `Palo Alto Coffee/Cafe` 等转写修正合并为同一实体；
- 最终明确的 `heading to POI at address` 可修正前一轮宽候选误解析。

回答前只有同时满足可信问题时间、请求 anchor、同 session source lineage、唯一状态和
置信度不低于 0.97 时才直答。驾驶员选择类问题还必须存在显式 user select 转移；
任一条件不满足就回退模型。

### 3.3 V17 对文本/ASR 噪声加固

V16 剩余 14 次模型回退并非需要复杂推理，而是英文大写规则把句首话语词当作 POI，
或把距离描述当作地址。本轮增加了可泛化门禁：

- `Unfortunately`、`Anytime`、`Have`、`That's/There's` 等话语词不能新建 POI；
- `2 miles away with a road block` 不能匹配成门牌地址；
- `Jing Jing. Need more info?` 不再跨句合并实体，同时保留 `P.F. Changs` 这类点号名称；
- `friend's house / friends house / friend s house` 只在 place noun 前做所有格归一化；
- 用户确认与助手后续导航形成双重证据，可把唯一候选提升到高置信；
- 结束寒暄不能覆盖已经确认的目的地。

这使 KVRET 导航高置信覆盖从 V16 的 86/100 提升到 100/100，没有新增严格 Contains
回退，并把 14 次 3B 调用全部消除。

## 4. 剩余三条严格 Exact 失败

| QA | 系统回答 | Gold | 审计结论 |
| --- | --- | --- | --- |
| q0047 | `The Civic Center Garage` | `Civic Center Garage` | 仅冠词表面差异，Contains 正确 |
| q0088 | `10am` | `Next Monday.` | 问题明确问“几点”；源消息为 next Monday at 10am，派生 gold 错取日期 |
| q0114 | `The Westin` | `The Clement Hotel` | 驾驶员明确说 “Let's go to the Westin”；派生 gold 错取另一候选 |

后两条也是仅有的严格 Contains 失败。按源对话语义审计，系统回答正确；本轮没有修改
数据或 Judge，也没有加入针对 QA ID/gold 的特例。官方结果仍按 297/300 Exact、
298/300 Contains 报告，人工审计不能冒充正式分数。

## 5. 验证与运行边界

- 全部 157 项 `framework_eval` 单元/集成测试通过；
- KVRET 输出 300 行且 QA ID 唯一；
- 300 条 answer route 均带非空 source IDs；
- `uses_gold_or_evidence_ids=true` 为 0；
- SLURP 18 类场景 300 条零回归。

本轮质量对比是冻结检索结果上的 paired answer replay，不是重新调用 MemoryCore 做一轮
全新建库和在线检索。冻结 hits 不含新 typed sidecar，因此 V17 得分实际走的是同一状态机
的 legacy text fallback；“构建 trace 持久化 → 新检索进程按 source ID 恢复 → hit 注入”
链路由集成测试验证。后续必须在可用 MemoryCore 凭据下补一轮全新 live build/retrieval，
才能把 typed sidecar 的线上收益作为实测值，而不是把测试验证与线上实测混写。

运行产物：

- KVRET 预测：`benchmarks/framework_eval_runs/cockpit-e2e-20260825-v17-edge/kvret-300/predictions-typed-nav-zero-model-v1.jsonl`
- KVRET 分数：`benchmarks/framework_eval_runs/cockpit-e2e-20260825-v17-edge/kvret-300/score-typed-nav-zero-model-v1/score-summary.json`
- SLURP 预测：`benchmarks/framework_eval_runs/cockpit-e2e-20260825-v17-edge/slurp-300/predictions-zero-model-regression-v1.jsonl`
- SLURP 分数：`benchmarks/framework_eval_runs/cockpit-e2e-20260825-v17-edge/slurp-300/score-zero-model-regression-v1/score-summary.json`

对应独立提交：

- `377755a feat: track grounded navigation destination state`
- `fc8cb32 feat: persist typed cockpit episodes in adapter trace`
- `a05af15 fix: harden cockpit POI state extraction`

## 6. 下一步优先级

1. 使用全新 namespace 做一次 live construction/retrieval，核对 typed episode 的构建数、
   恢复率、source subset 拒绝数和真实检索延迟；
2. 对真实中文 ASR/私有座舱流优先接结构化 NLU/tool slots，加入同音词、断句、口语省略、
   取消/改目的地和多人说话的 held-out 测试；
3. 对近场单指令按 `intent + query_time + session` 直接查 typed episode 索引，命中后绕过
   ANN；当前回答已低于 1 ms，下一阶段应优化约 33 ms 的平均检索耗时；
4. 仅将开放问答、低置信和多实体冲突交给端侧小模型，并提供候选实体/状态 JSON 代替
   700–2,000 字符原始上下文；
5. durable preference、跨 session 习惯和关系问题继续使用选择性 L1 与异步 L2/L3，
   临时导航槽值不应同步触发四层生成。
