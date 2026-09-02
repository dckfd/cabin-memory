# 通用记忆框架评测系统进展

更新日期：2026-08-11

## 1. 目标与边界

本评测系统用同一数据输入、同一回答模型、同一 Judge 和同一统计口径比较不同记忆框架。评测器不修改第三方框架算法，也不把各框架的依赖装进 TencentDB Agent Memory 主环境。

统一链路：

```text
LoCoMo-Refined / LongMemEval
  -> DatasetAdapter（保留 Session、时间、角色、证据 ID）
  -> MemoryAdapter（每个 conversation 独立 store/namespace）
  -> canonical retrieval.jsonl
  -> 统一回答模型和 Prompt
  -> canonical predictions.jsonl
  -> 数据集官方 Judge
  -> 检索、答案、Token、延迟、错误分层报告
```

必须分开报告的指标：

1. Evidence Recall / R@K：记忆检索是否覆盖标注证据。
2. QA F1 / BLEU：生成答案和参考答案的词面重合度。
3. LLM-as-Judge：语义上是否正确，不等同于检索召回率。
4. 输入、输出、建库 Token 和 Token 节省率。
5. 建库、检索、回答延迟及失败率。

MemPalace 公布的 LoCoMo R@10、LongMemEval R@5 属于检索指标，不能与 LoCoMo-Refined 的端到端 LLM Judge 百分比直接放在同一排名列。

## 2. LoCoMo-Refined 已知公开结果

以下数字来自 LoCoMo-Refined 官方仓库的公开表格，先作为外部参考值，后续只把本机、相同模型和相同配置复现的结果放进正式横向表。

| 系统 | 原始 LoCoMo Judge | Refined Judge | 本机 OSS 复现状态 |
|---|---:|---:|---|
| MemoraX AI | 82.65% | N/A | 无可下载的核心实现，不能复现 |
| EverMemOS | 58.25% | 22.07% | EverOS 源码已下载，待隔离运行时 |
| MemOS | 63.60% | 17.30% | 源码已下载，待 Neo4j/Qdrant 运行时 |
| MemPalace | 58.68% | 15.78% | 源码、隔离运行时和 4 题接口 Gate 已就绪；未跑正式 Judge |
| Mem0 | 48.91% | 15.56% | OSS 和托管版严格分轨，待 OSS 运行时 |

这组结果本身说明 Judge 修订会大幅改变排名和绝对分数。正式实验必须固定 `LoCoMo_refined` 数据版本、Qwen3-14B refined Judge、回答模型、Prompt、Top-K 和上下文预算。

## 3. 已下载框架

所有第三方源码均为浅克隆，保存在 `third_party/memory_frameworks/`，未修改其源码。

| ID | 框架 | 固定 commit | 许可证 | 状态/运行方式 |
|---|---|---|---|---|
| mem0 | Mem0 OSS | `4debc58a8337` | Apache-2.0 | 已下载；独立 Server/SDK |
| everos | EverOS / EverMemOS | `48fc9084888b` | Apache-2.0 | 已下载；独立 Server |
| memos | MemOS | `8d310a7a4be6` | Apache-2.0 | 已下载；Neo4j + Qdrant |
| mempalace | MemPalace | `8516db7fbc7f` | MIT | 隔离 venv + 默认 MiniLM/Chroma；首个 Gate 已通过 |
| memora | Microsoft Memora | `dec3f8f2444e` | MIT | 2026；原生 LoCoMo/LongMemEval runner |
| lightmem | LightMem | `8fc9a9179f91` | MIT | ICLR 2026；自带多框架工具 |
| hindsight | Hindsight | `899c07e6c447` | MIT | 2026；REST + PostgreSQL |
| memmachine | MemMachine | `d7856f23f3e0` | Apache-2.0 | 2026；独立 API Server |
| hypermem | HyperMem | `15c700908f3e` | Apache-2.0 | ACL 2026；原生 LoCoMo pipeline |
| memoryos | MemoryOS | `587ed7755c7a` | Apache-2.0 | EMNLP 2025 补充基线 |
| memory-benchmarks | Mem0 官方 benchmark | `4b61c5d` | Apache-2.0 | LoCoMo/LongMemEval 参考实现 |

未下载项：

- MemoraX AI：公开的是概念规范，未核验到与 82.65% 对应的开源核心实现。
- LazyMem：已发现 2026 论文，但尚未核验到官方开源仓库，因此只登记为 paper-only，不能伪装成可复现框架。

完整机器可读清单和检查结果：`benchmarks/framework_eval/frameworks.json` 与 `benchmarks/framework_eval_runs/doctor.json`。
固定下载版本另见可提交的 `benchmarks/framework_eval/sources.lock.json`；大体积源码、隔离 venv、模型缓存、store 和运行结果均已加入 `.gitignore`，本地文件不会被删除。

## 4. 已实现模块

代码位于 `benchmarks/framework_eval/`：

- `schema.py`：Conversation、Session、Message、Question、MemoryHit、MemoryAnswer 和多模态 ContentPart 统一数据契约。
- `datasets/base.py`：正式 `DatasetAdapter` 抽象；数据集插件不再写死在 CLI。
- `datasets/locomo_refined.py`：LoCoMo-Refined 数据入口，保留多模态 caption/query 和证据 ID。
- `datasets/longmemeval.py`：LongMemEval 入口，保留原始 session/evidence。
- `judges/base.py`：正式 `DatasetJudge` 抽象，评分规则与数据集绑定。
- `judges/locomo_refined.py`：包装 LoCoMo-Refined 官方 scorer。
- `judges/longmemeval.py`：按 abstention、preference、knowledge-update、temporal 等题型生成官方 Judge prompt，并严格解析 yes/no。
- `adapters/base.py`：能力驱动的记忆协议；除 ingest/search 外支持 answer、reflect、异步 ready、update/delete、profile/graph 和 drill-down。
- `adapters/tencentdb_http.py`：TencentDB Gateway 的 capture/session-end/recall 适配。
- `adapters/external_process.py`：第三方隔离环境 JSON stdin/stdout 协议。
- `adapters/full_context.py`、`adapters/bm25.py`：统一基线。
- `runner.py`：无 Token 的检索 Gate 和 Evidence Recall，可按声明调用 reflect/ready 生命周期。
- `native_runner.py`：框架自带 answer 的独立 native 赛道，防止和统一回答榜混算。
- `answering.py`：统一 OpenAI-compatible 回答层；默认纯文本，显式开启时传递图片 ContentPart，密钥只读环境变量。
- `plugins.py`、`profiles.json`、`datasets.json`：框架、数据集和 Judge 的声明式发现，无需修改 CLI 分支。
- `scoring.py`：官方 LoCoMo-Refined scorer 子集包装；不把未选题当空答案。
- `report.py`：整体与分类检索、上下文、Token 报告。
- `doctor.py`：下载、commit 和许可证检查。
- `schemas/`：ContentPart、框架插件、数据集插件、外部协议和标准产物的 JSON Schema。

第三方进程协议：

```json
{"protocol":1,"operation":"prepare|ingest_session|finalize|search|answer|reflect|wait_until_ready|update|delete|get_profile|get_graph|drill_down|close","payload":{}}
```

搜索响应：

```json
{"hits":[{"content":"...","score":0.9,"source_ids":["D1:3"],"metadata":{},"parts":[]}]}
```

这个边界保证每个框架可独立使用 Python 版本、数据库、Docker 和模型依赖，同时上层评测、回答及 Judge 完全一致。

## 5. 当前验证结果

### 5.1 自动测试

- 16/16 单元测试通过，覆盖数据保真、BM25/全文、标准产物、断点续跑、上下文预算、source lock、计划器、validator、插件动态发现、能力拒绝、native 赛道、LongMemEval Judge 规则和多模态端到端保留。
- 所有新模块 `py_compile` 通过。
- LongMemEval 标准化文件：500 题、23,867 sessions、246,738 messages，逐物理行解析通过。
- 15 个注册条目中 11 个本地源码/项目条目可检查，全部下载框架 commit 与许可证已落盘。

### 5.2 无 Token 检索 Gate

使用 LoCoMo-Refined `conv-26` 的前 4 题：

| Arm | Evidence Recall | 平均上下文字数 | 说明 |
|---|---:|---:|---|
| Full Context | 1.0000 | 77,749 | 上界参考，不是记忆框架 |
| Local BM25 | 0.5000 | 998.25 | 确定性词面基线 |
| MemPalace default | 0.5000 | 1,270.50 | MiniLM + Chroma + 框架原生 hybrid rerank |

MemPalace 与 BM25 的均值相同，但命中的题不同：BM25 命中 q0000/q0001，MemPalace 命中 q0000/q0002。因此 4 题只能用于验证链路和定位互补召回，不能用于宣布框架排名。这一步已经证明统一适配器能在不调用回答模型/Judge 的情况下定位“证据没有召回”与“有证据但答案没生成对”两类问题。

MemPalace 运行产物：

- 隔离环境：`benchmarks/framework_eval_runtimes/mempalace/.venv/`
- embedding 缓存：`benchmarks/framework_eval_runtimes/mempalace/models/`
- conversation store：`benchmarks/framework_eval_runs/stores/mempalace/`
- 检索明细：`benchmarks/framework_eval_runs/mempalace-smoke/retrieval.jsonl`
- 汇总：`benchmarks/framework_eval_runs/mempalace-smoke/summary.md`

统一回答 Gate 已尝试调用 `qwen3.8-max`，服务端连续返回 HTTP 429，因此没有把失败的空答案提交 Judge，也没有伪造 QA 得分。检索结果不受回答额度影响。

### 5.3 官方 scorer 兼容性 Gate

使用已有 `q48-general-v1` TencentDB 预测，调用官方 scorer 的纯 F1/BLEU 子集路径：

- 48/48 题成功评分。
- 独立按词面最优参考答案选择时：F1 0.4050、BLEU 0.3484。
- 旧的 `llm + f1 + bleu` 同时评分报告为 F1 0.3915、BLEU 0.3417。

差值是官方 scorer 在多参考答案时的设计造成：启用 LLM Judge 时先按 Judge/F1/BLEU 顺序选择同一个 matched reference；纯词面模式直接选择 F1 最优 reference。正式可比实验必须让所有 Arm 在一次运行中启用相同的 `llm f1 bleu`，不能混用这两种口径。

## 6. 公平评测约束

1. 不使用问题、标准答案或 evidence label 参与建库、摘要或检索调参。
2. 调参只在显式 dev split；最终报告保留 held-out split。
3. 每个 conversation 独立 namespace，禁止跨样本污染。
4. 所有框架接收相同 Session 顺序、时间、speaker 和多模态 caption。
5. 同一赛道固定回答模型、Judge、Prompt、Top-K 和上下文字符/Token 预算。
6. 原生配置与统一预算配置分两条赛道；不为了一个框架暗中扩大 Top-K。
7. 框架自带针对 LoCoMo 的 heuristic/hybrid 配置单列为 benchmark-tuned，不冒充通用默认配置。
8. OSS、托管 SaaS 和论文自报结果分开；无法下载的实现不填本机成绩。
9. 失败、超时和空召回都进入分母，不静默丢题。
10. 所有运行保存源码 commit、配置、原始 retrieval、prediction、scored 输出和 Token ledger。

评测分轨：

- `unified`：固定 Top-K、最大上下文预算、回答模型和 Judge，回答“在同等预算下谁更好”。
- `native`：保留框架官方默认/推荐配置，回答“该开源实现按官方方式能达到什么效果”。

两个赛道分别报告，禁止把 native 的更大召回预算放进 unified 排名。

## 7. 接入与运行顺序

按依赖和复现成本分批：

1. MemPalace：本地后端、无需记忆 LLM，先跑检索 Gate，再跑统一 QA/Judge。
2. Microsoft Memora、LightMem、HyperMem：都有原生 LoCoMo/LongMemEval 路径，做统一格式桥接。
3. Mem0 OSS、EverOS：启动独立服务后接 HTTP/进程桥。
4. Hindsight、MemMachine：独立 Server + 数据库，固定容器版本。
5. MemOS：Neo4j + Qdrant，依赖最重，最后接入。

每个框架先执行 1 conversation / 4 questions Gate。Gate 必须依次通过：数据条数、namespace 隔离、建库完整性、非空召回、Evidence Recall、统一回答、官方 Judge、Token/延迟落盘。只有通过后才扩到 48 题，再扩到 LoCoMo-Refined 1,382 题和 LongMemEval 500 题。

## 8. 当前尚未宣称完成的事项

- 已下载不等于已经复现：除内部基线/TencentDB 适配外，第三方框架仍需逐个隔离运行时和真实 Gate。
- 尚未产生同机、同回答模型、同 refined Judge 的全框架横向最终表。
- MemoraX AI 没有可验证开源实现，不能满足“下载源码运行”。
- 全量运行会消耗各框架建库 LLM、统一回答模型及 Qwen3-14B Judge 额度，应在小 Gate 通过后再启动。

## 9. 本轮停止点

用户将当前阶段明确限定为“搭建通用评测模块，不实际跑”。因此：

- 没有继续调用回答模型或 Judge。
- MemPalace q48 离线检索扩展在 34/48 时主动中止；部分 JSONL 保留在 `benchmarks/framework_eval_runs/mempalace-q48-general/retrieval.jsonl`，不进入正式结果表。
- `RetrievalRunner` 已支持 `--resume`，以后可以跳过这 34 个已完成 qa_id；在重新开始前仍应先通过 `validate` 并确认同一 commit/config/split。
- 当前正式可引用的第三方运行只限 1 conversation / 4 questions 接口 Gate，不作为总体效果结论。
- `plan` 已生成 14 个 Arm（含全文/BM25 基线）× 2 个数据集 = 28 项的 `unified` 矩阵；8 项只表示适配层就绪、未执行，其他项逐项标出 bridge 或服务依赖。
