# 通用记忆框架评测系统

本目录把数据集、记忆实现、回答模型和 Judge 分开。框架和数据集均由清单动态发现，CLI 不再包含 LoCoMo/LongMemEval 或 BM25/TencentDB 的分支判断。

## 300–500 条配对选择性记忆协议

`selective_memory_protocol.py` 固定五个 TencentDB 消融实验使用的相同题目：
完整 L0/L1、L0 Top-1、自适应 Top-1→Top-3/L1、选择性 L1 构建，以及
选择性 L1 + 缓冲式 L2/L3。SLURP 按完整说话人历史抽样，KVRET 使用 420
条均衡派生集，LongMemEval 按题型分层抽样；抽样不读取答案或 evidence ID。

```bash
python -m benchmarks.framework_eval.selective_memory_protocol
```

固定清单位于 `benchmarks/framework_eval/protocols/selective-memory-v1/`，应随
源码提交；运行时 Store 和输出仍保存在被忽略的
`benchmarks/framework_eval_runs/` 下。

420 条 KVRET 派生集可在另一台机器上无模型、确定性重建：

```bash
python -m benchmarks.framework_eval.datasets.prepare_kvret_memory \
  --input benchmarks/data/KVRET/raw/kvret_train_public.json \
  --input benchmarks/data/KVRET/raw/kvret_dev_public.json \
  --input benchmarks/data/KVRET/raw/kvret_test_public.json \
  --output-dir benchmarks/data/KVRET/normalized-420-v1 \
  --groups 10 --per-domain 140
```

协议先运行 Phase 1 的 SLURP/KVRET。LongMemEval 300 条包含约 1.47 亿源字符，
被放在 Phase 2；只有选择性构建在短指令集通过质量门槛后才允许启动，避免再次
出现全量建库快速耗尽额度。

自适应检索由 `TDAI_EVAL_RETRIEVAL_POLICY=adaptive` 开启。它先请求一个 L0
锚点及同 session 的有界后续窗口；单事实高置信度问题直接返回，更新、聚合、
多事实或低覆盖问题才追加 Top-3/L1。该判断是确定性规则，不增加一次 LLM 路由
调用。结果中的 `retrieval_route`、`adaptive_reason`、
`adaptive_search_calls` 和 `context_budget_chars` 用于分别核算成功率、延迟和
上下文成本。
完整 L0/L1 基线也对命中的 L0 锚点应用相同的 `before=1/after=12` session
窗口，避免 slot-filling 对话只命中 assistant 追问却漏掉紧随其后的用户补槽；
Top-K 和 3000 字符预算保持不变。

选择性构建由 `TDAI_EVAL_L1_WRITE_POLICY=cockpit_selective_v1` 开启。策略把
session 路由为 `profile/event/record/transient`：只有稳定偏好、状态变更事件、
用户纠正和持久化工具结果进入原生 L1；Record/Transient 仍完整写入 L0。当前
公开 `/v3/conversation/add` 没有 `extract=false` 字段，评测适配器通过把未选中
轮次的管线角色降为 `assistant` 来抑制 L1 notify（公开 v3 schema 仅接受
`user/assistant`），同时可逆内容头仍保留原始 speaker、source ID 和时间。这一
模拟没有修改 MemoryCore 源码或提取提示词；若
质量门槛通过，再将其产品化为显式服务端写入选项。

针对座舱零碎指令，选中的多个 source session 在写入前按 conversation 做 L1
微批；trace 保存原始 source ID/session 边界，因此相邻窗口仍只在原会话内展开。
MemoryCore 只看到同一条 transport cursor；一次 transport 写入最多 128 条，内部
按 32 条小批连续提取，从而适配能力较弱的小模型。L2 延迟窗口设为 15 分钟，确保
同一 cursor 的连续 L1 小批先排空，再基于完整 L1 快照只重建一次 profile；在线
检索不等待该窗口。这仍是适配与运行配置层优化，没有修改 MemoryCore 源码或提示词。

每个 session 的决定写入 `TDAI_EVAL_CONSTRUCTION_TRACE` 指定的 JSONL，runner
同时返回选择率、L0 写入数和各记忆类型计数。固定 Phase 1 样本上的无模型预估
选择率为 SLURP 32.0%、KVRET 48.3%，即分别减少约 68.0% 和 51.7% 的 L1
session 触发；最终 token 节省只以真实服务 ledger 为准。

生产/影子运行应同时设置 `TDAI_EVAL_TYPED_EPISODE_INDEX_PATH`。适配器会把可信的
typed episode 双写到 WAL SQLite 索引，按
`conversation/team/agent/user/task/source-session` 精确隔离；相同重放幂等，更新使用
单调 revision，并支持 TTL、显式失效和 supersede。检索进程冷启动后可直接读取该
索引，不再依赖 Python 进程缓存或每次解析 JSONL。索引缺失、损坏、过期或 namespace
不一致时不会返回捷径答案，而是回退正常 L0/L1 检索；生产环境再设置
`TDAI_EVAL_TYPED_EPISODE_INDEX_REQUIRED=true`，使写入异常显式阻断。旧 trace 会在
首次读取时按当前隔离清单做一次安全迁移，跨 namespace 的记录不会被重新绑定。
对于 conversation batch/并发 flush，episode 先以不可读 pending 状态写入；只有
对应 L0/L1 请求全部成功才激活并追加 source-commit marker。进程崩溃或后端失败留下
的无 marker 记录不会被冷启动迁移，从而避免 sidecar 早于原始证据可见。

readiness 使用 trace 中的隔离信息把源 session 映射回 MemoryCore transport
session，仅等待当前 conversation 的 L1/L2/L3；同一 worker 中其他数据集、store
group 的后台任务不会污染该臂延迟。若 trace 或服务端 session 列表不完整，则
安全回退到全局队列清空，避免在构建未完成时读取。
L2/L3 的 settle 窗口锚定在该 scope 最后一个 L1 任务完成时，任务排空后不再
重复等待一整段 L2 delay。

最后一组使用
`configs/tencentdb_selective_async_l23.env`。MemoryCore 原生异步定时器把产生 L1
的脏事件缓冲 15 分钟后批量推进 L2/L3；适配器不修改 MemoryCore 源码或提示词。
由于原生 L2/L3 使用 agent 级分布式锁，而单个 conversation 会同时产生多个
session task，该运行配置暂时使用单 worker：这避免多个未按 agent 分组的 worker
竞争同一把锁、超时后丢任务。未来若服务端提供 keyed worker，可恢复跨 agent 并发。
高置信路径仍只访问 L0，只有确定性路由判为 fallback 时才按
`L0 3 + L1 2 + L2 1 + L3 1` 读取，统一 Top-K 为 7、上下文预算为 2600 字符。
readiness 同样采用惰性路由：先读取同步落盘的 L0；只有 L0 窗口仍不足且确实要
追加某层时才访问它。座舱端到端配置的等待预算为 0：已生成的 L2/L3 自动注入，
未生成则以 L0 窗口/L1 立即回答，后台构建不会阻塞在线请求。若离线任务显式配置
正等待预算，等待时间通过 `adaptive_readiness_layers/seconds` 写入结果；缺失层则
记录到 `adaptive_unready_layers`。

`TDAI_EVAL_L23_READINESS_MODE=dirty_only` 根据构建 trace 判断 agent scope 是否
真正触发过 L1。已证明干净的 scope 不等待不存在的 L2/L3 文件；trace 缺失、损坏
或缺少该 scope 时会 fail closed，恢复为严格等待，避免把部分建库误当成完成。
分片建库时每个 shard 写独立 JSONL，检索进程可把
`TDAI_EVAL_CONSTRUCTION_TRACE` 指向包含这些文件的目录。结果中的
`adaptive_profile_hits/levels` 记录回退追加的层；`adaptive_search_calls` 只统计
动态 L0/L1 search endpoint，profile 读取耗时仍完整计入 `search_seconds`。

`selective_memory_ablation.py` 把这五组配置编译为可恢复的配对实验。默认 Phase 1
只包含 SLURP 440 条和 KVRET 420 条；LongMemEval 300 条必须显式指定
`--phase 2 --dataset longmemeval`。两个 store group 分别只建一次：前三组复用完整
L0/L1，后两组复用选择性 L1；每个 conversation 独立写 trace 和完成标记，失败时
不会自动重放可能已部分写入的 conversation。

```bash
# 先确认固定题目、预算、namespace 路径；不访问服务、不调用模型。
python -m benchmarks.framework_eval.selective_memory_ablation --stage plan

# 在同一评测用户下创建全新的 agent/task namespace，不复制旧记忆。
python -m benchmarks.framework_eval.selective_memory_ablation \
  --stage provision \
  --principal-manifest PREVIOUS_RUN/isolation.json \
  --principal-user-key-file PREVIOUS_RUN/.user-key

# 每个数据集先跑 4 题、五个 arm；核心 arm contains < 80% 时阻断大规模回答。
python -m benchmarks.framework_eval.selective_memory_ablation --stage smoke

# smoke 通过后才运行剩余建库、全量检索、回答、确定性 Judge 和对比报告。
python -m benchmarks.framework_eval.selective_memory_ablation --stage all \
  --principal-manifest PREVIOUS_RUN/isolation.json \
  --principal-user-key-file PREVIOUS_RUN/.user-key
```

`plan` 必须不带 `--dataset/--arm`，先冻结完整配对协议。后续 stage 可以重复指定
`--dataset` 或 `--arm` 只执行子集；编排器验证它们属于冻结 plan，但不会尝试改写
plan。这样可将 full L0/L1 与 selective async 分别放在 L2 延迟不同的隔离 volume
中运行，避免 full 基线的后台 L2/L3 消耗污染选择性构建成本。
包含 buffered L2/L3 的 `smoke/all` 会自动按数据集执行完整的
`ingest -> retrieve -> answer -> score`，上一数据集完成后才启动下一数据集，避免
全局 worker 队列造成跨数据集队头阻塞。拆分执行 `ingest/retrieve` 时则必须显式只
指定一个 `--dataset`，编排器会拒绝不安全的多数据集组合。
full store 启动时使用
`configs/tencentdb_full_l1_deferred_l23.env`，完成前三个 arm 后停止该 volume；
selective store 再使用 `configs/tencentdb_selective_async_l23.env`。两边的 isolation
manifest 分别保留自己的 user-key 文件，回答与评分产物仍合并到同一 run root。

运行 MemoryCore 前应加载 `configs/tencentdb_selective_async_l23.env`，否则编排器
记录的 15 分钟缓冲窗口与服务实际定时器不一致。最终
`comparison.json/.md` 同时报成功率、Evidence Recall、平均/P95 检索延迟、上下文
字符、回答 prompt token、快/慢路径比例和 L1 选择率。公开 v3 管线尚不返回逐
namespace 的真实构建 token，因此报告保留为 `null`，只接受服务端或供应商 ledger
导出的实际值，不用字符估算冒充 token 实测。

```text
datasets.json -> DatasetAdapter ---------> Conversation / Question / ContentPart
profiles.json -> MemoryAdapter capabilities -> canonical retrieval JSONL
                                               -> shared answer model
                                               -> DatasetJudge
                                               -> comparable report

具有 answer 能力的框架 ---------------------> native predictions JSONL
                                               -> DatasetJudge（独立 native 赛道）
```

## 四层扩展契约

1. `DatasetAdapter`：只负责把任意多 Session QA benchmark 规范化为 `Conversation` 和 `Question`，保留原始时间、角色、证据 ID 和模态，不包含某个记忆框架的逻辑。
2. `DatasetJudge`：评分规则归数据集所有。LoCoMo-Refined 使用其官方 scorer，LongMemEval 使用按题型区分的官方 yes/no Judge prompt；记忆框架不能替换评分口径。
3. `MemoryAdapter.capabilities`：框架声明 `ingest/search/answer/reflect/wait_until_ready/update/delete/profile/graph/drill_down` 的实际能力。运行器只调用声明过的能力，不要求所有框架伪装成同一种 API。
4. `ContentPart`：文本、图片、音频、视频和文件使用统一结构。默认文本赛道继续使用 caption/query；只有显式启用 `--multimodal` 时，统一回答层才把检索命中的图片 URI 发送给兼容模型。

`profiles.json`、`datasets.json` 和 `module:Class` 插件引用构成发现层。新增实现时改清单并提供插件类，无需修改 CLI。

## 为什么采用进程/HTTP 适配

Mem0、EverOS、MemOS、MemPalace、Memora、LightMem、Hindsight 等框架的 Python、数据库和模型依赖互相冲突。评测器不会把它们安装进 TencentDB 项目的主环境；每个框架使用独立 venv、uv 环境或 Docker，再通过 JSON 进程协议或 HTTP 适配器连接。

## 当前命令

```bash
python -m benchmarks.framework_eval.cli list

python -m benchmarks.framework_eval.cli doctor \
  --output benchmarks/framework_eval_runs/doctor.json

python -m benchmarks.framework_eval.cli sources

# 只生成框架 × 数据集实验矩阵，不启动服务、不调用模型
python -m benchmarks.framework_eval.cli plan \
  --split gate4 --track unified --top-k 8 --max-context-chars 20000 \
  --output benchmarks/framework_eval_runs/evaluation-plan.json

python -m benchmarks.framework_eval.cli retrieval \
  --adapter bm25 --conversation conv-26 --question-limit 4 --top-k 8 \
  --output benchmarks/framework_eval_runs/bm25-smoke/retrieval.jsonl

# MemOS OSS：独立 venv + general_text + 嵌入式 Qdrant。
# 环境重建方法见 benchmarks/framework_eval_runtimes/memos/README.md。
python -m benchmarks.framework_eval.cli retrieval \
  --adapter memos --conversation conv-30 --question-limit 1 --top-k 12 \
  --max-context-chars 20000 \
  --output benchmarks/framework_eval_runs/locomo-half-v2/memos/retrieval-gate.jsonl

# 对所有框架复用完全相同的既有 48 题 split
python -m benchmarks.framework_eval.cli retrieval \
  --adapter bm25 \
  --selection-manifest benchmarks/production_runs/results/q48-general-v1/selection-manifest.json \
  --top-k 8 --output RUN/retrieval.jsonl

python -m benchmarks.framework_eval.cli score-locomo \
  --input RUN/predictions.jsonl --output-dir RUN/score --metrics f1 bleu

# 通用入口：自动使用 datasets.json 中绑定的数据集 Judge
python -m benchmarks.framework_eval.cli score \
  --dataset locomo_refined --input RUN/predictions.jsonl \
  --output-dir RUN/score --metrics f1 bleu

python -m benchmarks.framework_eval.cli validate \
  --retrieval RUN/retrieval.jsonl --expected-count 4
```

加入 `llm` 指标时调用 LoCoMo-Refined 官方 refined Judge；其模型、URL 和密钥只从官方 scorer 支持的环境变量读取。子集实验会生成隔离的 `questions.selected.jsonl`，不会把未选题目当成空答案。

离线 `retrieval` Gate 不调用回答模型或 Judge，可在没有 Token 额度时验证数据规范、Session 隔离和检索产物。正式 QA 阶段将使用 LoCoMo-Refined 官方 Qwen3-14B Judge，并固定同一回答模型、Prompt、Top-K 和上下文预算。

### TencentDB 多方对话视角

TencentDB 核心默认面向“真人用户 ↔ Agent”，因此 L1 不把 Agent 自身输出当作用户长期记忆。LoCoMo 的 `assistant` 却代表第二位真人；v3 适配器在 `TDAI_EVAL_PERSPECTIVE_MODE=auto`（默认）下识别数据集的 `speaker_a/speaker_b`，为每位真人建立独立 agent + task namespace，并在各自 namespace 中把本人映射为 `user`、另一方映射为 `assistant`。task 隔离 L0/L1，agent 隔离 L2/L3。检索跨两个视角合并四层结果，并按 source id 去重重复的 L0。

isolation manifest 可预先提供，也可由适配器补建并持久化。若环境提供 `TDAI_EVAL_USER_KEY`，task 创建走正式 v3 metadata 鉴权；未提供时使用同一服务中仍受 Bearer/service-id 保护的兼容 v2 task 管理入口，密钥不会写入 manifest：

```json
{
  "conversations": {
    "conv-26": {
      "agent_id": "agent-26",
      "task_id": "fresh-base-task",
      "perspectives": {
        "Caroline": {
          "agent_id": "agent-caroline",
          "task_id": "task-caroline"
        },
        "Melanie": {
          "agent_id": "agent-melanie",
          "task_id": "task-melanie"
        }
      }
    }
  }
}
```

适配器还把原始 turn id、source time 和 speaker 写入统一可逆内容头，例如 `[D1:3] [source_time=2023-05-08T13:56:00Z] Caroline: ...`；L0 检索会把 source time 恢复成回答上下文的 `time=` 字段。设置 `TDAI_EVAL_MEMORY_LAYERS=L0,L1,L2,L3` 后，每题上下文除动态 L0/L1 外，还包含本地 BM25 选择的 L2 场景和两个视角的 L3 persona；默认配额为 L2 两条、L3 两条，均计入统一 Top-K 和上下文预算。readiness 要求三层异步队列持续静默，并确认每个 agent 的 L2/L3 文件真实存在。

### TencentDB 长对话检索配置

`configs/tencentdb_v3_long_dialogue.env` 提供不依赖数据集标准答案的长对话配置。它在适配层完成四件事：扩大后端候选池；按动态 Top-K 比例保留 L0；把语义锚点扩展为同 Session 相邻窗口；只对时间、聚合、情感和推断问题追加短小的目标人物原始对话块。语义命中另一人物的提问时，会在局部范围内优先选择目标人物后续回答。相对日期只在可确定时展开，模糊表达保持原样。

```bash
source benchmarks/framework_eval/configs/tencentdb_v3_long_dialogue.env
export TDAI_EVAL_ISOLATION_MAP=/absolute/path/to/isolation.json
export TDAI_API_KEY=...

python -m benchmarks.framework_eval.cli retrieval \
  --adapter tencentdb --dataset locomo_refined --conversation conv-26 \
  --top-k 26 --max-context-chars 50000 \
  --base-url http://127.0.0.1:8420 --skip-ingest \
  --output RUN/retrieval.jsonl
```

所有开关默认关闭，避免改变普通用户—Agent 业务路径。正式评测应保存所用 env profile、Top-K、统一上下文预算和 isolation manifest。聚焦块只读取已选语义锚点附近的公开 L0 历史，不读取 Question 的 `answers/evidence_ids/category`，也不调用额外模型。

旧的单视角 task/agent 已经形成有偏记忆，不能当作修复后的干净结果。正式对比必须先用 `python -m benchmarks.framework_eval.provision_tencentdb` 创建全新 namespace，再重新建库。普通用户—Agent 数据集默认保持单 namespace；可用 `TDAI_EVAL_PERSPECTIVE_MODE=single` 强制关闭，或用 `multi` 对没有 `speaker_a/speaker_b` 元数据但具有 speaker 字段的数据启用多视角。

### TencentDB 构建增强账本

`tencentdb_ledger.py` 是可选的适配层离线构建器，用于保留 MemoryCore L1
可能压缩掉的次要从句，并建立跨 Session 的集合、实体关系、保守计数和事件
去重。它只读取标准化后的 Conversation；不会读取 Question、参考答案、证据
标签或 Judge 输出。文本赛道只使用原文以及数据集已经提供的 caption/query，不
执行 OCR，也不会猜测图片中未被文字描述的内容。

```bash
python -m benchmarks.framework_eval.tencentdb_ledger \
  --conversation conv-26 --dataset-root LoCoMo_refined \
  --config benchmarks/production/locomo_config_qwen38.json \
  --concurrency 2 --resume --output RUN/ledger.json

source benchmarks/framework_eval/configs/tencentdb_v3_constructed_ledger.env
export TDAI_EVAL_ISOLATION_MAP=/absolute/path/to/isolation.json
export TDAI_EVAL_LEDGER_PATH=/absolute/path/to/RUN/ledger.json
export TDAI_API_KEY=...

python -m benchmarks.framework_eval.cli retrieval \
  --adapter tencentdb --dataset locomo_refined --conversation conv-26 \
  --top-k 16 --max-context-chars 30000 \
  --base-url http://127.0.0.1:8420 --skip-ingest \
  --output RUN/retrieval.jsonl
```

抽取、普通 rollup 和 reasoning rollup 都带版本号；`--resume` 会自动重建版本
落后的 checkpoint。检索时多个匹配事实压缩成一条带 source ID/source date 的
`L1X` 命中，只占一个 Top-K 槽位，并保留原始 event date 的时间粒度。该模式
没有改 MemoryCore 或数据集 Prompt，但它属于额外的 adapter-side
记忆表示，正式报告必须同时给出纯 TencentDB 与 TencentDB + ledger 的结果及其
额外建库 Token，不能把两者混为同一配置。

## 公平性要求

1. 每个 Conversation 使用独立 namespace/store。
2. 原始 Session 顺序、时间和消息角色保持一致。
3. 文本赛道统一使用数据集提供的 caption/query；视觉赛道通过 `ContentPart` 和 `answer --multimodal` 显式启用并单独报告。
4. 所有框架共享回答模型和 Judge；项目自报分数只作参考。
5. 同时报告建库 Token、QA Token、上下文 Token、延迟、错误和 Evidence Recall。
6. 托管版与 OSS 版分开记录，不能用托管成绩代表开源代码。

正式报告分两个赛道：`unified` 固定 Top-K 和上下文预算，用于直接横向比较；`native` 使用各框架官方推荐配置，用于复现其最佳实践。两个赛道不合并排名。

## 第三方代码

官方仓库以浅克隆方式保存在 `third_party/memory_frameworks/`。注册状态见 `frameworks.json`。`downloaded-needs-runtime` 表示源码已下载但专用依赖/数据库尚未启动，不代表已经复现官方分数。

MemOS 的统一文本赛道使用官方 `general_text` Memory、`universal_api`
Embedder 和 Qdrant 本地持久化模式，因此不需要 Neo4j。Neo4j 仍属于 MemOS
tree/graph 原生赛道的依赖，不能把当前 `general_text` Gate 描述成已复现完整图记忆能力。

## 新增一个框架

1. 在 `sources.lock.json` 固定官方仓库、commit 和许可证。
2. 在 `profiles.json` 声明运行时、服务依赖、官方入口和支持的数据集。
3. 实现 `MemoryAdapter`，或实现 JSON 进程协议桥，并在 `capabilities` 中只声明真实能力。桥不能读取标准答案或 Judge 输出。
4. 依次执行 `sources`、`plan`、4 题 `retrieval`、`validate`。只有 Gate 通过后才进入统一 Answer/Judge。

第三方桥的 stdout 必须只有一个 JSON 响应；日志写 stderr。所有 store、模型缓存和 venv 放在 `benchmarks/framework_eval_runs/` 或 `benchmarks/framework_eval_runtimes/`，不污染项目主环境。

## 新增一个数据集

1. 实现 `DatasetAdapter`，使数据集输出稳定的 conversation/question ID；若有图像、音频或附件，写入 `ContentPart`，不要塞入框架私有字段。
2. 实现 `DatasetJudge`，把官方指标、题型规则和逐题审计产物封装在该插件中。
3. 在 `datasets.json` 登记 `adapter`、`judge`、默认 root 和 modalities。此后 `retrieval` 与 `score` 会自动发现它。
4. 先运行纯本地结构验证和 1 conversation Gate；涉及回答/Judge 的步骤在额度可用且用户明确启动后再运行。

若一个框架原生完成“检索 + 回答”，使用 `native-answer`。其产物带 `answer_track=native`，不能与共享回答模型的 `unified` 成绩混为同一 Arm。

## 产物约定

```text
benchmarks/framework_eval_runs/<dataset>-<split>-<track>-<framework>/
├── retrieval.jsonl       # 原始命中、证据 lineage、延迟、上下文预算
├── predictions.jsonl     # 统一回答模型输出和 Token usage
├── score/                # 官方 Judge 原始逐题结果
├── summary.json
├── summary.md
└── validation.json
```

`retrieval --resume` 和 `answer --resume` 按 qa_id 跳过已完成项。任何失败或空召回都必须留在最终分母；不能通过删除失败行提高分数。
