# 中文座舱记忆数据契约

本仓库的通用边界是 `Conversation / Session / Message / Question`。新数据集可以实现
自己的 `DatasetAdapter`，也可以直接使用下面的两个 JSONL 文件。每行必须是一个完整
JSON 对象，文件编码为 UTF-8。

## conversations.jsonl

```json
{"sample_id":"car-user-001","metadata":{"timezone":"Asia/Shanghai"},"sessions":[{"source_session_id":"car-user-001-s001","date_time":"2026-09-01T08:30:00+08:00","messages":[{"role":"user","speaker":"张先生","content":"工作日早上空调设成二十二度"}]}]}
```

必需字段为 `sample_id`、非空 `sessions`、每个 session 的稳定 ID、带时区时间戳，
以及非空 `messages[].role/content`。session ID 可以命名为 `source_session_id` 或
`session_id`。消息没有 `message_id` 时，适配器按 `session_id:001` 的规则稳定生成。

多人输入必须提供 `messages[].speaker` 或能在原文中无歧义识别的人名。生产数据应
显式提供 `message_id`；它是更新、撤销、证据链和幂等重放的来源锚点。

## questions.jsonl

```json
{"qa_id":"car-user-001-q001","sample_id":"car-user-001","question":"张先生工作日早上希望空调设成多少度？","answer":["22度"],"category":"preference","answer_session_ids":["car-user-001-s001:001"],"question_date":"2026-09-03T09:00:00+08:00","is_abstention":false,"metadata":{"timezone":"Asia/Shanghai"}}
```

评测必需字段为 `qa_id`、`sample_id`、`question` 和数组形式的 `answer`。证据可以放在
`answer_session_ids` 或 `evidence`；适配器统一映射为 `evidence_ids`。时间问题应提供
带时区的 `question_date`，拒答样本应设置 `is_abstention=true`。

答案与 evidence 只用于评测，绝不能在 seed、检索或回答阶段作为输入特征。线上推理
可以没有 `questions.jsonl`，但没有金标就不能计算答案准确率和 Evidence Recall。

## 验证

```bash
python3 scripts/release/validate-cockpit-dataset.py /path/to/dataset
```

JSON Schema 位于 `schemas/cockpit-conversation.schema.json` 和
`schemas/cockpit-question.schema.json`。内置验证器不依赖第三方 Python 包，并额外检查：

- 会话、消息和题目 ID 唯一且引用关系完整；
- 时间戳包含时区；
- evidence 不跨 conversation；
- 角色合法且内容非空。

换数据集后必须使用新的隔离 namespace 并重新 seed。Recovery33 记忆只能用于仓库中
冻结的 v7 数据，不能作为其他数据集的构建结果。
