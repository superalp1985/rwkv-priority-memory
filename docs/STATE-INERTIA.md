# 让上一轮的 state 参与这一次记忆判断

2026-11-19 的研究补充。用户要求每次自评结合当时的 state，评分后更新状态，供下一次判断继承。提议先用 ChatRWKV 的原生 `forward` 做这个闭环。一次对话仍由同一组权重处理。哪一份状态承接评分反馈，是本方案的设计选择。

## 官方已经提供的部分

| 现有能力 | 已核对的位置 | 本项目如何用 |
|---|---|---|
| 原生递推和读出 | RWKV-7 论文描述向量门控及上下文学习率。x070 参考实现第 143–150 行先保留、修正旧矩阵并加本步写入，再读出本步结果。`forward` 同时沿用前一位置的 TimeMix 和 ChannelMix 状态。[1][2] | 直接从非空 state 续算。没有 token 经过时，张量不会自行按墙钟衰减。距上次用户提及的天数交给已有排序策略。 |
| 分叉与继续对话 | ChatRWKV 的 `API_DEMO.py` 第 54–62、86–88 行说明分段续算与深拷贝。x070 的 `model.py` 第 284–302 行定义三项逐层状态。聊天演示实际保存 `forward` 返回的 state，旧版 `v2/chat.py` 也演示保存和恢复。[3][4][5] 旧版函数见第 167–179 行。[16] | 用完整的独立副本。每个候选从同一轮边界开始，评分模板、生成的 JSON 和采样状态只留在临时分支。 |
| 高性能路径 | Albatross `faster3a_2607/app4b.py` 第 372–376、507–524 行已有一次 prefill 后将 shift、WKV、elapsed 复制到批解码槽的例子。引擎的 `forward` 返回 logits，状态由调用方持有。DeltaLog 另维护依附存储地址的阶段和工作区。[6][7] | 留到性能阶段复用。若迁移，先在单设备、状态完整物化的路径核对分叉。`copy_state_to_batch` 是该 demo 的辅助函数，不能推定适用于进行中的 DeltaLog 周期或多卡分段布局。 |
| 贡献入口 | RWKV-LM 的 README 把对话示例指向 ChatRWKV，把高效推理指向 Albatross，还把 state-tuning 定义为训练初始状态。BlinkDL 在 #270 解释了同一前缀续算多个后缀，在 ChatRWKV #199 指向自定义生成示例。[8][9][10] | 本次复用 `rwkv.model.RWKV.forward`。学习初始 state、实现状态服务器、修改 CUDA 衰减或 `PIPELINE.generate` 返回类型都不在示例依赖中。 |

## 一轮怎样读、评、写

设 `D` 是负责回复的原生 RWKV state，`M` 是同一模型负责连续评估的原生 RWKV state，`K` 是按四因子选出的、有来源和版本的有限事项卡。`M` 是模型张量，`K` 是可核对的文本和元数据。二者都按授权主体和会话管理。

1. **从此刻的状态开始。** 进入实验时，从同范围、同权重、同模板的聊天边界复制一份给 `M`。每轮截止到当前用户最后一个 token，尚无本轮助手答复。初次启用的边界如果已经包含本条用户消息，就从那里分叉，不再输入一次。在连续评估模式，已有 `M` 跟随此后真实的用户和助手 token，每条各一次。若当前 state 混入未授权的上下文，从获准历史重建，或采用文本基线并标记降级。
2. **分叉给分。** 每个候选的四个判断从同一冻结边界分别开始。模型评遗忘后果、用户表达强度、独立提及的强化程度和新近度。事项 ID、用户原文短引、去重次数及 UTC 时间是输入证据，不是模型生成的解释。上文已在 state 里，不再重复整轮对话。所有候选评完才选卡。
3. **按证据更新。** 用已有 `rank_memories` / `select_seed` 合并 `I`、`E`、独立用户提及和新近度，得到 `K_t`。只有可核对且仍生效的事项才有资格。原生只读分叉实验到这里更新外部评分和卡，下一轮重新从 `D` 分叉。**带反馈的实验再走一步**：从本轮评分前保留的 `M` 边界（未输入评分后缀），用原生 `forward` 输入新入选或有实质变更的卡片，得到下一轮将继承的 `M`。卡片仅含用户支持的事实、范围和版本。评分数字、上次置信度与模型评语均留在日志。有效卡未变时，输入零个卡片 token。
4. **继续真实对话。** `D` 只处理真实对话流和已有的会话开场卡。工作线程先提交本轮卡，再给 `M` 输入已实际发出的助手答复与下一条用户消息，各一次。前台不等评分；后台允许积压，但下一轮评分必须等前一轮反馈提交或明确按失败策略跳过，不能把从陈旧 `M` 算出的分冒充连续结果。提交键包含会话、轮次、事项版本及策略版本。版本过期的结果不能覆盖更新结果，重试不能重复写卡。异步提交、重建、并发隔离是设计契约，当前 CPU 探针只跑顺序的单次迁移。

**卡片失效或退出。** 明确更正先使旧版本退出，不等新评分。若需继续使用评估状态，从不含失效卡写入的快照重放获准历史和当前有效卡，避免复写快照已包含的有效卡。找不到合适快照就重建或降级到文本基线。卡片预算限制的是显式保留的事项；原生 state 仍可能从真实历史记得曾提到它。撤回授权必须连受影响的原文、状态和快照一并处理，不能只删卡。[删除与恢复](PLAN.md#3-有限卡片和完整档案是两种存储承诺)

```text
本轮用户话语 → 已含旧历史的评分边界 → 独立副本 + 评分后缀 → 评级及原话依据
                                    ↓ 核验后用四因子选卡
评分前保留的干净边界 + 有变化的有效卡 → M 更新 → 真实助手答复 → 下一条用户话语 → 下次自评
```

例如，同一句“就按九点吧”，前面可能在编练习题，也可能在定每周交接。模型应从各自的 state 理解它。此后用户说“下周起改十点”，周交接的旧版本退出，十点成为新的候选。模型看见上次的卡、自己复述约定或检索过原文，均不能冒充一次新的用户提及。`uncertain`、超时或证据不符时，保留待评标记，只对同一仍有效版本沿用已核验评级。保存分仍只管有限工作记忆，获授权的原文独立归档，精确回忆按需查档。[自评协议](APPRAISAL.md) · [四因子策略](PLAN.md)

这里所谓的惯性可以被测量：固定当前话语和量表，只改变此前 state，观察当前判断。再固定此前 state，只改变这次评分选中的有效卡，观察下一次判断。单纯把上次分数做指数平均回答不了其中任何一个问题。一旦 `M` 写入了额外的卡，就不能再用 `D` 的后续状态代替它；同步真实对话 token 的额外计算也要单列。是否让反馈进一步影响**本次前台回复的张量**，属于更强的实验。可在 `D` 的另外一份副本上对照反馈后的下一次正常回复，单报质量和偏移。本轮优先验证连续评估状态 `M`，直接写主对话 state 的效果仍待测。

## 现有实验能说明多少

本项目用固定提交的原版 ChatRWKV x070，CPU float32、单个随机初始化的 block、32 个无文本含义的 token ID。使用官方 `forward([token], state)` 和整份 state 的 `deepcopy`。本地只生成临时张量夹具。实验借用两个输出 token 的 logit 差作读数，并调用项目已有的四因子排序和一格选卡函数。[上游实现][model] · [实验代码](../probes/state_branch.py) · [完整输出](../reports/native-state-branch.json)

| 配对输入 | 实际观测 |
|---|---|
| 改早先前缀，同一当前 token 和评分后缀 | 两个标签的 logit 差从 0.574 变为 0.869。两边仍选同一个离散等级。复制原历史重跑与从快照续算逐位一致。 |
| 固定前缀，强制把目标事项评为 0 或 4，另一项与其余三因子固定 | 已有选择器在同一个 token 的预算里分别选了卡 16、17。只在评估分支写入相应卡，再输入相同回复、下一用户话语和评分后缀，下一次标签 logit 差为 0.480、0.580。同卡重放逐位一致。 |
| 开启评分分支与完全跳过评分 | 所测前台正常续写的 logits 和全部状态张量逐位一致。另跑了随机头读出 4、实际选卡 17 的完整路径。 |

这证明本例的两条因果路径可运行。随机权重给出的等级没有语义，logit 差也不是记忆收益。带可选原生集成项的本机测试共 24 项通过。复跑方式和范围见[验证记录](VALIDATION.md)。

## 用真实模型决定要不要提 PR

当前实现对比三个评估 state：`read_only` 继承真实对话但不接收卡；`replacement` 每轮从空 state 重放最近三个完整问答和当前用户话语，再恢复有效卡；`feedback` 持续继承评估 state 并更新卡。三者都不改模型权重。重建和循环在历史窗口、卡片输入位置和次数上有差异；整组偏好差不能单独归因为某一个张量操作。下面列的是研究对照，当前实现与其他可选扩展应分开：

| 组 | 唯一要检验的变化 |
|---|---|
| 只看当前句 / 近三轮文本 | 从空 state 读文本。最近三个已完成问答加当前用户话语是当前 replacement 实现。 |
| 继承 state，只读分叉 | 从真实用户边界 `D` 续算，评完丢弃临时分支。另设完整相同 token 前缀从零重放的控制，先核对 replay 与 fork 的输出差和计算量。要检验四轮外的线索，单列该子集。 |
| 继承 state，加评分反馈 | 首轮与只读分叉相同。从第二次评分起只改变 `M` 是否接收上轮有效卡，所有真实对话 token 相同。再换成等长的无关测试卡作对照，区分卡片的内容和额外 token 数。 |
| 可选直接影响回复 | 仅在复制出的前台分支输入同一有效卡，和未写卡的原生回复比较。分别检查当前任务、无关闲聊及更正后的答复。 |

主指标是**真人认为此时该带着什么记忆继续使用，以及实际回复如何用它**，不是生成答案的正确率。先盲评同预算的工作记忆选项；随后在三个策略下生成真实回复，再盲评自然沿用、纠正后切换、无关打扰和需要精确原话时查证。按整段对话比较并保留平局、分歧、弃评。问答正确率只作次要结果。标签、未来问题和其他策略的输出不得进评分提示；作者标签和模型置信度都不等于人类偏好。当前[合成试标包](HUMAN-RATING-PILOT.md)还没有真人评分，当前回放也不生成答复，因此这些端点尚未测量。

## 查重和最小贡献

2026-11-19 UTC 读取 BlinkDL 的 `RWKV-LM`、`ChatRWKV`、`Albatross` 三个 `main` 的提交和源码，用 GitHub REST `/issues?state=all&per_page=100` 翻到末页，按仓库和编号去重。依次取得 333、215、11 个公开条目，共 559 个，其中含 158 个 PR。按中英文标题和正文筛评分、工作记忆、状态分叉与状态探针，并核读相邻讨论。该范围内未发现同时实现“沿用 state 做事项自评、评分驱动下一评估状态、固定预算保存排序”的 PR。这次核查覆盖指定仓库和公开条目，发 PR 前应刷新分支和在审列表。[可复查的端点、提交、关键词与命中编号](../reports/upstream-state-scan.json)

- 与维护者已有的前缀复用答复衔接。ChatRWKV #199 已提出给 `PIPELINE.generate` 暴露状态，维护者给出的路径是现成的自定义生成示例。ChatRWKV PR #87（查阅时 open）处理回复出现角色名时恢复 state。两者都提供了明确的复用边界。[9][10][11]
- RWKV-LM #349 已有社区成员讨论状态轨迹和因果探针。维护者提及的社区项目用状态做语义投影，目标包括语义相似度。LUFY 用历史对话摘要评估以后有没有用。因此无需再造嵌入头、通用探针或把“模型看上下文评分”当成首创。[12][13][15]
- **首选的上游候选是 `BlinkDL/ChatRWKV` 的应用示例。** 候选标题 `Example: score a memory candidate from a forked RWKV-7 chat state`。若留出盲测的状态反馈组在同预算下比只读分叉多保留有用事项，且更正和开销可接受，先与维护者确认示例放置位置。最小 PR 为一个可选示例和同前缀重放、主线不受污染的回归。示例调用现有 tokenizer、`forward` 和副本，演示评分后下次续算的边界。正式评分量表、四因子政策、SQLite、卡片管理和长对话数据留在独立项目。若已有示例覆盖这条路径，或状态反馈相对只读分叉没有可靠收益，就保留独立实验和结果。若遇到所有公开接口都无法满足的物化/复制问题，再带最小复现向对应后端讨论。当前为本地候选提案。

[model]: https://github.com/BlinkDL/ChatRWKV/blob/2e2bb1cd390cefbedae0a89f4a343f6f754d6621/rwkv_pip_package/src/rwkv/model.py#L284-L302

## Sources

[1] https://arxiv.org/html/2503.14456v2 — Peng et al., RWKV-7 Goose, §3
[2] https://github.com/BlinkDL/RWKV-LM/blob/2eebb381a6160837ac45896c60d6963ef3cda929/RWKV-v7/rwkv_v7_demo_rnn.py — RWKV-LM · x070 TimeMix and forward
[3] https://github.com/BlinkDL/ChatRWKV/blob/2e2bb1cd390cefbedae0a89f4a343f6f754d6621/API_DEMO.py — ChatRWKV · x070 split forward and state copy
[4] https://github.com/BlinkDL/ChatRWKV/blob/2e2bb1cd390cefbedae0a89f4a343f6f754d6621/rwkv_pip_package/src/rwkv/model.py — ChatRWKV · x070 Python state and readout
[5] https://github.com/BlinkDL/ChatRWKV/blob/2e2bb1cd390cefbedae0a89f4a343f6f754d6621/API_DEMO_CHAT.py — ChatRWKV · custom chat generation
[6] https://github.com/BlinkDL/Albatross/blob/545dd1870ea6cf424d7f7d4da16b7f87491405db/faster3a_2607/app4b.py — Albatross · prefill and copy_state_to_batch
[7] https://github.com/BlinkDL/Albatross/blob/545dd1870ea6cf424d7f7d4da16b7f87491405db/faster3a_2607/rwkv7_fast_v3a.py — Albatross · state layout and DeltaLog phase
[8] https://github.com/BlinkDL/RWKV-LM/blob/2eebb381a6160837ac45896c60d6963ef3cda929/README.md — RWKV-LM · upstream chat and inference links
[9] https://github.com/BlinkDL/RWKV-LM/issues/270 — RWKV-LM #270 · maintainer on shared prefill
[10] https://github.com/BlinkDL/ChatRWKV/issues/199 — ChatRWKV #199 · maintainer on custom generation
[11] https://github.com/BlinkDL/ChatRWKV/pull/87 — ChatRWKV PR #87 · recover state after role marker
[12] https://github.com/BlinkDL/RWKV-LM/issues/349 — RWKV-LM #349 · contributor experiments and maintainer pointers
[13] https://github.com/cgisky1980/rwkv7-state-embedding/blob/e51c24ec2a3491c94cec1e4aca503fe298488df7/README.md — Community · RWKV-7 state semantic embedding
[14] https://github.com/BlinkDL/RWKV-LM/issues/353 — RWKV-LM #353 · carried-state numerical report
[15] https://aclanthology.org/2025.dnd-16.6 — Sumida et al., LUFY · importance judged from dialogue
[16] https://github.com/BlinkDL/ChatRWKV/blob/2e2bb1cd390cefbedae0a89f4a343f6f754d6621/v2/chat.py — ChatRWKV v2 · save/load dialogue state
