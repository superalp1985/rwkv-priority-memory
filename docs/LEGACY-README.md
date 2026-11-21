# RWKV State Recall

> 历史实验入口。当前方案见 [RWKV Priority Memory](../README.md)。

Experiments in multi-factor state scheduling for RWKV-7, with local retrieval-augmented memory and benchmarks that separate state retention from retrieval.

研究问题：面对很长的对话，显式标记重要事实后，模型能不能**不查库也记得更好**？另一路把经授权的原文可靠落盘、按需检索。两条路分开验证，不把检索命中算成模型自身记忆。独立研究项目，并非 RWKV 官方组件。

**当前状态：立项与机制验证。** 已有可运行的 CPU/SQLite 探针；没有发布可安装的推理插件，还没有真实模型的评测成绩。`configs/pilot.json` 是实验协议，不是已经接入引擎的配置。

| 入口 | 用途 |
|---|---|
| [旧版中文方案](LEGACY-STATE-FOCUSED-PLAN.md) | 调研证据、公式与挂载点、数据生命周期、四组实验、门槛与工期 |
| [旧版验证记录](LEGACY-RWKV-VALIDATION.md) | 测过什么、如何复跑、哪几件事还不能宣称 |
| [旧版实验配置](../configs/legacy-state-pilot.json) | 权重与词表校验值、参数、基线及量化目标 |
| [旧版文案和 RFC 草稿](LEGACY-GITHUB-PROPOSAL.md) | 历史 GitHub 文案与未发布沟通草稿 |
| [单层探针实测结果](../reports/legacy-upstream-probe.json) | 固定上游源码、两步合成输入的状态与层输出差异 |

## 本地复跑

从项目根目录执行；主测试和两个演示仅需 Python 3 与带 FTS5 的 SQLite，不安装模型，不写入真实聊天：

```bash
python3 -m unittest discover -s tests -p 'test_legacy_*.py' -v
python3 -m probes.legacy_spikes.decay
python3 -m probes.legacy_spikes.archive
```

`decay` 显示同一笔写入在当前/后续步骤的区别；`archive` 在临时库演示重开后补索引、查询范围与删除。可选的真实上游单层探针另需 CPU PyTorch 和固定版本参考文件，见[复跑命令](LEGACY-RWKV-VALIDATION.md)。没有启动服务或下载权重的隐式步骤。

## 成功标准

对同一批未见过的对话，比较原生、仅调度、仅检索、二者合用。先看「仅调度 − 原生」，再看「组合 − 仅检索」。如果只有检索有收益，就交付本地原文记忆和基准，不修改推理内核。

仓库不包含权重、对话数据、运行日志、原提案私人副本或上游完整源码；`local/` 仅供本机使用，已忽略。项目自身的发布许可由所有者在对外发布前确定。
