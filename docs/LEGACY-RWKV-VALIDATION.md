# 验证记录：机制已跑通，记忆效果未测

> 本页只记录旧版探针。四因子评分、落盘和适用范围见[当前验证记录](VALIDATION.md)。

本次验证于 2026-11-18 在 Linux 本机运行；仅使用虚构文本、合成张量和公开上游源码。未下载模型权重，没有改模型服务和系统依赖。

## 1. 日常复跑：无权重，无 GPU

在项目根目录执行以下命令。需要 Python 3 和编译进 SQLite 的 FTS5；用临时目录建库并在演示结束时清理。

```bash
python3 -m unittest discover -s tests -p 'test_legacy_*.py' -v
python3 -m probes.legacy_spikes.decay
python3 -m probes.legacy_spikes.archive
```

**结果：10 项通过。** 其中 6 项检查保留系数的恒等路径、单调有界性、因果顺序/半衰期、与序列内核编码的数学对应、空矩阵不放大写入、非法输入；4 项检查归档/任务整体回滚、重开补建与幂等、用户/会话和同意边界、已索引及未索引条目的删除与禁止复活。

标量演示把 1 写入空状态。在写入时两侧都是 `1.0`；下一步没有写入，原生为 `0.8`，候选为 `0.81717`。这个简化例不含秩一校正，不代表真实词的记忆强弱。

SQLite 演示实际读回：

```json
{"before_index": [], "pending_after_reopen": 1, "after_index": ["turn-1"], "other_user": [], "after_delete": []}
```

`put` 和待索引任务在同一事务；程序关闭并重开连接后调用 `index_one` 显式消费待办。检索是 FTS5 的词法匹配，不是向量检索。这证明所测事务、重开和逻辑隔离路径成立；未模拟进程硬杀/断电，未实现后台线程、FAISS、备份擦除或完整多租户服务。临时库中 `consent=False` 是默认值，测试的删除是逻辑不可见而非物理安全擦除。

## 2. 固定上游真实函数：可选 CPU PyTorch

参考文件是 `BlinkDL/RWKV-LM@2eebb381a6160837ac45896c60d6963ef3cda929` 的 `RWKV-v7/rwkv_v7_demo_rnn.py`，SHA-256 为 `a61f35716b2ef81fa1c97bfd7f67bccd78d3a8968d0570748d4631fecf885500`。脚本先验 SHA，只取未修饰的 `time_mixing__` 函数；不导入上游模块顶层、不启动 CUDA、不加载模型。原函数与候选函数共用同一组张量；候选只在实际保留系数算出后、构造外积前加有界的幂变换。[2]

本机已把固定文件放在 `local/upstream/rwkv_v7_demo_rnn.py`，该目录不进入 Git。其它机器可以先取公开源码：

```bash
mkdir -p local/upstream
curl -fL https://raw.githubusercontent.com/BlinkDL/RWKV-LM/2eebb381a6160837ac45896c60d6963ef3cda929/RWKV-v7/rwkv_v7_demo_rnn.py -o local/upstream/rwkv_v7_demo_rnn.py
```

用已经装有 CPU PyTorch 的解释器执行（本次测试版本 `2.7.0+cpu`，不需要给主环境安装依赖）：

```bash
CUDA_VISIBLE_DEVICES='' python -m probes.legacy_spikes.upstream_probe --source local/upstream/rwkv_v7_demo_rnn.py
```

| 观测项 | 结果 |
|---|---:|
| 测试范围 | 一个 TimeMix 层，2 个头，头宽 64，两个合成 token 向量，CPU float32，种子 2026 |
| 空 state 的第 1 步，上游与启用候选的四个返回张量逐位相同 | 是 |
| 非空 state 的第 2 步，候选关闭，与上游四个返回张量逐位相同 | 是 |
| 非空 state 的第 2 步，启用后 state 最大绝对差 | 0.04936647415161133 |
| 同步的层输出最大绝对差 | 0.009261727333068848 |
| 所有被测返回值有限 | 是 |

原始结果：[legacy-upstream-probe.json](../reports/legacy-upstream-probe.json)。这些数字是变化幅度，**没有“越大越好”的含义**。这里测试的是层输出，不是全模型 logits；不是 GPU 性能、自然语言召回或官方权重推理。数学等价测试也不能代替融合 CUDA、半精度舍入、分块边界和状态快照的回归测试。

## 3. 下一轮要补的证据

用锁定的词表/0.4B 权重跑 native、关闭调度和启用调度；分别记录近距证据可读性、留出集的远距召回、短文本 PPL、prefill/decode 延迟与显存。启用检索后要分别报告“证据找到了吗”和“读了证据答对了吗”，保留错例。此历史实验的判定条件见[旧版方案](LEGACY-STATE-FOCUSED-PLAN.md)。

## Sources

[2] https://github.com/BlinkDL/RWKV-LM/blob/2eebb381a6160837ac45896c60d6963ef3cda929/RWKV-v7/rwkv_v7_demo_rnn.py — BlinkDL · x070 reference RNN / RWKV_RNN.forward & time_mixing__
