# GitHub 建仓与上游沟通

> 历史草稿，仅供参考，不按本页命令发布。当前仓库名和 Description 见[新版建仓文案](GITHUB.md)。

**项目名：** RWKV State Recall  
**Repository name：** `rwkv-state-recall`  
**Description（英文，直接填 About）：** Experiments in multi-factor state scheduling for RWKV-7, with local retrieval-augmented memory and benchmarks that separate state retention from retrieval.

**中文描述：** 面向 RWKV-7 的多因子状态调度与本地检索记忆实验，用对照评测区分模型自身记住的信息和从原文找回的信息。  
**Topics（建议）：** `rwkv` `rwkv-7` `recurrent-neural-networks` `long-term-memory` `retrieval-augmented-generation` `benchmark`  
**定位：** 非官方独立实验；不承诺改善召回、零延迟、主干支持或与 RWKV-8 兼容。

## 新建空仓库

由仓库所有者在 GitHub 创建，公开性自行选择。为避免两份初始历史，建仓页先不要生成 README、`.gitignore` 或 LICENSE。当前未替所有者选定项目许可证；公开供他人复用前，建议审定 Apache-2.0 或其他适合的许可，按上游条款保留通知。模型、语料和第三方代码的许可要分别核对，本仓不发布模型、数据集或上游推理源码。

本机项目已经 `git init -b main`，没有推送。首次发布时，在本项目目录运行下列命令。尖括号是占位符，需要换成实际值；不要在 URL 中带 token。若本机尚无 Git 提交身份，只给本仓设置自己的 `git config user.name` 和 `git config user.email`（可用自己的 GitHub noreply 地址），不必改全局设置。

```bash
git status --short --untracked-files=all
git add .gitignore README.md docs/ configs/ spikes/ tests/ reports/
git diff --cached --check
git commit -m "Start RWKV State Recall research project"
git remote add origin <新仓库的SSH或HTTPS地址>
git push -u origin main
```

`local/`（含原始构想的私人副本）、对话、模型、索引和运行日志都不在发布清单；推送前仍须人工看一遍暂存区。项目代码和原始设计在本机是两份独立文件，不会回写桌面原稿。

## 准备发给上游的 Discussion / Issue（未发布）

**Title:** RFC: opt-in, inference-time retention control for RWKV-7 x070 (with isolated recall benchmarks)

> We are testing whether a small, bounded change to x070’s effective retention factor can improve delayed recall with frozen weights. The intervention acts on the previous matrix state; it does not address individual facts or replace the delta-rule term. Our first policy uses an explicit importance signal and a decaying post-write trace. The default path stays untouched.
>
> We plan to compare native RWKV, scheduler only, retrieval only, and both on identical histories. Persistence and retrieval live outside the inference library. A CPU probe with synthetic tensors checks the hook; language-model recall, GPU parity and overhead have not been established.
>
> Would a reference-backend example or a hook before the WKV call be preferable? What parity, quality and latency evidence would you require before reviewing an opt-in patch? We will share the pinned backend, checkpoint, ablations and negative results before proposing changes to a fused backend.

发布前再查对应仓库同主题 Issue/Discussion、贡献指南和模板。第一项独立成果是评测工具；第二项是本地归档/检索，留在应用层；第三项才是**通过门槛后**的最小后端 hook。上游不接纳不影响独立使用。不要预填 Issue 编号、维护者态度或 PR 合入日期。
