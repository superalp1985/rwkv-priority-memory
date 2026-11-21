# GitHub 建仓文案

**项目名：** RWKV Priority Memory  
**Repository name：** `rwkv-priority-memory`

**Description（可直接粘贴）：**

> A memory design for RWKV that ranks what to keep by model-assessed importance, emotional intensity, repeated user mentions, and recency, with on-demand recall from a full local archive.

**中文：** 面向 RWKV 的对话记忆设计：模型根据上下文自评重要性，结合情绪强度、重复提及和新近程度维持日常工作记忆；获授权的对话完整留档，需要细节时才查。

**建议 Topics：** `rwkv`、`working-memory`、`long-term-memory`、`conversational-ai`、`retrieval-augmented-generation`。

拟向 `BlinkDL/ChatRWKV` 讨论一个可选的 `forward` 示例，展示从已计算的对话 state 评估事项，以及评分后何时更新下一轮的评估状态。提交的判断依据是固定预算下多留下了哪些真正有用的事项，以及额外成本。查重范围、相邻 PR 和维护者的原话见[上游调研](STATE-INERTIA.md)。

这是独立研究项目，非 RWKV 官方组件。目前交付设计与离线探针，不宣称已部署或取得模型成绩。仓库由所有者新建；可见性、许可证由所有者决定。已有本地 `main` 和 README，建空仓最方便。首次提交前查看 `git status --short --untracked-files=all` 和暂存清单；勿提交 `local/`、权重、索引、原稿、聊天、真实自评输出或凭据。此文只提供文案，不会创建远端、配置 remote 或推送。
