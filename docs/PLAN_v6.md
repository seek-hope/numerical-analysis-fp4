# 终极实验方案：递进比特分解 vs 传统 QAT vs 传统 PTQ

> 模型：Micro-Gemma ~164M（从零训练）
> GPU：8× RTX 4090
> 周期：4 周

---

## 一、模型规格

```
Micro-Gemma 164M:
  hidden_size: 768, intermediate: 3072, layers: 12
  GQA 4:1 (12 Q heads, 3 KV heads)
  8 sliding + 4 full attention (交替)
  per-layer embeddings (64-dim)
  vocab: 32000 (GPT-2 tokenizer)
  max_seq: 1024
```

---

## 二、三组实验

```
Group A: FP8 全量训练 → PTQ 多精度
  ┌─────────────────────────────────────────────┐
  │ 在 D1+D2+D3+D4 上 FP8 QAT 训练               │
  │ → PTQ 到 FP4 / 2-bit / 1-bit               │
  │ → 多精度 benchmark                          │
  └─────────────────────────────────────────────┘

Group B: 递进比特分解 + 数据质量递增  ← 核心贡献
  ┌─────────────────────────────────────────────┐
  │ Phase 0 (1-bit): D1 (raw C4, ~50B tokens)   │
  │ Phase 1 (2-bit): D2 (filtered, ~10B)        │
  │ Phase 2-3 (3-4b): D3 (wiki+books, ~2B)     │
  │ Phase 4-7 (5-8b): D4 (curated, ~200M)      │
  │ → 单一模型, 原生支持 1/2/4/8 bit 推理         │
  └─────────────────────────────────────────────┘

Group C: QAT FP8/4 + SR + Adaptive
  ┌─────────────────────────────────────────────┐
  │ 在 D1+D2+D3+D4 上分别训练:                   │
  │   C1: QAT FP8 (工业基线)                     │
  │   C2: QAT FP4 + SR (W2 最优)                │
  │ → 与 A/B 对比                                │
  └─────────────────────────────────────────────┘
```

---

## 三、数据需求

```
Tier  Dataset               Token 量    用途           来源
─────────────────────────────────────────────────────────
 D1   C4 (raw, unfiltered)   ~50B       low-bit 训练   HuggingFace
 D2   FineWeb (filtered)     ~10B       mid-low 训练   HuggingFace  
 D3   Wiki + BookCorpus      ~2B        mid-high 训练  HuggingFace
 D4   OpenOrca / SlimOrca    ~200M      high-bit 训练  HuggingFace
```

**数据获取**：远程服务器无法访问 HuggingFace → 需要在本地下载后 `rsync` 到远程（每组合计 ~60GB tokenized 数据，约 30GB 压缩后可在 1-2 小时内同步完成）。

---

## 四、时间估算

```
Group A (FP8 train):   ~5B tokens × 1 exp     ≈ 30-40h
Group B (progressive): 8 phases, 早期多token   ≈ 40-50h
Group C (QAT × 2):     ~5B tokens × 2 exps    ≈ 60-80h
─────────────────────────────────────────────────────────
总计                                                ~130-170h
8 GPU 并行, 每组独立 GPU                           ~5-7 天
```

## 五、四周甘特图

```
Week 1: 基础设施 + 数据准备
  Day 1-2: 本地下载 tokenizer + 数据, rsync 到远程
  Day 3-4: ScaledMicroGemma 模型实现（FP 版 + BitDecomp 版）
  Day 5-7: 启动 Group A (FP8 训练, 后台 30-40h)

Week 2: 递进比特训练
  Day 8-10: 完成 Group A, 评估并做 PTQ
  Day 8-14: 启动 Group B 的 8 个 phase（后台持续运行）

Week 3: QAT 对比 + 评估
  Day 15-18: Group C 训练（FP8 QAT + FP4 QAT+SR）
  Day 18-21: 三组模型统一 benchmark

Week 4: 分析 + 报告
  Day 22-24: 多精度推理对比, 精度-效率 Pareto
  Day 25-26: 比特平面信息分布分析
  Day 27-28: 最终报告撰写
```

---

## 六、成功指标

| 指标 | Group A | Group B | Group C |
|------|---------|---------|---------|
| 800M token 训练 PPL | < 30 | < 30 | < 30 |
| 1-bit 推理可用性 | PTQ 后崩 | ★ 原生可用 | N/A |
| 4-bit vs 8-bit PPL 退化 | PTQ: +100%+ | ★ 预期 +20% | QAT: +60% |
| 单一模型多精度 | ✗ | ★ | ✗ |

---

> 下一步：先确认数据获取方案可行，再开始实现。
