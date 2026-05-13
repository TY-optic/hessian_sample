# 自由曲面 BFS 残差采样优化：完整结果报告

## 1. 总体结论

本轮数值验证支持将论文主线调整为：

> 先用低阶 XY 多项式拟合 BFS residual 中的全局缓变分量，再对多项式拟合后的剩余局部残差采用 cubic PHS 重建，并利用残差 Hessian 强度引导固定测点数下的采样点重分配。

该表述比“多项式模型与 PHS 模型直接竞争”更合理。多项式不再被要求无限提高阶数来描述局部细节，而是作为低阶基底；Hessian 引导采样优化的贡献集中在局部残差的高精度描述。

当前最有论文写作价值的结果是：

- `AFP exchange + PHS` 相对 `AFP + PHS` 在主验证中 local RMSE 平均下降 31.70%，global RMSE 平均下降 10.00%。
- 在传统采样层级中，`AFP exchange + PHS` 相对逐实例最佳传统采样 local RMSE 仍平均下降 34.01%。
- 在修正后的两阶段框架中，`AFP exchange poly+residual PHS` 相对 `AFP poly only` 的 local RMSE 平均下降 70.21%。
- Hessian 高变化区域的 local/global RMSE 平均比值为 1.729，说明局部二阶变化区域确实对应更高重建误差。

第二阶段轻量学习模型目前不宜作为主文核心结论。NN 初始化本身相对 AFP 的 local RMSE 平均改善为 -1.16%，但 NN 初始化后少量交换可达到 24.89% 改善，可作为补充探索。

---

## 2. 实验层级与各自回答的问题

| 层级 | 目录 | 核心问题 | 结论定位 |
|---|---|---|---|
| 主流程验证 | `main_phase_results/` | Hessian 引导点交换是否显著优于 AFP + PHS | 支撑主结论 |
| 传统采样层级 | `traditional_sampling_layer/` | 新方法是否只是优于单一 AFP 基线 | 支撑强基线公平性 |
| 仅多项式层级 | `polynomial_only_layer/` | 多项式平台是否适合作为采样优化主评价平台 | 诊断性结果，不作为主结论 |
| 多项式基底 + 残差精修 | `poly_residual_focus/` | 多项式作为低阶基底后，残差 PHS 与 Hessian 采样是否有效 | 建议作为论文主线 |
| 学习型方法 | `main_phase_results/outputs/generalization_tests/` | NN 是否能学习泛化采样规律 | 补充探索 |

---

## 3. 数据与统一设置

当前数值验证采用统一随机自由曲面 BFS residual 生成器，包含四类面型：

- 平滑低变化面型；
- 边界滚降面型；
- 局部 bump 面型；
- 中频起伏面型。

主流程每类生成 5 个随机实例，共 20 个实例。默认参数为：

- 网格尺寸：56；
- 固定采样点数：64；
- 候选点池：720；
- 高变化区域：Hessian 强度上 20% 区域；
- 主重建平台：cubic PHS；
- 基础多项式阶数：5。

这些参数适合当前交互式验证。正式论文复算建议增加随机实例数，并加入测点数扫描。

---

## 4. Hessian 误差关系验证

目的：确认高二阶变化区域是否确实对应较高重建误差。

结果：

- 平均 local/global RMSE 比值：1.729；
- Hessian 与绝对误差 Spearman 相关均值：0.316；
- 判定：高二阶变化区域整体对应更高误差。

解释：

该结果支持用 Hessian 强度定义局部高变化区域，并把 local RMSE 作为主要评价指标。Spearman 相关并非很高，说明 Hessian 不是逐点误差的唯一决定因素；但 local/global 比值显著大于 1，足以支持“高二阶变化区域是重建误差重点区域”的方法论假设。

---

## 5. 主流程：AFP 与 Hessian 点交换对比

目的：判断在 cubic PHS 平台下，Hessian 引导点交换是否相对 AFP 形成论文级显著区分。

### 关键统计

- `AFP exchange + PHS` 相对 `AFP + PHS` 的平均 local RMSE 改善：31.70%；
- 平均 global RMSE 改善：10.00%；
- local RMSE 正改善实例比例：100.0%；
- global RMSE 正改善实例比例：75.0%；
- Wilcoxon p(local)：9.537e-07；
- Wilcoxon p(global)：0.001576。

### 方法统计

| 方法 | 平均 global RMSE | 平均 local RMSE | local 中位数 | local 标准差 |
|---|---:|---:|---:|---:|
| AFP + PHS | 0.000568093 | 0.000974815 | 0.00057836 | 0.000840684 |
| Hessian weighted + PHS | 0.000595075 | 0.000832114 | 0.000520102 | 0.000747732 |
| AFP exchange + PHS | 0.000436614 | 0.000614646 | 0.000400442 | 0.000554716 |
| Estimated Hessian exchange + PHS | 0.000454822 | 0.000715652 | 0.000420985 | 0.000656618 |

### 判断

主流程达到预设论文级显著区分标准。结果说明，固定测点数下，基于 Hessian 的点交换优化能够稳定降低局部高变化区域误差，并同步降低全局误差。

---

## 6. 传统采样方式层级

目的：排除“新方法只是优于单一 AFP 基线”的质疑。

纳入方法：

- regular grid；
- random uniform；
- jittered grid；
- Latin hypercube；
- Poisson disk；
- AFP baseline；
- Hessian weighted；
- AFP exchange。

统一采用 cubic PHS 重建。

### 关键统计

- `AFP exchange` 相对 `AFP baseline` 的 local RMSE 平均改善：40.37%；
- 相对逐实例最佳传统采样的 local RMSE 平均改善：34.01%；
- 相对 AFP baseline 的正改善比例：100.0%；
- Wilcoxon p(exchange < AFP)：9.537e-07；
- Wilcoxon p(exchange < best traditional)：9.537e-07。

### 方法统计

| 方法 | 平均 global RMSE | 平均 local RMSE | local 中位数 | local 标准差 |
|---|---:|---:|---:|---:|
| AFP baseline | 0.000707472 | 0.00123668 | 0.00104646 | 0.00103 |
| AFP exchange | 0.000554248 | 0.000763203 | 0.000467499 | 0.000728041 |
| Hessian weighted | 0.000737883 | 0.00108612 | 0.000915207 | 0.000945306 |
| jittered grid | 0.00108033 | 0.00198255 | 0.00134132 | 0.00178172 |
| Latin hypercube | 0.00103668 | 0.00184231 | 0.00160814 | 0.00124163 |
| Poisson disk | 0.000697301 | 0.00126448 | 0.000889042 | 0.00111555 |
| random uniform | 0.00110248 | 0.00193286 | 0.00172132 | 0.00132743 |
| regular grid | 0.00107621 | 0.00190758 | 0.00122413 | 0.00173468 |

### 判断

该层级通过。结果说明，Hessian 引导点交换的优势不是建立在 AFP 弱基线之上；即使与传统采样中的较优结果相比，仍有明显 local RMSE 改善。

---

## 7. PHS 与 Gaussian RBF 对比

目的：确认 cubic PHS 是否适合作为采样设计评价平台。

结果：

- cubic PHS 失败数：0；
- Gaussian RBF 失败数：0；
- cubic PHS 平均 global RMSE：0.000573958；
- Gaussian RBF 平均 global RMSE：0.00269264。

判断：

cubic PHS 明显优于 Gaussian RBF，并且在当前实例中更稳定。后续采样优化采用 cubic PHS 作为主评价平台是合理的。

---

## 8. 仅多项式条件层级

目的：诊断 XY 多项式是否适合作为局部采样优化的主评价平台。

### 自动评价

- 全局最佳平均阶数：5；
- 阶数效应/布局效应均值比：1.08；
- 最佳阶数下 `AFP exchange` 相对 `AFP` 的 local RMSE 平均改善：4.46%；
- 最佳阶数下正改善比例：80.0%；
- Wilcoxon p(exchange < AFP)：0.0005083。

### 代表性结果

| 方法 | 阶数 | 平均 global RMSE | 平均 local RMSE | 平均条件数 |
|---|---:|---:|---:|---:|
| AFP baseline | 3 | 0.00230816 | 0.00319592 | 18.1 |
| AFP baseline | 5 | 0.0015803 | 0.00222424 | 180 |
| AFP baseline | 7 | 0.00101435 | 0.00165159 | 2.23e+03 |
| AFP baseline | 9 | 0.00280192 | 0.00534279 | 5.52e+04 |
| AFP baseline | 11 | 0.00833037 | 0.0148517 | 1.88e+05 |
| AFP exchange | 5 | 0.00161369 | 0.00214392 | 185 |
| Hessian weighted | 5 | 0.00164627 | 0.00217065 | 211 |
| regular grid | 5 | 0.002395 | 0.00408475 | 327 |

### 判断

仅多项式层级不应作为主结论。其原因不是简单的“采样无效”，而是：

- 多项式阶数对误差有显著影响；
- 高阶多项式出现明显条件数病态；
- 多项式平台下采样优化收益较弱，且与阶数和数值稳定性耦合；
- 全局多项式不适合承担局部残差细节的高精度描述。

因此，多项式应作为低阶基底，而不是作为局部高精度重建模型。

---

## 9. 修正主线：多项式基底 + PHS 残差精修 + Hessian 采样优化

目的：将多项式定位为低阶基底，用 PHS 和 Hessian 采样优化专注描述多项式拟合后的剩余局部残差。

### 方法框架

1. 用 5 阶 XY 多项式拟合 BFS residual 中的全局低阶趋势；
2. 计算多项式拟合后的 residual；
3. 用 cubic PHS 对 residual 进行精修；
4. 用 residual Hessian 强度引导候选点权重；
5. 通过点交换优化固定点数采样布局。

### 关键统计

- `AFP poly+residual PHS` 相对 `AFP poly only` 的 local RMSE 平均改善：50.88%；
- `AFP exchange poly+residual PHS` 相对 `AFP poly+residual PHS` 的 local RMSE 平均改善：42.61%；
- `AFP exchange poly+residual PHS` 相对 `AFP poly only` 的 local RMSE 平均改善：70.21%；
- 交换相对 AFP 残差精修的正改善比例：100.0%；
- Wilcoxon p(poly only > poly+residual)：9.537e-07；
- Wilcoxon p(AFP residual > exchange residual)：9.537e-07。

### 方法统计

| 方法 | 平均 global RMSE | 平均 local RMSE | local 中位数 | local 标准差 | 平均条件数 |
|---|---:|---:|---:|---:|---:|
| AFP exchange poly+residual PHS | 0.000515163 | 0.000666021 | 0.000590423 | 0.000505414 | 200 |
| AFP poly only | 0.0015465 | 0.00217901 | 0.0021412 | 0.00120224 | 179 |
| AFP poly+residual PHS | 0.000610908 | 0.0010608 | 0.0011404 | 0.00061967 | 179 |
| Hessian weighted poly+residual PHS | 0.000538705 | 0.00082358 | 0.000737795 | 0.000642087 | 185 |
| regular poly only | 0.00242365 | 0.00423672 | 0.00409406 | 0.00317346 | 327 |
| regular poly+residual PHS | 0.00168422 | 0.00335597 | 0.00262138 | 0.00312419 | 327 |

### 判断

该层级通过，且是当前最清晰的论文主线。它避免了把多项式和 PHS 放在同一层级竞争，也避免了审稿人要求不断提高多项式阶数来追逐局部细节。多项式负责低阶趋势，PHS 和 Hessian 采样负责局部残差，这一分工更符合物理和数值逻辑。

---

## 10. 学习型方法泛化验证

目的：评估神经网络是否可以学习采样分布规律。

结果：

- 训练样本数：20；
- 轻量 MLP 训练 MSE：8.33087e-05；
- NN 初始化相对 AFP 的平均 local RMSE 改善：-1.16%；
- NN 初始化加少量交换相对 AFP 的平均 local RMSE 改善：24.89%；
- 完整交换相对 AFP 的平均 local RMSE 改善：41.37%。

判断：

学习型方法目前不宜写入主文核心结论。它可以作为补充探索：NN 初始化本身不稳定，但与少量点交换结合后有一定收益。若后续扩大数据集、采用更稳定的密度图标签和约束采样策略，可作为扩展工作。

---

## 11. 建议的论文实验组织

建议主文按以下逻辑组织：

### 11.1 模型分解

首先说明 BFS residual 可分为低阶缓变分量和局部剩余残差。低阶 XY 多项式用于去除全局趋势，不承担高精度局部细节重建任务。

### 11.2 评价平台选择

通过多项式阶数和 PHS/Gaussian 对比说明：

- 低阶多项式表达能力有限；
- 高阶多项式存在条件数病态；
- Gaussian RBF 在当前设置下误差较大；
- cubic PHS 更适合作为残差精修和采样评价平台。

### 11.3 核心方法验证

以 `poly baseline + residual PHS` 为基础，比较：

- `AFP poly only`；
- `AFP poly+residual PHS`；
- `Hessian weighted poly+residual PHS`；
- `AFP exchange poly+residual PHS`。

核心 claim 应聚焦：

> Hessian 引导采样优化能够在固定测点数下提升多项式剩余残差的局部重建精度。

### 11.4 强基线对比

加入传统采样方式对比，证明方法不是只优于 AFP。

### 11.5 局部误差机制解释

用 Hessian-error relation 说明，高 Hessian 区域是误差集中的主要区域，因此 local RMSE 是合理指标。

---

## 12. 不建议作为主结论的内容

以下内容建议降级为补充或方法选择说明：

- 仅多项式条件下的新方法优势；
- NN 学习型初始化；
- 高阶 XY 多项式拟合局部细节；
- Gaussian RBF 作为主重建平台。

这些内容不是无价值，而是不适合作为主结论支点。

---

## 13. 后续最值得补充的验证

建议优先补充点数扫描，但只放在主线框架下：

- 点数：36、49、64、81、100；
- 方法：`AFP poly+residual PHS` 与 `AFP exchange poly+residual PHS`；
- 指标：global RMSE、local RMSE、local 改善比例、正改善实例比例；
- 目的：验证 Hessian 引导残差采样优化是否在不同固定测点数下稳定有效。

不建议对所有层级全面做点数扫描，否则会形成过大的实验矩阵并模糊主线。

---

## 14. 最终论文级判断

基于当前结果，最稳妥的论文主结论是：

> 在固定测点数条件下，低阶多项式可有效描述 BFS residual 的全局缓变趋势；对多项式拟合后的剩余局部残差，cubic PHS 提供了更适合的高精度重建平台。进一步利用残差 Hessian 强度引导点交换优化，可以稳定降低局部高变化区域的重建误差，并在全局误差上保持同步改善。

该结论有当前数值结果支撑，且相较于“采样优化直接优于多项式拟合”的表述更严谨、更不容易被审稿人质疑。

