# 第一阶段论文级显著区分判断

**结论：达到论文级显著区分，可进入第二阶段学习验证。**

## 关键统计

- AFP exchange 相对 AFP + PHS 的平均 local RMSE 改善：31.70%。
- AFP exchange 相对 AFP + PHS 的平均 global RMSE 改善：10.00%。
- local RMSE 正改善实例比例：100.0%。
- global RMSE 正改善实例比例：75.0%。
- Wilcoxon p(local)：9.537e-07。
- Wilcoxon p(global)：0.001576。

## 方法统计

| 方法 | 平均 global RMSE | 平均 local RMSE | local 中位数 | local 标准差 |
|---|---:|---:|---:|---:|
| afp_phs | 0.000568093 | 0.000974815 | 0.00057836 | 0.000840684 |
| hessian_weighted_phs | 0.000595075 | 0.000832114 | 0.000520102 | 0.000747732 |
| afp_exchange_phs | 0.000436614 | 0.000614646 | 0.000400442 | 0.000554716 |
| estimated_hessian_exchange_phs | 0.000454822 | 0.000715652 | 0.000420985 | 0.000656618 |

## 阈值自检

- mean_local_improvement_ge_15pct: 是
- mean_global_improvement_ge_5pct: 是
- local_positive_ratio_ge_70pct: 是
- global_positive_ratio_ge_60pct: 是
- wilcoxon_significant: 是

## 论文可用性判断

当前方法相对 AFP 基线形成了足够稳定的 local 与 global 改善。
