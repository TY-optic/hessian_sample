# 多项式基底与残差精修评价报告

## 新逻辑定位

本实验不再把 XY 多项式与 PHS 作为同层级竞争模型。XY 多项式被定位为低阶基底，用于描述 BFS residual 中的全局低阶形貌；PHS 与 Hessian 引导采样优化只用于描述多项式拟合后的剩余局部残差。

这种定位避免了无限提高多项式阶数来追逐局部细节的问题，也更符合“低阶趋势 + 局部残差精修”的误差分解逻辑。

## 自动评价

- 多项式基底阶数：5。
- `AFP poly+residual PHS` 相对 `AFP poly only` 的 local RMSE 平均改善：50.88%。
- `AFP exchange poly+residual PHS` 相对 `AFP poly+residual PHS` 的 local RMSE 平均改善：42.61%。
- `AFP exchange poly+residual PHS` 相对 `AFP poly only` 的 local RMSE 平均改善：70.21%。
- 交换相对 AFP 残差精修的正改善比例：100.0%。
- Wilcoxon p(poly only > poly+residual)：9.537e-07。
- Wilcoxon p(AFP residual > exchange residual)：9.537e-07。

**结论：通过，建议将该两阶段框架作为论文主线表述。**

## 方法统计

| 方法 | 平均 global RMSE | 平均 local RMSE | local 中位数 | local 标准差 | 平均条件数 |
|---|---:|---:|---:|---:|---:|
| afp_exchange_poly_residual_phs | 0.000515163 | 0.000666021 | 0.000590423 | 0.000505414 | 200 |
| afp_poly_only | 0.0015465 | 0.00217901 | 0.0021412 | 0.00120224 | 179 |
| afp_poly_residual_phs | 0.000610908 | 0.0010608 | 0.0011404 | 0.00061967 | 179 |
| hessian_weighted_poly_residual_phs | 0.000538705 | 0.00082358 | 0.000737795 | 0.000642087 | 185 |
| regular_poly_only | 0.00242365 | 0.00423672 | 0.00409406 | 0.00317346 | 327 |
| regular_poly_residual_phs | 0.00168422 | 0.00335597 | 0.00262138 | 0.00312419 | 327 |

## 若结果不及预期的原因检查

- 结果达到预期：多项式作为低阶基底，PHS 残差精修与 Hessian 引导采样构成了更清晰的主线。

## 论文表述建议

建议把多项式写成基础低阶校正项，而不是高精度重建模型。可表述为：低阶 XY 多项式用于去除 BFS residual 中的全局缓变分量；剩余误差包含边界滚降、局部 bump 和中频起伏等局部细节，因此采用 cubic PHS 进行残差插值，并用 Hessian 强度引导固定点数下的测点重分配。
