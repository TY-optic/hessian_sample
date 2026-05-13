# 多项式基底重心调整后的最终结论

## 是否可行

可行。更合理的逻辑不是让 XY 多项式与 PHS 直接竞争高精度重建能力，而是将多项式作为低阶基底，用于描述 BFS residual 中的全局缓变分量；PHS 与 Hessian 引导采样优化则负责多项式拟合后的局部剩余残差。

## 当前数值结论

- `AFP poly+residual PHS` 相对 `AFP poly only` 的 local RMSE 平均改善为 50.88%。
- `AFP exchange poly+residual PHS` 相对 `AFP poly+residual PHS` 的 local RMSE 平均改善为 42.61%。
- `AFP exchange poly+residual PHS` 相对 `AFP poly only` 的 local RMSE 平均改善为 70.21%。
- 交换优化相对 AFP 残差精修的正改善比例为 100.0%。
- 两个关键 Wilcoxon 配对检验的 p 值均为 9.537e-07。

## 建议论文主线

建议将方法表述为两阶段框架：

1. 低阶 XY 多项式拟合全局趋势项。
2. 对多项式拟合后的剩余残差进行 cubic PHS 重建。
3. 用残差 Hessian 强度引导固定测点数下的点重分配。

该表述避免了“不断提高多项式阶数来追逐局部细节”的不合理比较，也能把方法贡献集中到局部残差高精度描述与采样优化上。
