# 最终汇总报告

## 1. 主线结果是否成功复现

成功复现。主线保持不变：5阶 XY 多项式去趋势、residual cubic PHS 重建、residual Hessian 强度定义高变化区域、固定测点数下 Hessian 引导点交换优化。

- AFP exchange 相对 AFP poly+residual PHS 的 local RMSE 平均改善：33.41%。
- 正改善比例：100.0%。
- Wilcoxon p：9.537e-07。

## 2. 哪些缺失证据已补齐

已补齐六类直接数值证据：

- A. 预算重分配证据：采样预算是否向高 Hessian 区域移动。
- B. local 到 global 的误差传递证据。
- C. 点数扫描。
- D. 趋势项与残差项分层证据。
- E. 分类型稳定性。
- F. 统一统计汇总。

每个模块均输出了 `module_summary.md`、`machine_readable_summary.json`、CSV 表格和 PNG 图。

## 3. 趋势项/残差项分层是否得到直接支撑

得到支撑。

- 趋势项低频/高频功率比：1.2e+03。
- 去趋势残差低频/高频功率比：492。
- 5阶趋势能量占比均值：0.905。
- 5阶去趋势残差能量占比均值：0.095。

这说明 5阶多项式主要承担低频缓变趋势项，去趋势残差保留更多局部结构。该结果支持“多项式基底 + residual PHS 精修”的主线定位。

## 4. Hessian 是否确实驱动了预算重分配

得到支撑。

- AFP 在 top 20% Hessian 区域内的采样点占比均值：0.273。
- exchange 后占比均值：0.387。
- 占比平均增量：0.113。
- 高 Hessian 区域平均 fill distance 降低：0.0142539。
- 换入点相对换出点 Hessian 权重平均增量：0.357。

因此，exchange 不是黑箱式地改善误差，而是确实把有限测点预算重分配到了更高 residual Hessian 的区域。

## 5. local 改善如何传递到 global 改善

RMSE 平方分解得到验证：

`global RMSE^2 = theta * RMSE_high^2 + (1-theta) * RMSE_low^2`

- 分解平均绝对误差：3.062e-10。
- high-Hessian 区域 RMSE 平均改善：33.41%。
- low-Hessian 区域 RMSE 平均改善：-10.50%。
- global RMSE 平均改善：21.86%。
- global 未改善案例数：0。

解释是：当 high 区域误差下降足够大，且 low 区域没有明显恶化时，global RMSE 会通过面积加权平方误差同步下降。少数不改善案例通常来自 low-Hessian 区域误差恶化抵消了 high-Hessian 区域收益，或 high 区域改善幅度不足。

## 6. 点数扫描的主结论

- 通过点数数量：5 / 5。
- 所有点数下 local 改善均值：27.11%。
- exchange local RMSE 最低的点数：81。

结论：Hessian 引导点交换在多数固定测点数下保持正向收益。点数扫描应作为主线稳健性分析，而不是扩展成所有方法的全矩阵对比。

## 7. 不同面型下的稳定性结论

- 四类面型是否均达到正改善比例阈值：是。
- 面型平均 local 改善：33.41%。
- 最弱面型：edge_rolloff。
- 最强面型：mid_frequency_undulation。

当前结果支持方法在四类面型上具有基本稳定性，但正式论文中仍建议按面型分别报告，避免只给总体平均值。

## 8. 可以安全写入主文的结论

可以写入主文的结论包括：

- 5阶 XY 多项式适合作为低阶趋势项，而非无限提高阶数的高精度局部重建模型。
- 多项式去趋势后的 residual 保留局部结构，适合用 cubic PHS 进行精修。
- residual Hessian 高变化区域对应更高局部重建风险，因此 local RMSE 和 high-Hessian 区域指标具有物理和数值意义。
- Hessian 引导点交换确实使采样预算向高 Hessian 区域重分配，并降低该区域 fill distance。
- local RMSE 的下降可以通过面积加权平方误差传递到 global RMSE。
- 在固定测点数扫描和四类面型分组中，该方法保持较稳定的正向收益。

## 9. 仍需谨慎表述的部分

需要谨慎表述：

- 不应声称 Hessian 是逐点误差的唯一决定因素，它是采样预算分配的有效引导量。
- 不应把 NN 模块重新抬升为核心；本项目未使用 NN 作为主线证据。
- 不应把 Gaussian RBF 重新设为主平台；本项目继续使用 cubic PHS。
- 点数扫描当前用于稳健性，不应扩展成所有方法、所有模型、所有点数的全矩阵对比。
- 当前结果基于合成自由曲面 residual，正式论文仍需与实验数据或更接近实际检测误差的仿真模型相互印证。
