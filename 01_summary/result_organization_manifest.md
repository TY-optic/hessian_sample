# 结果归拢清单

## 当前分组

- `main_phase_results/`：原主流程结果，包括第一阶段 A-F、第二阶段学习验证、主流程配置、日志、模型与报告。
- `traditional_sampling_layer/`：新增传统采样方式层级验证，包含独立代码、输出与报告。
- `polynomial_only_layer/`：新增仅多项式条件层级验证，包含独立代码、输出与报告。
- `poly_residual_focus/`：修正后的“多项式低阶基底 + PHS 残差精修 + Hessian 采样优化”主线验证，包含独立代码、输出与报告。
- `cross_layer_validation_results/`：新增层级验证的跨层级汇总报告。

## 根目录保留内容

- `src/`：公共代码与总控脚本。
- `configs/`、`outputs/`、`logs/`、`models/`、`reports/`：保留为空目录或后续运行入口，避免未来脚本因目录不存在而失败。

## 边界说明

本次归拢仅移动 `sampling_optimization_study/` 内部文件，没有写入或修改该目录之外的文件。
