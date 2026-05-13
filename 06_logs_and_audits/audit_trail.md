# Audit trail

本次任务的首要目标不是继续抬高指标，而是验证当前指标在无真值泄漏条件下是否仍然成立。

- Practical mode: Hessian weights, high-Hessian masks, and exchange decisions are computed from the reconstructed residual field obtained from current sampled measurements only.
- Oracle mode: full-field truth residual may be used only to form an upper-bound Hessian reference.
- Truth usage allowed: final RMSE evaluation, diagnostic decomposition, plotting/statistical comparison.
- Truth usage forbidden: practical Hessian weights, practical high-Hessian mask, practical exchange decision.
- Leakage audit: implemented by TruthLeakageGuard plus static source audit.
