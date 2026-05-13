# Final audited summary

本次任务的首要目标不是继续抬高指标，而是验证当前指标在无真值泄漏条件下是否仍然成立。

## 1. Current oracle baseline

- Oracle local improvement: 55.71%.
- Oracle positive local ratio: 100.0%.
- Oracle high-Hessian point-ratio increase: 0.261.
- Oracle point scan passed: 5 / 5.

Oracle 结果只作为理论上界，不作为主文实际可实现结果。

## 2. Practical no-leakage results

- Practical leakage audit: passed.
- Practical local improvement: 54.46%.
- Practical global improvement: 29.08%.
- Practical positive local ratio: 100.0%.
- Practical bootstrap 95% CI for local improvement: [48.72%, 60.34%].

## 3. Oracle vs practical gap

- Practical keeps 97.7% of oracle local improvement.
- Practical high-Hessian point-ratio increase is 0.224, oracle is 0.261.
- Practical point scan passed 5 / 5; oracle passed 5 / 5.

## 4. Whether core claim survives in practical mode

Yes. The practical chain retains positive local improvement, passes most point counts, and does not rely on full-field truth for Hessian weights or exchange decisions.

## 5. Strong baseline check under practical mode

- Exchange vs best traditional mean improvement: 44.54%.
- Exchange positive ratio vs best traditional: 100.0%.
- Exchange vs Hessian weighted mean improvement: 26.71%.
- Exchange positive ratio vs Hessian weighted: 90.0%.

## 6. Which conclusions are safe for main paper

- The practical reconstructed-Hessian chain can be used as the main paper result.
- Cubic PHS remains the residual reconstruction platform.
- The method improves local high-variation residual reconstruction under fixed sample count.
- Budget reallocation toward reconstructed high-Hessian areas is directly observed.
- Strong baseline checks under practical mode are completed.

## 7. Which conclusions must be downgraded to upper-bound / oracle-only statements

- Any result using full-field residual Hessian must be labelled oracle upper bound.
- Oracle improvement percentages must not be presented as practical measurement-chain performance.
- Oracle high-Hessian budget concentration should only be used to show the best-case value of perfect Hessian knowledge.

## 8. Remaining missing evidence

- Experimental measured surfaces are still needed beyond synthetic residuals.
- Practical Hessian robustness to measurement noise is not yet tested.
- The reconstructed Hessian could be sensitive to initial AFP sampling density; this should be discussed.
