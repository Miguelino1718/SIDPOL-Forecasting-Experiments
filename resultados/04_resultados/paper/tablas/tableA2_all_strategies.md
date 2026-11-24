| Target | Strategy (A) | Status | Median MASE | Median MAE | A − Naïve-1 | A − SES |
|---|---|---|---|---|---|---|
| Robbery | Naïve-1 | Reference | 0.528 | 10.6 | — | +0.015 [0, +0.036] (14) |
| Robbery | Seasonal naïve | Reference | 1.269 | 25.4 | +0.737 [+0.623, +0.851] (1) | +0.726 [+0.695, +0.855] (0) |
| Robbery | SES | Post hoc | 0.536 | 9.6 | -0.015 [-0.036, 0] (26) | — |
| Robbery | SES + ½ OLS slope (ad hoc)† | Post hoc | 0.536 | 9.6 | -0.013 [-0.036, +0.002] (26) | +0.002 [+0.001, +0.002] (8) |
| Robbery | Local XGB, levels | Primary | 0.642 | 10.7 | +0.088 [+0.076, +0.137] (9) | +0.122 [+0.084, +0.152] (4) |
| Robbery | Global XGB, levels (H_L) | Primary | 0.584 | 9.8 | +0.024 [-0.008, +0.047] (17) | +0.027 [+0.014, +0.063] (13) |
| Robbery | Global XGB, levels (H_G) | Sensitivity | 0.579 | 10.1 | +0.031 [+0.011, +0.059] (14) | +0.045 [+0.025, +0.058] (8) |
| Robbery | Local XGB, Δz | Exploratory | 0.519 | 10.2 | -0.002 [-0.010, +0.005] (23) | +0.009 [-0.004, +0.024] (18) |
| Robbery | Global XGB, Δz (H_L5) | Exploratory | 0.535 | 10.1 | -0.013 [-0.024, -0.003] (30) | +0.003 [-0.006, +0.017] (19) |
| Robbery | Global XGB, Δz (H_G5) | Exploratory | 0.599 | 11.7 | +0.047 [+0.024, +0.070] (11) | +0.049 [+0.038, +0.067] (2) |
| Robbery | Ridge (global), Δz | Exploratory | 0.571 | 10.1 | +0.035 [-0.007, +0.056] (19) | +0.039 [+0.027, +0.058] (10) |
| Theft | Naïve-1 | Reference | 0.568 | 18.8 | — | +0.026 [+0.014, +0.047] (10) |
| Theft | Seasonal naïve | Reference | 1.164 | 41.0 | +0.566 [+0.459, +0.653] (2) | +0.574 [+0.524, +0.644] (1) |
| Theft | SES | Post hoc | 0.566 | 18.4 | -0.026 [-0.047, -0.014] (31) | — |
| Theft | SES + ½ OLS slope (ad hoc)† | Post hoc | 0.566 | 18.6 | -0.024 [-0.045, -0.014] (31) | +0.001 [+0.0005, +0.001] (6) |
| Theft | Local XGB, levels | Primary | 0.693 | 21.3 | +0.087 [+0.050, +0.145] (9) | +0.109 [+0.084, +0.145] (3) |
| Theft | Global XGB, levels (H_L) | Primary | 0.632 | 20.1 | +0.020 [-0.015, +0.050] (19) | +0.036 [+0.016, +0.052] (10) |
| Theft | Global XGB, levels (H_G) | Sensitivity | 0.606 | 20.3 | +0.019 [-0.012, +0.040] (16) | +0.036 [+0.028, +0.051] (6) |
| Theft | Local XGB, Δz | Exploratory | 0.568 | 19.3 | -0.003 [-0.022, +0.003] (23) | +0.011 [+0.006, +0.032] (12) |
| Theft | Global XGB, Δz (H_L5) | Exploratory | 0.574 | 19.5 | -0.025 [-0.030, -0.018] (36) | +0.007 [-0.003, +0.011] (20) |
| Theft | Global XGB, Δz (H_G5) | Exploratory | 0.566 | 19.2 | -0.030 [-0.036, -0.011] (35) | +0.005 [-0.0005, +0.014] (17) |
| Theft | Ridge (global), Δz | Exploratory | 0.590 | 19.9 | -0.008 [-0.024, +0.008] (25) | +0.026 [+0.004, +0.042] (14) |

*The A − B columns report the median of 43 paired district differences MASE(A) − MASE(B) [95% bootstrap CI over districts] (districts where A has lower MASE). Negative favours A. Absolute paired differences below 1e-12 are treated as numerical ties. A displayed "0" denotes zero after this tolerance, not practical equivalence.*

*† Ad hoc reference kept from the internal registry: SES level plus half the OLS slope of log1p(y); not the standard Theta method. Differences are reported in the A − SES column; no practical equivalence margin was defined.*
