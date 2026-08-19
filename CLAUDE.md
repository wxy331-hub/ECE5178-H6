# ECE5178 Sphero Project Instructions

Project-specific rules only. The workspace `CLAUDE.md` one level up carries the
general working method, verification, research-integrity, Git and MCP graph
rules, and they apply here unchanged.

Monash ECE5178 Intelligent Robotics. Student ID **33377006** — it is the CSV
filename prefix for every automarked submission.

## Submission contracts

Strict — the automarker rejects any deviation.

| | Lab 1 | Lab 2 |
| --- | --- | --- |
| File | `labs/lab1/33377006_lab1.csv` | `labs/lab2/33377006_lab2.csv` |
| Data rows | 100 | 200 |
| Columns | `sim_x,sim_y,real_x,real_y` | `sim_x,sim_y,real_x,real_y,P_xx,P_xy,P_yy` |
| Thresholds | final distance ≤ 0.10 m, RMSE ≤ 0.20 m | mean Mahalanobis ≤ 4.0, chi-square pass rate ≥ 0.90 |

`analyze_lab1.py` and `analyze_lab2.py` hold these limits as constants; import
them rather than restating the numbers.

Lab 1 has been submitted. Its CSV records one specific hardware run, so
changing `lab1.py` behaviour would fork the code from what produced the marked
result — comment and documentation edits are fine, logic changes are not
without saying what it costs. Lab 2 has not been submitted yet.
