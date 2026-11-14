# Small-sample model evidence (g-prior): development notes

Development only (dev split of the capability-acquisition benchmark, 400 tasks, seeds
1,000,000+); nothing here is a held-out result. `dev_compare.py` reproduces it.

S0 = student-t likelihood, 2 falsification rounds, flexible baseline, predictive pooling;
the only change is `evidence="gprior"` (Zellner g-prior Bayes factor, g = n, residuals
winsorized at 3 robust sd for the robust likelihood) instead of BIC with a plug-in scale.

| | BIC | g-prior |
|---|---|---|
| familiar score (n=251) | 0.8448 | 0.8621 (+0.0173, CI [+0.0043, +0.0299]) |
| familiar recovered | 80.9% | 91.6% |
| familiar answered "none" | 13.6% | 0% |
| gap score (n=149) | 0.6244 | 0.5962 (-0.0281, CI [-0.0595, +0.0026]) |
| gap confidently wrong | 16.1% | 19.5% |
| mean measurements | 16.4 | 22.8 |
| all tasks | 0.7627 | 0.7631 (+0.0004, CI [-0.0141, +0.0151]) |

Reading: the g-prior removes the small-sample flips toward the flexible description on
familiar laws, as intended. It also makes "none of my laws fits" harder to conclude when
it is true, and it reaches the stopping threshold later (more measurements). Net: no change.

Hypothesis for a pre-registered follow-up: g-prior evidence and acquired capabilities are
complementary (the g-prior fixes familiar tasks; capabilities give gap tasks a law to claim
instead of "none"), so S0+gprior+capabilities beats both S0+capabilities and S0+gprior.
