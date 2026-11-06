# Exploratory analyses (fixed before the final evaluation)

Written during the acquisition phase of replication 0, after the first review and before
any report-only split (test, test_familiar, test_regime, test_compose, sealed, external)
had been evaluated. These analyses are **exploratory**: they cannot change any
pre-registered verdict, and they are reported separately from the confirmatory results.

## What prompted it

The first reviewed candidate (cap-0004, `c + c·exp(a·x)·sin(b·x + c)`) passed every gate
except one: held-out gain +0.030 (CI [+0.020, +0.040]), transfer gain +0.031 (CI [+0.014,
+0.048]), but familiar regression −0.0042 with CI [−0.0125, +0.0044], whose lower bound is
below the pre-registered −0.01 margin. The familiar-task interval is wider than the power
analysis assumed (per-task sd about 0.115 rather than 0.06–0.10). It was rejected, as the
protocol requires.

## Analyses

* **X1. Rejected-but-promising candidates.** For every candidate whose reviewer held-out
  and transfer gates both passed but which was rejected by another gate, evaluate
  S0 + that candidate (alone) and S0 + all such candidates together on every report-only
  split, paired against C0, with the same statistics as the confirmatory analysis.
* **X2. Gate power.** Report the realized per-task sd of the familiar-regression
  comparison in every review, and the sample size the -0.01 margin would have needed for
  80% power at the observed effect.
* **X3. Where the familiar cost comes from.** For X1's configurations, decompose the
  familiar-task score difference into claim, interpolation, extrapolation and measurement
  cost components, and count answer changes.

Nothing in X1–X3 feeds back into the acquisition loop, the registry or the reviewer.
