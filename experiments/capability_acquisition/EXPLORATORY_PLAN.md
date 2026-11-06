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

## Amendment (2026-11-06, during the acquisition phases, before any report-only split was evaluated)

Development analysis on the dev split (no held-out data) found that (a) the familiar-task
cost of a capability is posterior dilution in the base solver (tasks whose answer does not
change take ~0.45 more measurements and lose credence), and (b) with small-sample model
evidence (`evidence="gprior"`, developed on a separate branch), S0 plus the three acquired
structures scored +0.007 on familiar tasks and +0.07 overall on 400 dev tasks.

* **X4. Corrected base solver.** After both replications' registered phases have finished
  and the development branch is merged, evaluate S0' = S0 + `evidence="gprior"` alone and
  S0' + all X1 candidates on every report-only split, paired against C0 and against S0 + all
  X1 candidates. Exploratory: the confirmatory test of this idea is a separate, newly
  pre-registered experiment on fresh seeds.
