# Algorithm 1: Acoustic differential-area fish counting

## Inputs

- Sonar image sequence `I`
- Single-fish calibration values `C[1..59]` in pixels per fish
- Number of annular zones `R = 59`
- Annular-zone width `d = 5 pixels`
- Selected frames per dataset `F = 15`

## Output

- Corrected cage-level fish estimate `N_est`

## Pseudocode

```text
1.  Select 15 frames at approximately uniform intervals from each scan dataset.
2.  Resize every selected frame to 800 x 600 pixels.
3.  Apply the cage-region mask and remove known net/truss background regions.
4.  Divide each image into 59 concentric rings, each 5 pixels wide.
5.  FOR each selected frame f = 1,...,15:
6.      FOR each ring r = 1,...,59:
7.          Estimate the Otsu threshold from pixels inside ring r.
8.          Count foreground pixels P[f,r] inside ring r.
9.          Convert area to a fish-equivalent feature:
                Q[f,r] = P[f,r] / C[r]
10.     END FOR
11.     Calculate the frame-level direct estimate:
                N_frame[f] = SUM(Q[f,r]), r = 1,...,59
12. END FOR
13. Sum the 15 selected water-layer totals:
                N_direct = SUM(N_frame[f]), f = 1,...,15
14. Input the scalar direct estimate N_direct to a BP network with one
    30-neuron hidden layer and one scalar fish-quantity output.
15. Fit the network using Bayesian regularisation.
16. Output the corrected cage-level estimate N_est.
17. Calculate absolute error, error rate, accuracy, MAE, RMSE, and the
    leave-one-dataset-out error distribution.
```

The approximately 4000 coordinate observations produced within a dataset are
nested spatial observations. They increase spatial information density but are
not treated as approximately 4000 independent experimental replicates.

The study reported leave-one-out cross-validation with the same fixed network
architecture. Repeated or nested cross-validation was not reported and remains
future work.
