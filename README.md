# Market-HMM

## Configuration
-"tickers" 

-"diversification_level" an int between 10 and 90.
-Note: Higher numbers lead to more diversifiaction.

-"bearish_cutoff" a float between 0.01 and 0.99.
-Note: This is the cut off probability of a bearish state transition, used to filter assets when building the portfolio.

-"start_date"

-"end_date"

-"max_retries" an int for max number of retries of training a model.
-Note: If training fails beyond nth retries for convergence and on jitter of states then an assessment of the underlying data is necessary.

## Train
`python run.py --train`

## Infer
`python run.py --infer`

## Build Portfolio
`python run.py --build`

## Artifacts
-Artifacts are located within each sub directory of hmm directory.
-Examples: hmm/train/artifacts, hmm/infer/artifacts, hmm/build/artifacts
