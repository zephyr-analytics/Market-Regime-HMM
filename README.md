# Market-HMM

## Configuration
-"tickers":

-"momentum_intervals":

-"volatility_interval":

-"bearish_cutoff": a float between 0.01 and 0.99.
-Note: This is the cut off probability of a bearish state transition, used to filter assets when building the portfolio.

-"start_date":

-"end_date":

-"max_retries": an int for max number of retries of training a model.
-Note: If training fails beyond nth retries for convergence and on jitter of states then an assessment of the underlying data is necessary.

-"moving_average": simple moving average lookback window (trading days)
-Note: this is used to further filter assets based on being below thier SMA.
-See: A Quantitative Approach to Tactical Asset Allocation, 2007 Faber, M

## Train
`python run.py --train`
-Training is utilized to inspect the model performance on identifing states, convergence, and labeling states.

## Infer
`python run.py --infer`
-Inferencing is utilized to inspect the model performance on predicted new states and provides checking for identifing new states, labeling predicted states, and propagating probabilities to future t.

## Build Portfolio
`python run.py --build`

## Test Portfolio
`python run.py --test`
NOTE: currently has failures on timeouts from data handling.

## Artifacts
-Artifacts are located within each sub directory of hmm directory.
-Examples: hmm/train/artifacts, hmm/infer/artifacts, hmm/build/artifacts
