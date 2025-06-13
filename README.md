# Market-HMM
-Built with Python 3.10.1
-Install requirements with `pip install -r requirements.txt`
-Adjust the "data_file_path" of the config to match the repo.

## Configuration
The config files are stored within the configs directory of the repository. 
The config files are by portfolio construction.
`--global_macro`
`--global_stock`
`--us_macro`

-"tickers": (list of str), List of tickers that will represent the portfolio the model is working with.

-"momentum_intervals": (list of int), Intervals used to calculate momentum over a time window (trading days.)

-"volatility_interval": (int), Interval used to calculate volatility over a time window (trading days.)

-"use_equal_weight": (true or false), this flags whether the portfolio constructor should not alter the original weighting mechanics.

-"max_assets_per_cluster": (int), number of assets to select per cluster.

-"train_test_split": (float), percent of data to train with the remainder will be for testing.

-"bearish_cutoff": (float), between 0.01 and 0.99.
-Note: This is the cut off probability of a bearish state transition, used to filter assets when building the portfolio.

-"start_date": (str), Start date to begin the model
-Note: Model needs a warmup period before beginning to train models. See model_warmup

-"end_date": (str), End date for the model to finish.

-"model_warmup": (int), time period in years for the model to utilize for training dataset.

-"max_retries": (int), max number of retries of training a model.
-Note: If training fails beyond nth retries for convergence and on jitter of states then an assessment of the underlying data is necessary.

-"moving_average": (int), Simple moving average lookback window (trading days).
-Note: This is used to further filter assets based on being below thier SMA.
-See: A Quantitative Approach to Tactical Asset Allocation, 2007 Faber, M

-"data_file_path": (string), Path to the repo for storage of data file.

-"persist": (true or false), This should be set to false if running test. Set to true to inspect overall model performance.

## Config types
-NOTE: Rather than swap tickers around a second argument is used to flag which config file to use.
`python run.py --test --stock` or `python run.py --test --etf`

## Train
`python run.py --train --etf`
-Training is utilized to inspect the model performance on identifing states, convergence, and labeling states.

## Infer
`python run.py --infer --stock`
-Inferencing is utilized to inspect the model performance on predicted new states and provides checking for identifing new states, labeling predicted states, and propagating probabilities to future t.

## Build Portfolio
`python run.py --build --etf`

## Test Portfolio
`python run.py --test --stock`
NOTE: currently has failures on timeouts from data handling.

## Artifacts
-Artifacts are located within each sub directory of hmm directory.
-Examples: hmm/train/artifacts, hmm/infer/artifacts, hmm/build/artifacts

## ETF Portfolio:

The ETF portfolio focuses on a global macro approach to asset selection and allocation. Although this strategy focuses on absolute returns on a risk adjusted basis. The benchmark index used for the grading performance is the Bloomberg Macro Hedge Fund Index (BHMACR:IND). This portfolio does not currently use short position in any asset as a long only approach is easy for most retail investors to replicate the performance. Also, this ensures that model can be used across almost any account type where ETFs can be traded.
