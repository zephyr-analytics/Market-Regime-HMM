# Market-HMM
-Built with Python 3.10.1
-Install requirements with `pip install -r requirements.txt`
-Adjust the "data_file_path" of the config to match the repo.

## Configuration
The config files are stored within the configs directory of the repository. 
The config files are by portfolio construction.
`--global_macro`
`--global_stock`

-"tickers": (list of str), List of tickers that will represent the portfolio the model is working with.

-"momentum_intervals": (list of int), Intervals used to calculate momentum over a time window (trading days.)

-"volatility_interval": (int), Interval used to calculate volatility over a time window (trading days.)

-"min_clusters": (int, tunable parameter), default number should be based on category of assets in portfolio.
-Note: Asset categories such as equities, bonds, real assets, and cash.

-"max_clusters": (int, tunable parameter), default set to 15.

-"max_assets_per_cluster": (int, tunable parameter), default set to 3.

-"train_test_split": (float), percent of data to train with the remainder will be for testing.

-"start_date": (str), Start date to begin the model
-Note: Model needs a warmup period before beginning to train models. See model_warmup

-"end_date": (str), End date for the model to finish.

-"model_warmup": (int), time period in years for the model to utilize for training dataset.

-"max_train_years": (int, tunable parameter), max time before model begins data dropout.

-"max_retries": (int), max number of retries of training a model.
-Note: If training fails beyond nth retries for convergence and on jitter of states then an assessment of the underlying data is necessary.

-"moving_average": (int, tunable parameter), Simple moving average lookback window (trading days).
-Note: This is used to further filter assets based on being below thier SMA.
-See: A Quantitative Approach to Tactical Asset Allocation, 2007 Faber, M

-"stop_loss": (negative float, tunable parameter), threshold single day loss to exit an assest position.

-"data_file_path": (string), Path to the repo for storage of data file.

-"persist": (true or false), This should be set to false if running test. Set to true to inspect and calibrate overall model performance.

-"grid": (dict), Dictionary of tunable parameter lists.

## Config types
-NOTE: Rather than swap tickers around a second argument is used to flag which config file to use.
`python run.py --test --global_macro` or `python run.py --test --global_stock`

## Train
`python run.py --train --global_macro`
-Training is utilized to inspect the model performance on identifing states, convergence, and labeling states.

## Infer
`python run.py --infer --global_macro`
-Inferencing is utilized to inspect the model performance on predicted new states and provides checking for identifing new states, labeling predicted states, and propagating probabilities to future t.

## Build Portfolio
`python run.py --build --global_macro`

## Test Portfolio
`python run.py --test --global_macro`

## Tune Portfolio
`python run.py --tune --global_macro`

## Artifacts
-Artifacts are located within each sub directory of hmm directory.
-Examples: hmm/train/artifacts, hmm/infer/artifacts, hmm/build/artifacts

## Asset Analysis
All trade information is presisted for review after running a test pipeline.
Plots of asset portfolio contribution are persisted for review by the user. These should only be utilized for adjusting asset exposure or asset swapping due to better performance.
