"""Variables and Literal used for this repo."""

from typing import Literal

# Analysis
HfModel = Literal["prosusai", "yiyanghkust", "ziweichen", "aventiq_ai"]
CorrFn = Literal["pearsonr", "spearmanr", "kendalltau"]
CointFn = Literal["coint"]
CointCorrFn = Literal[CointFn, CorrFn]
Component = Literal["word", "punct", "special"]
StratComponent = Literal[
    "entry_type",
    "entry_struct",
    "exit_struct",
    "stop_method",
    "hf_model",
    "coint_corr_fn",
    "period",
]

# Trades
PriceAction = Literal["buy", "sell", "wait"]
EntryType = Literal["long", "short", "longshort"]
EntryMethod = Literal["MultiEntry", "MultiHalfEntry", "SingleEntry"]
ExitMethod = Literal[
    "FIFOExit", "LIFOExit", "HalfFIFOExit", "HalfLIFOExit", "TakeAllExit"
]
StopMethod = Literal["no_stop", "PercentLoss", "LatestLoss", "NearestLoss"]
