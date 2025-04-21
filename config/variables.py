"""Variables and Literal used for this repo."""

from typing import Literal, get_args

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
EntryMethod = Literal["multiple", "multiple_half", "single"]
ExitMethod = Literal["fifo", "lifo", "half_fifo", "half_lifo", "take_all"]
StopMethod = Literal[
    "no_stop",
    "percent_loss",
    "latest_loss",
    "nearest_loss",
]

# Mapping
SIGNAL_MAPPING = {
    "senti_entry": "SentiEntry",
    "senti_exit": "SentiExit",
    "senti_trades": "SentiTrades",
}

TRADES_MAPPING = {
    "senti_trades": "SentiTrades",
}

STRUCT_MAPPING = {
    "multiple": "MultiEntry",
    "multiple_half": "MultiHalfEntry",
    "single": "SingleEntry",
    "fifo": "FIFOExit",
    "lifo": "LIFOExit",
    "half_fifo": "HalfFIFOExit",
    "half_lifo": "HalfLIFOExit",
    "take_all": "TakeAllExit",
}

EXIT_PRICE_MAPPING = {
    method: "".join([meth.title() for meth in method.split("_")])
    for method in get_args(StopMethod)
}
