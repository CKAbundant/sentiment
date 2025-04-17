"""Variables and Literal used for this repo."""

from typing import Literal, get_args

# Download OHLCV
URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
IGNORE_LIST = ["BRK.B", "BF.B", "CTAS", "LEN"]

# Analysis
HF_MODEL = Literal["prosusai", "yiyanghkust", "ziweichen", "aventiq_ai"]
CORR_FN = Literal["pearsonr", "spearmanr", "kendalltau"]
COINT_FN = Literal["coint"]
COINT_CORR_FN = Literal[COINT_FN, CORR_FN]
Component = Literal["word", "punct", "special"]

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
