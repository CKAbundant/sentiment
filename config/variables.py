from typing import Literal

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
IGNORE_LIST = ["BRK.B", "BF.B", "CTAS", "LEN"]

# Literal
HF_MODEL = Literal["prosusai", "yiyanghkust", "ziweichen", "aventiq_ai"]
CORR_FN = Literal["pearsonr", "spearmanr", "kendalltau"]
COINT_FN = Literal["coint"]
COINT_CORR_FN = Literal[COINT_FN, CORR_FN]
Component = Literal["word", "punct", "special"]

PriceAction = Literal["buy", "sell", "wait"]
EntryType = Literal["long_only", "short_only", "long_or_short"]
EntryStruct = Literal["multiple", "single"]
ExitStruct = Literal["drawdown", "fifo", "lifo", "half_fifo", "half_lifo", "take_all"]
FixedPL = Literal["drawdown", "max_drawdown"]
