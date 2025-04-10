from typing import Literal

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
IGNORE_LIST = ["BRK.B", "BF.B", "CTAS"]

# Literal
HF_MODEL = Literal["prosusai", "yiyanghkust", "ziweichen", "aventiq_ai"]
CORR_FN = Literal["pearsonr", "spearmanr", "kendalltau"]
COINT_FN = Literal["coint"]
COINT_CORR_FN = Literal[COINT_FN, CORR_FN]
Component = Literal["word", "punct", "special"]
EntryType = Literal["long_only", "short_only", "long_or_short"]
