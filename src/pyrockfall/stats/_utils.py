import numpy as np

def formatTextDist(x: float) -> str:
    if np.isposinf(x): return "inf"
    if np.isneginf(x): return "-inf"
    if np.isnan(x):    return "nan"
    return f"{float(x):.6g}"  # compact, no trailing zeros
