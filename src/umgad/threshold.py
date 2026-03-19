from __future__ import annotations
import numpy as np

def select_threshold(scores: np.ndarray) -> tuple[float, int]:
    """UMGAD label-free threshold selection (Eq. 20-23).

    Steps:
      1) sort descending: s(1) >= ... >= s(n)
      2) moving average smoothing with window w = max(floor(0.0001*n), 5)
      3) compute first and second differences
      4) choose T = argmax |Δ2(i)|; tie-break: pick the candidate with smallest |s̄(i) - s̄(end)|
      5) threshold = s̄(T), anomalies are those with raw score >= threshold
    """
    s = np.asarray(scores).astype(np.float64)
    n = s.shape[0]
    order = np.argsort(-s)
    s_sorted = s[order]

    w = max(int(np.floor(0.0001 * n)), 5)
    if n - w - 1 <= 1:
        # tiny graph fallbacks
        thr = float(np.median(s_sorted))
        return thr, int(np.sum(s >= thr))

    # moving average
    cumsum = np.cumsum(np.insert(s_sorted, 0, 0.0))
    s_bar = (cumsum[w:] - cumsum[:-w]) / w   # length n-w+1

    d1 = s_bar[:-1] - s_bar[1:]
    d2 = d1[:-1] - d1[1:]

    mags = np.abs(d2)
    max_mag = mags.max()
    cand = np.where(mags == max_mag)[0]  # indices into d2
    # T corresponds to i in 1..n-w-1; our 0-based cand aligns to that.
    # tie-break: smallest difference to s_bar(end)
    endv = s_bar[-1]
    best = cand[np.argmin(np.abs(s_bar[cand] - endv))]
    T = int(best)
    thr = float(s_bar[T])
    num_pred = int(np.sum(s >= thr))
    return thr, num_pred

def predict(scores: np.ndarray) -> tuple[np.ndarray, float, int]:
    thr, k = select_threshold(scores)
    pred = (scores >= thr).astype(np.int64)
    return pred, thr, k
