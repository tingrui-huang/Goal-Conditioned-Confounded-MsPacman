"""Phase-2 oxygen probes (four-frame). Supervised diagnostics on the frozen raw_hf
dataset; NO critic training, NO oracle, NO env. Shared data + net + trainer here so the
three probe scripts (leakage, U->A, U->future) and the qualification report reuse one
frozen construction and one episode-level split.
"""
