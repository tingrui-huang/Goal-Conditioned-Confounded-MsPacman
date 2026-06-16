"""Seaquest Stage-H0 hostile-field qualification (collection / parity / export tooling).

Made an explicit package (rather than a namespace package) so
`from seaquest_stage_h0.validate_recollection_parity import validate` works as long as
the repository ROOT is on sys.path — e.g. in Colab after
`sys.path.insert(0, '/content/Goal-Conditioned-Confounded-MsPacman')`.
"""
