# Scoring and review

Written in M1b.

`Lexicographic` concatenates member keys in order, so each member must return
keys of one fixed length within a candidate batch. Different members may use
different lengths. If one member returns valid keys of differing lengths,
the whole batch receives judge errors and no composite scores are produced.
For example, a member returning `(1.0,)` for one candidate and `(1.0, 2.0)` for
another must pad or otherwise normalize its keys before combining it with
another judge.
