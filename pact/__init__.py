"""PACT: Position As Confounder for Treatment — graph uplift modeling.

Module layout:

  layers.py   - hand-rolled GraphConvolution + multi-head GAT (no PyG)
  fusion.py   - attention-based fusion of node features and positional encoding
  heads.py    - MLP head builder, X-learner heads (mu, tau, log-sigma2, prop)
  model.py    - full PACT model (GPE -> GAT -> heads) and ablation variants
  losses.py   - masked / variance-weighted X-learner loss + uncertainty weighting
  dgp.py      - semi-synthetic data generation (heteroscedastic, community bias)
  metrics.py  - PEHE, ATE, Qini, AUUC, Lift@k
  data.py     - dataset loaders (graph + real-world uplift)
  train.py    - unified training loop for s/t/x/pact learners
  main.py     - CLI entry point
"""
