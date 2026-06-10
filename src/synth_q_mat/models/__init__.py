"""Thin wrappers around the three pretrained models in the pipeline.

    DiffCSP  -> crystal structure generator (the RL policy's action space)
    ALIGNN   -> topological-class probability predictor
    M3GNet   -> formation energy / stability evaluator (via matgl)

Weights are fetched by scripts/download_models.py into models/pretrained/.
"""
