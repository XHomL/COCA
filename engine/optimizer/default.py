HYPER_DICT = {
    'linear': {
        'optim': "adamw",
        'lr': 1e-4,
        'weight_decay': 0.01,
        'lr_scheduler': "cosine",
        'warmup_iter': 50,
        'warmup_type': "linear",
        'warmup_min_lr': 1e-5,
    },
    'adapter': {
        'optim': "adamw",
        'lr': 1e-4,
        'weight_decay': 0.01,
        'lr_scheduler': "cosine",
        'warmup_iter': 50,
        'warmup_type': "linear",
        'warmup_min_lr': 1e-5,
    },
}