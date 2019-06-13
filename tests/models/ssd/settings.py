#-*- coding: utf-8 -*-
# Extract from presets.py and ssd.py

__all__ = ['SSD_SETTINGS']

SSD_SETTINGS = {
    'ssd_512_mobilenet1_0_voc': {
        "sizes": [51.2, 102.4, 189.4, 276.4, 363.52, 450.6, 492],
        "ratios": [[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 3 + [[1, 2, 0.5]] * 2,
        "steps": [16, 32, 64, 128, 256, 512],
        "stds": (0.1, 0.1, 0.2, 0.2),   # default
        "offset": 0.5,    # default
        "flip": False,    # default
        "clip": False,    # default
        "num_classes": 21,
        "nms_thresh": 0.45,   # default
        "nms_topk": 400,  # default
        "post_nms": 100,  # default
        "conf_thresh": 0.01  # default
    }
}