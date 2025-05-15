AUGMENTATION_PARAMS = {
    # Optical effects
    'fisheye': {
        'enable': True,
        'strength': 1.2,
        'probability': 1
    },
    'wideangle': {
        'enable': True,
        'k1': -0.5,
        'k2': 0.05,
        'k3': 0.05,
        'k4': 0,
        'probability': 1
    },

    # Atmospheric effects
    'fog': {
        'enable': True,
        'intensity': 0.8,
        'color': (0.9, 0.9, 0.9),
        'focal_length': 1200,
        'probability': 1
    },

    # Geometric effects
    # Augmentor pipeline
    'augmentor': {
        'enable': True,
        'contrast': {'probability': 0.5, 'min_factor': 0.8, 'max_factor': 1.2},
        'rotation': {'probability': 0.7, 'max_left_rotation': 15, 'max_right_rotation': 15},
        'brightness': {'probability': 0.5, 'min_factor': 0.8, 'max_factor': 1.2},
        'color': {'probability': 0.5, 'min_factor': 0.8, 'max_factor': 1.2},
        'hist_eq_prob': 0.3,
        'flip_lr_prob': 0.5,
        'flip_tb_prob': 0.2,
        'samples': 30
    }
}