import numpy as np

from utils.box_utils import Spec, BoxSizes, generate_priors


image_size = 300
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    Spec(19, 16, BoxSizes(60, 105), [2, 3]),
    Spec(10, 32, BoxSizes(105, 150), [2, 3]),
    Spec(5, 64,  BoxSizes(150, 195), [2, 3]),
    Spec(3, 100, BoxSizes(195, 240), [2, 3]),
    Spec(2, 150, BoxSizes(240, 285), [2, 3]),
    Spec(1, 300, BoxSizes(285, 330), [2, 3])
]


priors = generate_priors(specs, image_size)

