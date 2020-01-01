# -*- coding: utf-8 -*-
"""
@author: Sharad Kumar Gupta and Vanshika Gupta
"""
import numpy as np
np.random.seed(19680801)

import Augmentor
p = Augmentor.Pipeline("./data/train/image")

# Point to a directory containing ground truth data.
# Images with the same file names will be added as ground truth data
# and augmented in parallel to the original data.
p.ground_truth("./data/train/label")
# Add operations to the pipeline as normal:
p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
#p.flip_top_bottom(probability=0.5)
p.sample(200, multi_threaded=False)
#p.process()