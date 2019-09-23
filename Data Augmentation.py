import Augmentor
import numpy as np
np.random.seed(19680801)

path_to_data = "./Train"

# Create a pipeline
p = Augmentor.Pipeline(path_to_data)

# Add some operations to an existing pipeline.

p.ground_truth("./Label")

# First, we add a horizontal flip operation to the pipeline:
p.flip_left_right(probability=0.4)

# Now we add a vertical flip operation to the pipeline:
#p.flip_top_bottom(probability=0.7)

# Add a rotate90 operation to the pipeline:
#p.rotate90(probability=0.1)

p.rotate(probability=0.7, max_left_rotation=5, max_right_rotation=5)

# Here we sample 100,000 images from the pipeline.

# It is often useful to use scientific notation for specify
# large numbers with trailing zeros.
num_of_samples = int(50)

# Now we can sample from the pipeline:
p.sample(num_of_samples, multi_threaded=True)
p.process()