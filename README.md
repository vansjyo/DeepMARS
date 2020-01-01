# DeepMARS
Deep Network for Discontinuity Detection in SHARAD Radargram of MARS.

Usage:
1. Please install the required packages using "requirements.txt" file.

	`pip install -r requirements.txt`

2. Keep the test and training images in the data folder.

3. Due to size issue in github, the model has not provided with the repository. Please download the trained model from below link. Keep the downloaded model in the home directory i.e. the code directory.

	https://cloud.iitmandi.ac.in/f/be50f650fa/?raw=1
	
4. If training is required on your data set please change "training=True" in the line 139 of DeepMARS.py code.
	
	`def __init__(self, img_rows=256, img_cols=256, training=False, log_dir="./tensorboard_net"):`

5. The results of the test data are available in /results/image/ folder.

Please feel free to contact Sharad Kumar Gupta @ sharadgupta27@gmail.com or Vanshika Gupta @ vanshika421@gmail.com
