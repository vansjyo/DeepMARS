# DeepMARS
CNN Model for Discontinuity Detection in SHARAD Radargram of MARS.

Paper Link : [https://www.mdpi.com/2076-3417/10/7/2279](https://www.mdpi.com/2076-3417/10/7/2279)

Usage:
1. Install the required packages using "requirements.txt" file.

	`pip install -r requirements.txt`

2. Keep the test and training images in the data folder.

3. Due to size issue in github, the model has not been provided with the repository. Please download the trained model from [this link](https://cloud.iitmandi.ac.in/f/be50f650fa/?raw=1). Keep the downloaded model in the home directory i.e. the code directory.
	
4. If training is required on your data set, change "training=True" in the line 139 of DeepMARS.py code.
	
	`def __init__(self, img_rows=256, img_cols=256, training=False, log_dir="./tensorboard_net"):`

5. The results of the test data are available in /results/image/ folder.

Feel free to contact Sharad Kumar Gupta (sharadgupta27@gmail.com) or Vanshika Gupta (vanshika421@gmail.com) in case of queries.
