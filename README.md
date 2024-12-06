# DS542_final
DS542 Deep Learning final project. Contrastive Learning on IMDb-Faces dataset

# Data Loading
There are no scripts for data ingestion, instead go to the following link: [Link to IMDb Dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

This is the original paper for which the IMDb-Faces dataset was created. On that page are two download buttons called "Download images meta data" and "Download faces only (7GB)". Use both to download imdb_meta.tar and imdb_crop.tar respectively.

Once you have both .tar files, you need to extract the internal .mat files. Do this however you please. I chose to use the Linux command "tar -xvf <filename>.tar"

After doing this, you will have directories that lead to directories with "imdb.mat" for the metadata and a group of numbered directories containing images for the cropped images.

# Data Exploration + Pre-processing
In the notebook data_analysis.ipynb, I briefly investigate the data, formatting the image paths with their corresponding names and dimensions. I qualitatively decided to only include images with at least 100 x 100 resolution for performance. I also decided to only include images of individuals with at least 50 images, though I think this was in error, as more data would have been better for training and I think the likelihood of overfitting is slim.

I also explored the idea of filtering on facial confidence score as provided in the original metadata, however I found that when I filtered even loosely on confidence that the resulting corpus was too small for training.

Candidate images are then sampled from the desired set and loaded into a /selected folder

Then, in the notebook data_preprocessing.ipynb, I load all the candidate images in from /selected and scale them to 224 x 224

# Model Training
The script SimCLR.py contains the training schedule.
Run this with "python SimCLR.py" followed by arguments flags if you want to change anything about the schedule.

I inlcuded the shell script that I used to run it on the Boston University Shared Computing Cluster.

More on this is included in my final writeup

# Model Assessment
In model_assessment.ipynb, I qualitatively explored the results of my training. My final training schedule feature 50 epochs and took about 34 hrs.
After 50 Epochs, I observed moderate accuracy, however there were some outliers, like the images with the odd color streaks from the original dataset.

In this notebook, I embed all of the testing images and then compare them to each other to determine if my embeddings meaningfully capture facial features.

The comments in the notebook should loosely take you through how to assess the model.






