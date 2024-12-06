# DS542_final
DS542 Deep Learning final project. Contrastive Learning on IMDb-Faces dataset

# Data Loading
There are no scripts for data ingestion, instead go to the following link: [Link to IMDb Dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

This is the original paper for which the IMDb-Faces dataset was created. On that page are two download buttons called "Download images meta data" and "Download faces only (7GB)". Use both to download imdb_meta.tar and imdb_crop.tar respectively.

Once you have both .tar files, you need to extract the internal .mat files. Do this howeverr you please. I chose to use the Linux command "tar -xvf <filename>.tar"

After doing this, you will have directories that lead to directories with "imdb_meta.mat" and "imdb_crop.mat".

# 