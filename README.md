# Bayesian Machine learning project : study of the Stochastic Weighted average method for Bayesian Bayesian_ML_SWAG

This is the repository of our study of the article : https://arxiv.org/abs/1902.02476

The students who work on this project are : Samuel Guilluy and Marwan Wehaiba El Khazen

All the code use Pytorch with cuda gpu.

We use as a reference to many functions we used the github repository of https://github.com/wjmaddox/swa_gaussian which implements the SWAG algorithm for many image dataset and image classification models.

We provide three jupyter notebook :

1.   NLP_comparasion_optimizer_AG_NEWS.ipynb contains a classification model on the AG NEWS Dataset of 4 optimizers : SGD + Step LR Scheduler, Adagrad, SGD + Cosine Scheduler.

2.   NLP_comparasion_optimizer_AG_NEWS.ipynb contains a classification model on the AG NEWS Dataset of 4 optimizers : SGD + Step LR Scheduler, Adagrad, SGD + Cosine Scheduler.

3.   plot_the_results.ipynb which read the csv files and plot the results. The resulting figures are saved in the folder : ./figures

The ./main_folder contains our implementation of the SWAG Algorithm.

The steps in order to launch the project are :
(for a launch under windows with cuda and Python 3.6)
 - Install the requirements presents in requirements.txt
 - then go to ./main_folder/experiments/train/ 
 
The datasets will be download automatically from pytorch.datasets

 - for testing the SWAG Algorithm on the AG NEWS dataset, you need to run the all_nlp_file.py file.
 - for testing the SWAG Algorithm on the Yelp review dataset, you need to run the all_nlp_file_Yelp_Review.py file.
 
The path to the folders need to be updated in the all_nlp_file.py and all_nlp_file_Yelp_review.py files.

In order to test SWAG on different hyperparameters, you can modify the parameters from the dictionnary 'dictionnary_arguments' wich is on top of two python files. 
The results will be save in the "optimizer_results" folder.

The ./main_folder/results store the parameters of our models at a given epoch frequence

The ./main_folder/swag floder contains the model of the SWAG class. It comes from https://github.com/wjmaddox/swa_gaussian but many functions are adjusted in order to match with our data requirements.

As a reminder of the main parameters : 
  - epochs : number of epochs
  - swa_start : the idex of the epoch where the SWA begin
  - eval_freq : the frequence on which we use the SWAG algorithm once the swa_start epoch is passed
  - max_num_models : the number of parameters from the max_num_epochs last epochs we wand to use in order to compute our SWAG Model.
 
