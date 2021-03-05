This repository host additional information for the paper "Task-specific information outperforms surveillance-style big data in predictive analytics".

The repository contains information (full code) related to our model. In addition it contains background data to generate the plots for main results as well as additional auxiliary results and model output. The target and features used to train and test the models are not available in the repository for privacy reasons. See details working with the data in publication.

Table of content
- The scripts 1a_parallel_estimate.py and 1b_parallel_estimate_even_bin_size.py trains and tests the models using the target and features
- The script 2_export_data_plot_GPA_distribution.py collects the output for all iterations of the training and testing of models and output into 3 model output files that are found in the folder "data/".
- The notebook 3_output_generate.ipynb generates the main results of the paper as well as auxiliary results.
- The notebook 4_results_main_auxiliary.ipynb contains the main and additional results on robustness to changing the classifier or modifying the target variable.
