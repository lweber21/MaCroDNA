# Resampling and stability analyses on Barretts' esophagus data set

For all the following experiments, we prepared and preprocessed the DNA and RNA input data according to the instructions that we described in [Instructions for reproducing the results of our analysis on Barretts' esophagus data]https://github.com/NakhlehLab/MaCroDNA/tree/main/BE_data_analysis#instructions-for-reproducing-the-results-of-our-analysis-on-barretts-esophagus-data-set-originally-introduced-in).
Here, we describe the experiments briefly and provide the instructions for running their corresponding codes.

## Random assignment test for BE biopsies
The first experiment is the random assignment of scRNA-seq cells to scDNA-seq cells for each biopsy. For each biopsy, we repeated the random assignment for 100 million times to ensure we had enough number of samples. For each random assignment, we computed the sum of the Pearson correlation coefficients between the paired cells as the *score* of the random trial. Next, we compared the scores obtained from random assignments with that of MaCroDNA's assignment, calculated the p-value of the MaCroDNA assignment, and plotted the figures.
The script `random_assignment_test.py` performs this test. To run `random_assignment_test.py`:

1. Set `dna_src_dir` to the path to the directory containing the filtered (from the section [Filtering on the scDNA-seq read count tables](https://github.com/NakhlehLab/MaCroDNA/tree/main/BE_data_analysis#filtering-on-the-scdna-seq-read-count-tables)).
2. Set `rna_src_dir` to the path to the directory containing the filtered gene expression data (from the section [Filtering on the scRNA-seq gene expression tables](https://github.com/NakhlehLab/MaCroDNA/blob/main/BE_data_analysis/README.md#filtering-on-the-scrna-seq-gene-expression-tables)).
3. Set `tgt_dir` to the desired path for storing the results.

Please note that, here, we assume the DNA data of each biopsy is a CSV file stored in the `dna_src_dir` named as `<biopsy name>_annotated_filtered_normed_count_table.csv`. Similarly, for the RNA data, all of them are stored in `rna_src_dir` as CSV files named `<biopsy name>__filtered_normed_count_table.csv`.

The outputs of the above script consist of a dictionary containing the scores obtained from MaCroDNA on each biopsy, named `macrodna_objvals.pkl`, the MaCroDNA assignments of cells stored as `<biopsy name>_cell2cell_assignment_indexed.csv`, and all the random scores from the random trials for each biopsy saved as `<biopsy name>_random_samples.npy`. All these files can be found in the `tgt_dir`.

The sample outputs of this code for 100 random trials for each BE biopsy can be found in the directory [macrodna_res_log_random_test](https://github.com/NakhlehLab/MaCroDNA/tree/main/Resampling_stability_analyses/BE_data_analyses/sample_outputs/macrodna_res_log_random_test).

### Plotting the results of random assignments 
The code, named `plot_random_assignments.py` plots the histograms of the BE biopsies along with the red vertical line indicating the score obtained from MaCroDNA. To run this code, set `src_dir` to the path to the directory where the results of the random assignments are stored (`tgt_dir` in the `random_assignment_test.py` code). 

### Correlation between the two scoring metrics for random assignments (sum of Pearson correlation values vs. median of Pearson correlation values)
To account for the outliers that may affect the sum of Pearson correlation values as a score, we additionally measured the median of Pearson correlation values and compared the results of the two experiments. The code, named `random_assignment_test_meidan.py` replicates the same random trials (using the same random seeds) and stores the median of Pearson correlations. The code and outputs are the same as for `random_assignment_test.py` (except for the added suffix of _median_). 
After running `random_assignment_test.py`, we visualized the histograms along with the p-values for MaCroDNA's results using `plot_random_samples_median.py`. This code also works the same as `plot_random_samples.py`. To measure and visualize the correlation between the two experiments and scorings, we used `plot_sum_median_correlation.py`. To run this code, first, set the variables named `src_dir_meidan` and `src_dir_sum` to the paths you used for `tgt_dir`s in `random_assignment_test_meidan.py` and `random_assignment_test.py`, respectively. The output is a figure named `sum_median_correlation_plot.pdf` that shows a regression plot between the median scores of random assignments in the two experiments. The Pearson and Spearman correlation values are also demonstrated at top of the figure.


## Stability analysis of MaCroDNA's assignments for BE biopsies
To measure the stability of scRNA-seq cells’ assignments in the BE biopsies, we designed a leave-one-out experiment that involved introducing a small perturbation into the input data by leaving out one of the scRNA-seq cells.
Such perturbation in the RNA inputs might change the assignment of not only the left-out scRNA-seq cell but also the remaining scRNA-seq cells. Therefore, for each scRNA-seq cell, we aggregated the scDNA-seq cells’ IDs assigned to it across all the leave-one-out trials. Next, for each scRNA-seq cell, we collected the scDNA-seq cell IDs that were different from the assigned scDNA-seq cell ID when running MaCroDNA on the entire RNA data and defined the AII as the number of such different scDNA-seq cell IDs. The higher the AII, the more unstable and indefinitive the assignment of a scRNA-seq is in the presence of perturbation.

The script named `run_loo_experiment.py` performs the leave-one-out trials on all the BE biopsies. To run this code:

1. Set `dna_src_dir` to the path to the directory containing the filtered (from the section [Filtering on the scDNA-seq read count tables](https://github.com/NakhlehLab/MaCroDNA/tree/main/BE_data_analysis#filtering-on-the-scdna-seq-read-count-tables)).
2. Set `rna_src_dir` to the path to the directory containing the filtered gene expression data (from the section [Filtering on the scRNA-seq gene expression tables](https://github.com/NakhlehLab/MaCroDNA/blob/main/BE_data_analysis/README.md#filtering-on-the-scrna-seq-gene-expression-tables)).
3. Set `tgt_dir` to the desired path for storing the results. Here, this is set to `./macrodna_res_log_rna_stability/`.

The names of the input CSV files are assumed to follow the same format as in the experiment [Random assignment test for BE biopsies](https://github.com/NakhlehLab/MaCroDNA/blob/main/Resampling_stability_analyses/BE_data_analyses/README.md#random-assignment-test-for-be-biopsies).

The sample outputs of this code for one of the BE biopsies (named PAT6_LGD) can be found in the directory [macrodna_res_log_rna_stability](https://github.com/NakhlehLab/MaCroDNA/tree/main/Resampling_stability_analyses/BE_data_analyses/sample_outputs/macrodna_res_log_rna_stability).

### Visualizing the box plots of the assignment instability indices 
The script named `box_plots_from_loo_errors.py` draws the box plots showing the range of values for the assignment instability indices across all scRNA-seq cells for each biopsy. To run this code, set the variables `dna_src_dir` and `rna_src_dir` as for `run_loo_experiment.py` and set `src_dir` to the same path for `tgt_dir` in `run_loo_experiment.py`. The output is a figure named `diversity_idx_boxplots.pdf`.

### Visualizing the correlation between assignment instability index and heterogeneity score
For each biopsy, we calculated the pairwise L1-norm distances between all scDNA-seq cells; then, to avoid the effect of outliers, instead of the mean, we used the median of the distribution of the pairwise distances as the heterogeneity score. The script named `compute_loo_error.py` calculates the heterogeneity scores for all biopsies, and computes the Pearson and Spearman correlations between the heterogeneity scores and the assignment instability indices. Finally, it plots the scatter plot of the data along with the correlation coefficients.
To run this code, set the variables `dna_src_dir`, `rna_src_dir`, and `src_dir` the same way as for the previous code `box_plots_from_loo_errors.py`

## Retrieval of BE biopsy labels
In this experiment, we aggregated all the scDNA-seq and scRNA-seq cells from all biopsies and ran MaCroDNA on the pooled data without providing the cells' biopsy labels to the method. This way, we allowed the scRNA-seq cells to be assigned to any biopsy’s scDNA-seq cell. Having the true biopsy labels, we measured the accuracy of assignments per biopsy, as the number of the scRNA-seq cells that are assigned to a scDNA-seq cell from the same biopsy divided by the total number of scRNA-seq cells in the biopsy.  

The script named `macrodna_patient_label_retrieval.py` pools all the cells from the two modalities and runs MaCroDNA on the pooled data. To run this script:

1. Set `dna_src_dir` to the path to the directory containing the filtered (from the section [Filtering on the scDNA-seq read count tables](https://github.com/NakhlehLab/MaCroDNA/tree/main/BE_data_analysis#filtering-on-the-scdna-seq-read-count-tables)).
2. Set `rna_src_dir` to the path to the directory containing the filtered gene expression data (from the section [Filtering on the scRNA-seq gene expression tables](https://github.com/NakhlehLab/MaCroDNA/blob/main/BE_data_analysis/README.md#filtering-on-the-scrna-seq-gene-expression-tables)).
3. Set `tgt_dir` to the desired path for storing the results. Here, this is set to `./macrodna_res_log_rna_stability/`.

The code stores the outputs in the `tgt_dir` which will be used for measuring the accuracy in the following. 
The sample outputs of this code on all BE biopsies can be found in the directory [macrodna_res_log_cross_cells](https://github.com/NakhlehLab/MaCroDNA/tree/main/Resampling_stability_analyses/BE_data_analyses/sample_outputs/macrodna_res_log_cross_cells).

### Accuracy measurement and bar plots of the results 
The script named `plot_dists_patient_label_retrieval.py` calculates and prints out the accuracy of all BE biopsies in this experiment and draws the bar plots representing the distributions of the cell assignments across all biopsies. To run this code:

1. Set `dna_src_dir` to the path to the directory containing the filtered (from the section [Filtering on the scDNA-seq read count tables](https://github.com/NakhlehLab/MaCroDNA/tree/main/BE_data_analysis#filtering-on-the-scdna-seq-read-count-tables)).
2. Set `res_path` to the output file from `macrodna_patient_label_retrieval.py` that contains the cell assignment information. We set the name of this file to `cross_cells_cell2cell_assignment_indexed.csv"` and can be found in the `tgt_dir` of `macrodna_patient_label_retrieval.py`.

