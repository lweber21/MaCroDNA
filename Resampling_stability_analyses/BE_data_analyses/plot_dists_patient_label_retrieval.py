import pandas as pd
import numpy as np 
import os

from scipy import stats
from scipy import spatial
from scipy.stats import spearmanr
from numpy.linalg import norm
from scipy.spatial.distance import pdist
import math
import random
import pickle
import multiprocessing
import time
import heapq
import sklearn
from sklearn.linear_model import LinearRegression

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

def label_point(x, y, val, ax):
	a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
	for i, point in a.iterrows():
		ax.text(point['x'], point['y'], str(point['val']), size=1)

def calculate_pdist(dna_src_dir, sample):

	dna = pd.read_csv(os.path.join(dna_src_dir,sample+"_annotated_filtered_normed_count_table.csv"),index_col=0)
	dna_np = dna.T.to_numpy()
	pdv = pdist(dna_np, 'minkowski', p=1.)
	return pdv

if __name__=="__main__":
	biop_sample = ["PAT20_CARD", "PAT20_ESO", "PAT9_NDBE", "PAT14_NDBE", "PAT16_NDBE", "PAT6_LGD", "PAT19_LGD", "PAT6_HGD", "PAT14_HGD", "PAT20_HGD1", "PAT16_EAC"]
	dna_src_dir = "./annotated_scDNAseq_filtered_cells_pseudocount_copynumber_log"
	res_path = "./macrodna_res_log_cross_cells/cross_cells_cell2cell_assignment_indexed.csv"
	biop_names = [x.replace("_","(")+")" for x in biop_sample]
	df = pd.read_csv(res_path, header=0)
	print(df.head())
	print(f"set of all existing steps in the df {set(df['step'].values.tolist())}")
	rna_cells = df['cell'].values.tolist()
	dna_pairs = df['predict_cell'].values.tolist()
	assert len(rna_cells) == len(dna_pairs), "the number of the rna cells and their dna pairs are not the same!"
	d_assignments = [[0 for i in range(len(biop_names))] for j in range(len(biop_names))]
	ith_list = []
	acc_list = []

	for biop_idx in range(len(biop_sample)):
		pdv = calculate_pdist(dna_src_dir = dna_src_dir, sample = biop_sample[biop_idx])
		metric = np.median(pdv)
		ith_list.append(metric)

	for l in range(len(rna_cells)):
		rna_b_name = rna_cells[l].split("_")[0]
		dna_b_name = dna_pairs[l].split("_")[0]
		rna_idx = biop_names.index(rna_b_name)
		dna_idx = biop_names.index(dna_b_name)
		d_assignments[rna_idx][dna_idx] += 1
	print(d_assignments)

	# calculate the accuracy for each biopsy
	for rna_idx in range(len(d_assignments)):
		# total number of rna cells from this rna biopsy
		total_rna_cells = sum(d_assignments[rna_idx])
		# number of cells that are accurately assigned
		accurately_assigned = d_assignments[rna_idx][rna_idx]
		# calculate the percentage of the accuracy 
		acc = float(accurately_assigned)/float(total_rna_cells)
		acc_list.append(100*acc)
		print(f"accuracy of biopsy {biop_names[rna_idx]} is {100*acc}")


	sns.set_style("darkgrid", {"axes.facecolor": ".87"})

	n = len(biop_names)
	colors = sns.husl_palette(l=0.5, s=1.0, h=0.8, as_cmap=True)(np.linspace(0,1,n))

	plt.cla()
	plt.clf()

	title_fontsize = 20
	ylabel_ = "Count"
	ylabel_fontsize = 20
	tick_size = 18
	f, ax = plt.subplots(7, 2, figsize=(14, 25), sharex=False, constrained_layout = True, sharey = False)
	f.delaxes(ax[4,1])
	f.delaxes(ax[5,1])
	f.delaxes(ax[6,1])

	g = sns.barplot(x = biop_names, y = d_assignments[0], color=colors[0], label=biop_names[0], ax=ax[0,0])
	ax[0,0].set_title(biop_names[0], fontweight="bold", fontsize=title_fontsize)
	ax[0,0].tick_params(axis='x', rotation=45, labelsize=tick_size)
	g.set_ylabel(ylabel_, fontsize=ylabel_fontsize)

	g = sns.barplot(x = biop_names, y = d_assignments[1], color=colors[1], label=biop_names[1], ax=ax[1,0])
	ax[1,0].set_title(biop_names[1], fontweight="bold", fontsize=title_fontsize)
	ax[1,0].tick_params(axis='x', rotation=45, labelsize=tick_size)
	g.set_ylabel(ylabel_, fontsize=ylabel_fontsize)

	g = sns.barplot(x = biop_names, y = d_assignments[2], color=colors[2], label=biop_names[2], ax=ax[2,0])
	ax[2,0].set_title(biop_names[2], fontweight="bold", fontsize=title_fontsize)
	ax[2,0].tick_params(axis='x', rotation=45, labelsize=tick_size)
	g.set_ylabel(ylabel_, fontsize=ylabel_fontsize)

	g = sns.barplot(x = biop_names, y = d_assignments[3], color=colors[3], label=biop_names[3], ax=ax[3,0])
	ax[3,0].set_title(biop_names[3], fontweight="bold", fontsize=title_fontsize)
	ax[3,0].tick_params(axis='x', rotation=45, labelsize=tick_size)
	g.set_ylabel(ylabel_, fontsize=ylabel_fontsize)

	g = sns.barplot(x = biop_names, y = d_assignments[4], color=colors[4], label=biop_names[4], ax=ax[4,0])
	ax[4,0].set_title(biop_names[4], fontweight="bold", fontsize=title_fontsize)
	ax[4,0].tick_params(axis='x', rotation=45, labelsize=tick_size)
	g.set_ylabel(ylabel_, fontsize=ylabel_fontsize)

	g = sns.barplot(x = biop_names, y = d_assignments[5], color=colors[5], label=biop_names[5], ax=ax[5,0])
	ax[5,0].set_title(biop_names[5], fontweight="bold", fontsize=title_fontsize)
	ax[5,0].tick_params(axis='x', rotation=45, labelsize=tick_size)
	g.set_ylabel(ylabel_, fontsize=ylabel_fontsize)

	g = sns.barplot(x = biop_names, y = d_assignments[6], color=colors[6], label=biop_names[6], ax=ax[6,0])
	ax[6,0].set_title(biop_names[6], fontweight="bold", fontsize=title_fontsize)
	ax[6,0].tick_params(axis='x', rotation=45, labelsize=tick_size)
	g.set_ylabel(ylabel_, fontsize=ylabel_fontsize)

	g = sns.barplot(x = biop_names, y = d_assignments[7], color=colors[7], label=biop_names[7], ax=ax[0,1])
	ax[0,1].set_title(biop_names[7], fontweight="bold", fontsize=title_fontsize)
	ax[0,1].tick_params(axis='x', rotation=45, labelsize=tick_size)
	g.set_ylabel(ylabel_, fontsize=ylabel_fontsize)

	g = sns.barplot(x = biop_names, y = d_assignments[8], color=colors[8], label=biop_names[8], ax=ax[1,1])
	ax[1,1].set_title(biop_names[8], fontweight="bold", fontsize=title_fontsize)
	ax[1,1].tick_params(axis='x', rotation=45, labelsize=tick_size)
	g.set_ylabel(ylabel_, fontsize=ylabel_fontsize)

	g = sns.barplot(x = biop_names, y = d_assignments[9], color=colors[9], label=biop_names[9], ax=ax[2,1])
	ax[2,1].set_title(biop_names[9], fontweight="bold", fontsize=title_fontsize)
	ax[2,1].tick_params(axis='x', rotation=45, labelsize=tick_size)
	g.set_ylabel(ylabel_, fontsize=ylabel_fontsize)

	g = sns.barplot(x = biop_names, y = d_assignments[10], color=colors[10], label=biop_names[10], ax=ax[3,1])
	ax[3,1].set_title(biop_names[10], fontweight="bold", fontsize=title_fontsize)
	ax[3,1].xaxis.set_tick_params(which='both', labelbottom=True)
	ax[3,1].tick_params(axis='x', rotation=45, labelsize=tick_size)
	g.set_ylabel(ylabel_, fontsize=ylabel_fontsize)

	plt.savefig(f'cross_cells_analysis.pdf')
