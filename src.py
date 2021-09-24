import gudhi as gd
from gudhi import representations
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# data read ane preparation

local_path = '/mnt/c/Users/lubie/Desktop/Dokumenty/scratches/'
df = pd.read_csv(local_path + 'Data_Cortex_Nuclear.csv')
df = df.loc[(df['Treatment'] == 'Memantine') & (df['Behavior'] == 'C/S')]
df.drop('Treatment', inplace=True, axis=1)
df.drop('Behavior', inplace=True, axis=1)
df.drop('class', inplace=True, axis=1)
df.drop('MouseID', inplace=True, axis=1)

# analysing nulls

print(df.describe())
sns.heatmap(df.isnull())
plt.show()
df = df.dropna(axis='rows')

# pick control and trisomic data

control = df.loc[(df['Genotype'] == 'Control')]
control.drop('Genotype', axis=1)

trisomic = df.loc[(df['Genotype'] == 'Ts65Dn')]
trisomic.drop('Genotype', axis=1)

# prepare correlation matrix

control_corr = control.corr().to_numpy()
trisomic_corr = trisomic.corr().to_numpy()


# plot correlation and distance heatmaps

def plot_corr_heatmap(corr_matrix, title, file_name):
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix)
    plt.title(title)
    plt.savefig(local_path + file_name + '.png')
    plt.show()


plot_corr_heatmap(control_corr, "Correlation matric (control group)", 'control_corelation')
plot_corr_heatmap(trisomic_corr, "Correlation matric (trisomic group)", 'trisomic_corelation')


def plot_distance_heatmap(corr_matrix, title, file_name):
    plt.figure(figsize=(10, 6))
    sns.heatmap(1 - np.abs(corr_matrix))
    plt.title(title)
    plt.savefig(local_path + file_name + '.png')
    plt.show()


plot_distance_heatmap(control_corr, "Distance matric (control group)", 'control_distance')
plot_distance_heatmap(trisomic_corr, "Distance matric (trisomic group)", 'trisomic_distance')


# create distance matrix, simplicial complex and persistent homology

def get_simplex_tree(corr_matrix, max_tree_dimension=3):
    dist_matrix = 1 - np.abs(corr_matrix)
    rips = gd.RipsComplex(distance_matrix=dist_matrix)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_tree_dimension)
    return simplex_tree


def get_persistence(simplex_tree):
    return simplex_tree.persistence()


control_simplex_tree = get_simplex_tree(control_corr)
trisomic_simplex_tree = get_simplex_tree(trisomic_corr)

control_diag = get_persistence(control_simplex_tree)
trisomic_diag = get_persistence(trisomic_simplex_tree)


# plot persistence diagrams

def plot_pers_diagrams(pers_homology, diag_title, file_name):
    gd.plot_persistence_diagram(pers_homology)
    plt.title(diag_title)
    plt.savefig(local_path + file_name + '.png')
    plt.show()


plot_pers_diagrams(control_diag, "Persistence diagram (control group)", 'controldiag')
plot_pers_diagrams(trisomic_diag, "Persistence diagram (trisomic group)", 'trisomicdiag')


# plot persistence barcodes

def plot_pers_barcodes(pers_homology, barcode_title, file_name):
    gd.plot_persistence_barcode(pers_homology)
    plt.title(barcode_title)
    plt.savefig(local_path + file_name + '.png')
    plt.show()


plot_pers_barcodes(control_diag, "Persistence barcode (control group)", 'controldiag')
plot_pers_barcodes(trisomic_diag, "Persistence barcode (trisomic group)", 'trisomicdiag')


# plot persistence landscapes

def plot_pers_landscapes(simplex_tree, lsc_title, file_name, dimension):
    cls = representations.Landscape(resolution=1000)
    ls = cls.fit_transform([simplex_tree.persistence_intervals_in_dimension(dimension)])
    plt.plot(ls[0][:1000])
    plt.plot(ls[0][1000:2000])
    plt.plot(ls[0][2000:3000])
    plt.title(lsc_title)
    plt.savefig(local_path + file_name + '.png')
    plt.show()


plot_pers_landscapes(control_simplex_tree,
                     'Persistence landscape for degree 1 homology group (control group)',
                     'controlland_1',
                     1)
plot_pers_landscapes(trisomic_simplex_tree,
                     'Persistence landscape for degree 1 homology group (trisomic group)',
                     'trisomicland_1',
                     1)


# plot persistence barcodes of given dimension

def plot_pers_barcode_by_dim(pers_diagram, title, file_name, dim):
    curves = [elem for elem in pers_diagram if elem[0] == dim]
    gd.plot_persistence_barcode(curves)
    plt.xlabel("t")
    plt.ylabel(fr"$\beta_{dim}$")
    plt.title(title)
    plt.savefig(local_path + file_name + '.png')
    plt.show()


plot_pers_barcode_by_dim(control_diag, "Control group", 'controlb0', 0)
plot_pers_barcode_by_dim(control_diag, "Control group", 'controlb1', 1)
plot_pers_barcode_by_dim(control_diag, "Control group", 'controlb2', 2)
plot_pers_barcode_by_dim(trisomic_diag, "Trisommic group", 'trisomicb0', 0)
plot_pers_barcode_by_dim(trisomic_diag, "Trisommic group", 'trisomicb1', 1)
plot_pers_barcode_by_dim(trisomic_diag, "Trisommic group", 'trisomicb2', 2)


# calculate bottleneck distances


def calculate_bottleneck_dist_by_dim(simplex_tree_1, simplex_tree_2, dim):
    interval_1 = simplex_tree_1.persistence_intervals_in_dimension(dim)
    interval_2 = simplex_tree_2.persistence_intervals_in_dimension(dim)
    bottle_dist = gd.bottleneck_distance(interval_1, interval_2)
    return bottle_dist


bottleneck_distance_dim_0 = calculate_bottleneck_dist_by_dim(control_simplex_tree, trisomic_simplex_tree, 0)
bottleneck_distance_dim_1 = calculate_bottleneck_dist_by_dim(control_simplex_tree, trisomic_simplex_tree, 1)
bottleneck_distance_dim_2 = calculate_bottleneck_dist_by_dim(control_simplex_tree, trisomic_simplex_tree, 2)

print("Bottleneck distance for dimension 0", bottleneck_distance_dim_0)
print("Bottleneck distance for dimension 1", bottleneck_distance_dim_1)
print("Bottleneck distance for dimension 2", bottleneck_distance_dim_2)
