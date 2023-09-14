#!/usr/bin/python3
import sys
import numpy as np
import pandas as pd
import os
from os import path
from argparse import ArgumentParser
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


#define function, can be imported into another script
def pca_plot(genomes, generated_genomes_df=None, e="genome", dir="./"):   

    '''
    plots the PCA of genomes and generated_genomes at epoch e
    '''

    genomes_pca = genomes.drop(genomes.columns[1], axis=1)
    genomes_pca.columns = list(range(genomes_pca.shape[1]))
    genomes_pca.iloc[:,0] = 'Real'

    if generated_genomes_df is not None:
        generated_genomes_pca = generated_genomes_df.drop(generated_genomes_df.columns[1], axis=1)
        generated_genomes_pca.columns = list(range(genomes_pca.shape[1]))
        df_all_pca = pd.concat([genomes_pca, generated_genomes_pca])
    
    else:
        df_all_pca = genomes_pca
    
    pca = PCA(n_components=2)
    PCs = pca.fit_transform(df_all_pca.drop(df_all_pca.columns[0], axis=1))
    PCs_df = pd.DataFrame(data = PCs, columns = ['PC1', 'PC2'])
    PCs_df['Pop'] = list(df_all_pca[0])
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    pops = ['Real', 'AG']
    colors = ['r', 'b']

    for pop, color in zip(pops,colors):
        ix = PCs_df['Pop'] == pop
        ax.scatter(PCs_df.loc[ix, 'PC1'], 
            PCs_df.loc[ix, 'PC2'],
            c = color,
            s = 50, 
            alpha=0.2
        )
    ax.legend(pops)
    fig.savefig(f'{dir}/{e}_pca.pdf', format='pdf')

if __name__ == "__main__":
    #set required and optional arguments
    parser = ArgumentParser(description='Plot PCA of genotypic dataset(s)')
    parser.add_argument("input", help="link to dataset for PCA")
    #parser.add_argument("-i", "--input", help="link to dataset for PCA", action="store")
    parser.add_argument("-p", "--projection", help="project provided dataset on the PCA of the input",
                        action="store", default=None)
    parser.add_argument("-e", "--epoch", help="epoch number or plot name extension",
                        action="store",default="genome")
    parser.add_argument("-d", "--directory", help="PCA plot output directory",
                        action="store",default="./")

    #parse arguments
    args = parser.parse_args()

    #check arguments validity:
    if not path.isfile(args.input) :
        print("ERROR: provided input file does not exist")
        exit()

    #load genome dataset - required argument
    try:
        genomes = pd.read_csv(args.input, sep = ' ', header=None)
    except:
        print(f"ERROR: {genomes} - input file couldn't be opened")
        exit()

    #load optional arguments, if any

    e = args.epoch
    directory = args.directory

    if not path.isdir(directory):
        print("ERROR: output directory not valid!")
        exit()

    #load AG
    generated_genomes = args.projection
    if generated_genomes is not None:
        if path.isfile(generated_genomes):
            try:
                generated_genomes_df = pd.read_csv(generated_genomes, sep = ' ', header=None)
            
            except:
                print(f"ERROR: {generated_genomes} file couldn't be opened")
                exit()
            pca_plot(genomes=genomes, generated_genomes_df=generated_genomes_df, e=e, dir=directory)
            print("SUCCESS!")

        else:
            print("ERROR: provided -p file is not valid")
    else:
        pca_plot(genomes=genomes, e=e)
        print("SUCCESS!")
