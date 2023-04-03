import argparse
import importlib
import itertools
import json
import pickle
import random
import warnings
from pathlib import Path

from scipy.stats import binned_statistic
from scipy.stats import sem
from sklearn.decomposition import PCA

from utils.consts import *
from utils.utils import *


def load_data(results_folder: str, experiment_name: str, output_dir: str, number_of_samples: int):
    print("* Loading data (short script sumstats)...")
    model_name_to_input_file = {'Real': "resource/10K_SNP_1000G_real.hapt",
                                'GAN': "fake_genotypes_sequences/preview_sequences/10K_SNP_GAN_AG_10800Epochs.hapt",
                                'RBM': "fake_genotypes_sequences/preview_sequences/10K_RBM.hapt",
                                'NEW_RBM': "fake_genotypes_sequences/preview_sequences/10K_SNP_RBM_AG_1050epochs.hapt",
                                'WGAN': 'fake_genotypes_sequences/preview_sequences/10K_SNP_RBM_AG_1050epochs.hapt',
                                "GS-AC-GAN": os.path.join(results_folder, experiment_name, "genotypes.hapt")}

    # same SNP positions for all datasets, so it is just repeated for all keys:
    model_name_to_color = dict({'Real': "gray",
                                'GAN': "red",
                                'RBM': "green",
                                'NEW_RBM': "blue",
                                'WGAN': "orange",
                                'GS-AC-GAN': "black"})
    # RBM_labels = ['RBM_ep{}'.format(ep) for ep in np.concatenate([np.linspace(200, 650, 10).astype(int), [690]])]
    # model_name_to_color.update(dict(zip(RBM_labels, sns.color_palette('Reds_r', len(RBM_labels) + 3))))
    # Update current color palette to the dataset type in input_files
    color_palette = {key: model_name_to_color[key] for key in model_name_to_input_file.keys()}
    sns.set_palette(color_palette.values())
    print(color_palette)
    sns.set_palette(color_palette.values())
    print(model_name_to_input_file.keys())
    sns.palplot(sns.color_palette())
    # set the seed so that the same real individual are subsampled (when needed)
    # to ensure consistency of the scores when adding a new model or a new sumstat
    np.random.seed(3)
    random.seed(3)
    transformations = {'to_minor_encoding': False, 'min_af': 0, 'max_af': 1}
    if not transformations is None:
        tname = ';'.join([f'{k}-{v}' for k, v in transformations.items()])
        output_dir = os.path.join(output_dir, tname + '/')
    print(f"Figures will be saved in {output_dir}")
    if os.path.exists(output_dir):
        print('This directory exists, the following files might be overwritten:')
        print(os.listdir(output_dir))
    # Load  data
    datasets, model_keep_all_snps, sample_info = dict(), dict(), dict()
    for model_name, file_path in model_name_to_input_file.items():
        print(model_name, "loaded from", file_path)
        model_sequences = pd.read_csv(file_path, sep=' ', header=None)
        if model_name == 'GS-AC-GAN':
            model_sequences.columns = [column if column == 0 else column + 1 for column in model_sequences.columns]
            model_sequences.insert(0, 1, [f"AG{sample_id}" for sample_id in range(model_sequences.shape[0])])
        print(model_sequences.shape)
        if model_sequences.shape[1] == 808:  # special case for a specific file that had an extra empty column
            model_sequences = model_sequences.drop(columns=model_sequences.columns[-1])
        if model_sequences.shape[0] > number_of_samples:
            model_sequences = model_sequences.drop(
                index=np.sort(
                    np.random.choice(np.arange(model_sequences.shape[0]),
                                     size=model_sequences.shape[0] - number_of_samples,
                                     replace=False))
            )
        print(model_sequences.shape)
        # overwrite file first column to set the label name chosen in infiles (eg GAN, etc):
        model_sequences[0] = model_name
        sample_info[model_name] = pd.DataFrame({'label': model_sequences[0], 'ind': model_sequences[1]})
        datasets[model_name] = np.array(model_sequences.loc[:, 2:].astype(int))

        # transformations can be maf filtering, recoding into major=0/minor=1 format
        if transformations is not None:
            datasets[model_name], model_keep_all_snps[model_name] = datatransform(datasets[model_name],
                                                                                  **transformations)
        print(model_name, datasets[model_name].shape)
    extra_sample_info = pd.DataFrame(np.concatenate(list(sample_info.values())), columns=['label', 'id'])
    print("Dictionnary of datasets:", len(datasets))
    # Compute counts of "1" allele (could be derived, alternative or minor allele depending on the encoding)
    # And check whether some sites are fixed
    # matching_SNPs will be set to True if all datasets have the same nb of SNPs
    # in this case we automatically consider that there can be a one-to-one comparison
    # ie 1st SNP of generated datset should mimic the 1st real SNP and so on
    models = model_name_to_input_file.keys()
    ac_d, ac_scaled = dict(), dict()
    nindiv = dict()
    is_fixed_dic = dict()
    for model, genotypes_sequences in datasets.items():
        nindiv[model] = genotypes_sequences.shape[0]
        print(model, nindiv[model])
        ac_d[model] = np.sum(genotypes_sequences, axis=0)
        ac_scaled[model] = ac_d[model] / nindiv[model]
        is_fixed_dic[model] = (ac_d[model] % nindiv[model] == 0)
        print(f"{is_fixed_dic[model].sum()} fixed sites in {model}")
    # is site fixed in at least one of the dataset ?
    # requires to have the same number of SNPs for all datasets
    # (makes sense for "matching" SNPs)
    if all_same([d.shape[1] for d in datasets.values()]):
        matching_SNPs = True
        is_fixed = np.vstack([is_fixed_dic[cat] for cat in models]).any(axis=0)
        print(f"{is_fixed.sum()} sites fixed in at least one dataset")
        [print("{count} fixed SNPs in {cat} that are not fixed in Real".format(
            count=((is_fixed_dic[cat]) & (~is_fixed_dic['Real'])).sum(),
            cat=cat
        )) for cat in models]
    else:
        matching_SNPs = False
        is_fixed = None
    print(f'Matching SNPs?: {matching_SNPs}')
    print('*****************\n*** INIT DONE ***\n*****************')
    return model_name_to_input_file, model_name_to_color, color_palette, datasets, ac_d, ac_scaled, \
        extra_sample_info, model_keep_all_snps, is_fixed, is_fixed_dic


def main(output_dir: str, number_of_samples: int, compute_AATS: bool, experiment_name: str,
         results_folder: str):
    model_name_to_input_file, model_name_to_color, color_palette, datasets, ac_d, ac_scaled, extra_sample_info, \
        model_keep_all_snps, is_fixed, is_fixed_dic = load_data(results_folder=results_folder,
                                                                experiment_name=experiment_name, output_dir=output_dir,
                                                                number_of_samples=number_of_samples)
    print("* Plotting allele frequency characteristics...")

    figwi = 12
    #    plt.figure(figsize=(15,5*len(infiles.keys())))
    l, c = len(model_name_to_input_file.keys()) - 1, 2
    plt.figure(figsize=(figwi * c / 4, figwi * l / 4))
    win = 1
    for i, model_name in enumerate(model_name_to_input_file.keys()):  # ['GAN','RBM']):
        if model_name == 'Real': continue
        plt.subplot(l, c, win)
        win += 1
        plt.plot(ac_scaled['Real'][(ac_d[model_name] == 0)], alpha=1, marker='.', lw=0)
        plt.ylabel("Allele frequency in Real")
        plt.title("Real frequency of alleles \n absent from {}".format(model_name))
        plt.subplot(l, c, win)
        win += 1
        plt.hist(ac_scaled['Real'][(ac_d[model_name] == 0)], alpha=1)
        plt.title("Hist real freq of alleles \n absent from {}".format(model_name))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "RealAC_for_0fixed_sites.pdf"))

    # Plotting Allele frequencies in Generated vs Real
    # below a certain real frequency (here set to 0.2 ie 20%)
    l, c = np.ceil(len(ac_scaled) / 3), 3
    # plt.figure(figsize=(15,10))
    plt.figure(figsize=(figwi, figwi * l / c))
    # plt.figure(figsize=(figwi, figwi/len(ac_scaled)))
    maf = 0.2
    keep = (ac_scaled['Real'] <= maf)
    for i, (model_name, val) in enumerate(ac_scaled.items()):
        ax = plt.subplot(int(l), c, i + 1)
        plotreg(x=ac_scaled['Real'][keep], y=val[keep],
                keys=['Real', model_name], statname="Allele frequency",
                col=color_palette[model_name], ax=ax)
        plt.title(f'{model_name} vs Real')
    plt.suptitle(f'AF below {maf} in Real')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'AC_generated_vs_Real_zoom.pdf'))

    # Plotting Allele frequencies in Generated vs Real
    l, c = np.ceil(len(ac_scaled) / 3), 3
    # plt.figure(figsize=(15,10))
    plt.figure(figsize=(figwi, figwi * l / c))
    # plt.figure(figsize=(figwi, figwi/len(ac_scaled)))
    # l,c=1,len(ac_scaled)
    for i, (model_name, val) in enumerate(ac_scaled.items()):
        ax = plt.subplot(int(l), 3, i + 1)
        plotreg(x=ac_scaled['Real'], y=val,
                keys=['Real', model_name], statname="Allele frequency",
                col=color_palette[model_name], ax=ax)
        plt.title(f'Allele Frequencies {model_name} vs Real')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'AC_generated_vs_Real.pdf'))

    # Plotting quantiles of Allele frequency distributions in Generated vs Real
    # Particularly useful if there is no one-to-one correspondence between SNPs (ie if matching_SNPs is False)
    # But computable in all case
    l, c = np.ceil(len(ac_scaled) / 3), 3
    # plt.figure(figsize=(15,10))
    plt.figure(figsize=(figwi, figwi * l / c))
    real = ac_scaled['Real']
    for i, (model_name, val) in enumerate(ac_scaled.items()):
        ax = plt.subplot(int(l), c, i + 1)
        plotregquant(x=real, y=val,
                     keys=['Real', model_name], statname="Allele frequency",
                     cumsum=False, step=0.05,
                     col=color_palette[model_name],
                     ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'AC_quantiles_generated_vs_Real.pdf'))

    # Recomputing counts for major/minor encoding
    # No effect if the datset was already encoded as minor
    ac_minor = dict()
    for key, ac in ac_scaled.items():
        ac_minor[key] = np.minimum(ac, 1 - ac)
    print(
        '*************************************************************\n*** Computation and plotting Allele Frequencies DONE. Figures saved in {} ***\n*************************************************************'.format(
            output_dir))

    print("* Computing and plotting PCA...")

    score_list = []
    number_of_components = 6  # change to compute more PCs

    methodname = "Combined PCA"
    method = 'combined_PCA'
    print(f'Computing {methodname} ...')
    pca = PCA(n_components=number_of_components)
    pcs = pca.fit_transform(
        np.concatenate(list(datasets.values()))
    )
    pcdf = pd.DataFrame(pcs, columns=["PC{}".format(x + 1) for x in np.arange(pcs.shape[1])])
    pcdf["label"] = extra_sample_info.label.astype('category')
    plotPCAallfigs(pcdf, methodname, orderedCat=model_name_to_input_file.keys(), output_dir=output_dir,
                   colpal=color_palette)
    plt.show()

    """
    # These scores could be computed, but it is better to compute them on the coupled PCA 
    # so that they do not change when removing or adding another dataset
    if check_all: # W2d distance computation is time consuming
        k = computePCAdist(pcdf,method,output_dir,stat='wasserstein')
        score_list.append(k)
        plt.show()

        k = computePCAdist(pcdf, method, output_dir, stat='wasserstein2D',reg=1e-3)
        score_list.append(k)
    """

    methodname = "Coupled PCA"
    method = 'coupled_PCA'
    print(f'Computing {methodname} ...')
    nReal = datasets['Real'].shape[0]
    pcdf = pd.DataFrame()
    for model_name in model_name_to_color.keys():
        pca = PCA(n_components=number_of_components)
        pcs = pca.fit_transform(
            np.concatenate([datasets['Real'], datasets[model_name]])
        )  # PCA on combined Real + model_name individuals
        # df = pd.DataFrame(pcs[nReal:,:], columns=["PC{}".format(x+1) for x in np.arange(pcs.shape[1])]) #keep only pc values for individuals in model_name
        # df.insert(number_of_components,'label',model_name)
        df = pd.DataFrame(pcs, columns=["PC{}".format(x + 1) for x in
                                        np.arange(
                                            pcs.shape[1])])  # keep only pc values for individuals in model_name
        df['label'] = np.concatenate([['Real'] * nReal, [model_name] * datasets[model_name].shape[0]])
        df['coupled_with'] = model_name
        pcdf = pd.concat([pcdf, df], ignore_index=True)

    # plot all PCA figures and compute KS
    plotPCAallfigs(pcdf, methodname, orderedCat=model_name_to_input_file.keys(), output_dir=output_dir,
                   colpal=color_palette)

    k = computePCAdist(pcdf, method, output_dir, stat='wasserstein')
    score_list.append(k)

    k = computePCAdist(pcdf, method, output_dir, stat='wasserstein2D', reg=1e-3)
    score_list.append(k)

    # compute if not matching_SNPs even if not check_all
    # because combined or coupled PCA are not possible in this case
    methodname = "Independent PCA"
    method = 'independent_PCA'
    print(f'Computing {methodname} ...')
    pcdf = pd.DataFrame()
    for model_name in model_name_to_color.keys():
        pca = PCA(n_components=number_of_components)
        pcs = pca.fit_transform(datasets[model_name])
        df = pd.DataFrame(pcs, columns=["PC{}".format(x + 1) for x in np.arange(pcs.shape[1])])
        df.insert(number_of_components, 'label', model_name)
        pcdf = pd.concat([pcdf, df], ignore_index=True)

    # plot all PCA figures
    plotPCAallfigs(pcdf, methodname, orderedCat=model_name_to_input_file.keys(), output_dir=output_dir,
                   colpal=color_palette)

    scores_pca = pd.concat(score_list, sort=False)
    scores_pca.to_csv(os.path.join(output_dir, 'scores_all_PCA.csv'))

    # average scores (distances) accross PC axes
    sc_sum_over_PCs = scores_pca.groupby(['method', 'stat', 'label'])['statistic'].sum()
    sc_mean_over_PCs = scores_pca.groupby(['method', 'stat', 'label'])['statistic'].mean()
    print(sc_mean_over_PCs)

    print(
        '*****************************************************************\n*** Computation and plotting PCA DONE. Figures saved in {} ***\n*****************************************************************'.format(
            output_dir))

    print("* Computing and plotting LD...")
    #### Compute correlation between all pairs of SNPs for each generated/real dataset

    model_names = model_name_to_input_file.keys()
    hcor_snp = dict()
    for i, model_name in enumerate(model_names):
        print(model_name)
        with np.errstate(divide='ignore', invalid='ignore'):
            # Catch warnings due to fixed sites in dataset (the correlation value will be np.nan for pairs involving these sites)
            hcor_snp[model_name] = np.corrcoef(datasets[model_name], rowvar=False) ** 2  # r2

    _, region_len, snps_on_same_chrom = get_dist(REAL_POSITION_FILE_NAME, region_len_only=True,
                                                 kept_preprocessing=model_keep_all_snps['Real'])

    nbins = 50
    nsamplesets = 10000
    logscale = True
    bins = nbins
    binsPerDist = nbins
    if logscale: binsPerDist = np.logspace(np.log(1), np.log(region_len), nbins)

    # Compute LD binned by distance
    # Take only sites that are SNPs in all datasets (intersect)
    # (eg intersection of SNPs in Real, SNPs in GAN, SNPs in RBM etc)
    # -> Makes sense only if there is a correspondence between sites

    binnedLD = dict()
    binnedPerDistLD = dict()
    kept_snp = ~is_fixed
    n_kept_snp = np.sum(kept_snp)
    realdist = get_dist(REAL_POSITION_FILE_NAME, kept_preprocessing=model_keep_all_snps['Real'], kept_snp=kept_snp)[
        0]
    mat = hcor_snp['Real']
    # filter and flatten
    flatreal = (mat[np.ix_(kept_snp, kept_snp)])[np.triu_indices(n_kept_snp)]
    isnanReal = np.isnan(flatreal)
    i = 1
    plt.figure(figsize=(10, len(hcor_snp) * 5))

    for model_name, mat in hcor_snp.items():
        flathcor = (mat[np.ix_(kept_snp, kept_snp)])[np.triu_indices(n_kept_snp)]
        isnan = np.isnan(flathcor)
        curr_dist = realdist

        # For each dataset LD pairs are stratified by SNP distance and cut into 'nbins' bins
        # bin per SNP distance
        ld = binned_statistic(curr_dist[~isnan], flathcor[~isnan], statistic='mean', bins=binsPerDist)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # so that empty bins do not raise a warning
            binnedPerDistLD[model_name] = pd.DataFrame({'bin_edges': ld.bin_edges[:-1],
                                                        'LD': ld.statistic,
                                                        # 'sd': binned_statistic(curr_dist[~isnan], flathcor[~isnan], statistic = 'std', bins=binsPerDist).statistic,
                                                        'sem': binned_statistic(curr_dist[~isnan], flathcor[~isnan],
                                                                                statistic=sem,
                                                                                bins=binsPerDist).statistic,
                                                        'model_name': model_name, 'logscale': logscale})

        # For each dataset LD pairs are stratified by LD values in Real and cut into 'nbins' bins
        # binnedLD contains the average, std of LD values in each bin
        isnan = np.isnan(flathcor) | np.isnan(flatreal)
        ld = binned_statistic(flatreal[~isnan], flathcor[~isnan], statistic='mean', bins=bins)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # so that empty bins do not raise a warning
            binnedLD[model_name] = pd.DataFrame({'bin_edges': ld.bin_edges[:-1],
                                                 'LD': ld.statistic,
                                                 'sd': binned_statistic(flatreal[~isnan], flathcor[~isnan],
                                                                        statistic='std',
                                                                        bins=bins).statistic,
                                                 'sem': binned_statistic(flatreal[~isnan], flathcor[~isnan],
                                                                         statistic=sem,
                                                                         bins=bins).statistic,
                                                 'model_name': model_name, 'logscale': logscale})

        # Plotting quantiles ?
        plotregquant(x=flatreal, y=flathcor,
                     keys=['Real', model_name], statname='LD', col=color_palette[model_name],
                     step=0.05,
                     ax=plt.subplot(len(hcor_snp), 2, i))
        i += 1
        plt.title(f'Quantiles LD {model_name} vs Real')

        # removing nan values and subsampling before doing the regression to have a reasonnable number of points
        isnanInter = isnanReal | isnan
        keepforplotreg = random.sample(list(np.where(~isnanInter)[0]), nsamplesets)
        plotreg(x=flatreal[keepforplotreg], y=flathcor[keepforplotreg],
                keys=['Real', model_name], statname='LD', col=color_palette[model_name],
                ax=plt.subplot(len(hcor_snp), 2, i))
        i += 1
        plt.title(f'LD {model_name} vs Real')
    plt.savefig(os.path.join(output_dir, "LD_generated_vs_real_intersectSNP.pdf"))

    # Plot LD as a fonction of binned distances
    # except when SNPs are spread accross different chromosomes
    if snps_on_same_chrom:  # (position_fname['Real']!="1kg_real/805snps.legend"):
        plt.figure(figsize=(5, 5))
        for model_name, bld in binnedPerDistLD.items():
            plt.errorbar(bld.bin_edges.values, bld.LD.values, bld['sem'].values, label=model_name, alpha=.65,
                         linewidth=3)
        plt.title("Binned LD +/- 1 sem")
        if (logscale): plt.xscale('log')
        # plt.yscale('log')
        plt.xlabel("Distance between SNPs (bp) [Left bound of distance bin]")
        plt.ylabel("Average LD in bin")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "correlation_vs_dist_intersectSNP.pdf"))

    # For each dataset LD pairs were stratified by LD values in Real, cut into nbins bins
    # binnedLD contains the average LD in each bin
    # Plot generated average LD as a function of the real average LD in the bins
    plt.figure(figsize=(10, 10))
    for model_name, bld in binnedLD.items():
        plt.errorbar(bld.bin_edges.values, bld.LD.values, bld['sem'].values, label=model_name, alpha=1, marker='o')
    plt.title("Binned LD +/- 1 sem")
    # if (logscale): plt.xscale('log')
    plt.xlabel("Bins (LD in Real)")
    plt.ylabel("Average LD in bin")
    plt.legend()
    plt.savefig(
        os.path.join(output_dir, 'LD_{}bins_{}fixedremoved.pdf'.format(nbins, 'logdist_' if logscale else '')))

    print("* Plotting LD block matrices...")

    # Set edges of the region for which to plot LD block matrix (l=0, f='end') for full region
    # not used as for now apart from the filename
    l_bound = None
    r_bound = None

    # mirror (bool): plot symmetrical matrix or generated vs real?
    # diff (bool): plot LD values or generated minus real ?

    for snpcode in ("fullSNP", "intersectSNP"):
        mirror, diff = False, False
        outfilename = f"LD_HEATMAP_{snpcode}_bounds={l_bound}-{r_bound}_mirror={mirror}_diff={diff}.pdf"
        # print(outfilename)
        fig = plt.figure(figsize=(10 * len(hcor_snp), 10))
        plotLDblock(hcor_snp,
                    left=l_bound, right=r_bound,  # None, None -> takes all SNPs
                    mirror=mirror, diff=diff,
                    is_fixed=is_fixed, is_fixed_dic=is_fixed_dic,
                    suptitle_kws={'t': outfilename}
                    )
        plt.title(outfilename)
        plt.savefig(os.path.join(output_dir, outfilename))
        plt.show()

    print(
        '****************************************************************\n*** Computation and plotting LD DONE. Figures saved in {} ***\n****************************************************************'.format(
            output_dir))

    print("* Computing pairwise distances, minimal distances, and AATS and saving to compressed files...")

    # Compute only pairwise distances

    haplo = np.concatenate(list(datasets.values())).T  # orientation of scikit allele

    # Compute AATS with reference being Test sets Test1 and Test2 if they exist

    # Variable defined in the main notebook but could be overwritten by:
    # compute_AATS=True # False

    outFilePrefix = ''
    for ref in ['Test1', 'Test2']:
        if not ref in model_name_to_input_file.keys(): continue
        if compute_AATS:
            print("Computing AATS with ref " + ref)
            AA, MINDIST = computeAAandDist(
                pd.DataFrame(haplo.T),
                extra_sample_info.label,
                model_name_to_input_file.keys(),
                refmodel_names=ref,
                saveAllDist=True,
                output_dir=output_dir,
                outFilePrefix=outFilePrefix)

            # save AA and MINDIST pd.DataFrame to csv
            # np.array of all pariwise distances are saved as npz automatically when calling computeAAandDist with saveAllDist=True
            AA.to_csv(os.path.join(output_dir, f'AA_{ref}.csv.bz2'), index=None)
            MINDIST.to_csv(os.path.join(output_dir, f'MINDIST_{ref}.csv.bz2'), index=None)
        else:
            print("Loading precomputed AATS and MINDIST")
            AA = pd.read_csv(os.path.join(output_dir, f'AA_{ref}.csv.bz2'))
            MINDIST = pd.read_csv(os.path.join(output_dir, f'MINDIST_{ref}.csv.bz2'))

    # Compute AATS with reference being 'Real' (supposed to be the label of the Training set)

    # Variable defined in the main notebook but could be overwritten by:
    # compute_AATS=True # False
    print(f'compute_AATS: {compute_AATS}')
    if compute_AATS:
        print("Computing AATS")
        AA, MINDIST = computeAAandDist(
            pd.DataFrame(haplo.T),
            extra_sample_info.label,
            model_name_to_input_file.keys(),
            saveAllDist=True,
            output_dir=output_dir,
            outFilePrefix=outFilePrefix)
        # save AA and MINDIST pd.DataFrame to csv
        # np.array of all pariwise distances are saved as npz automatically when calling computeAAandDist with saveAllDist=True
        AA.to_csv(os.path.join(output_dir, 'AA.csv.bz2'), index=None)
        MINDIST.to_csv(os.path.join(output_dir, 'MINDIST.csv.bz2'), index=None)
    else:
        print("Loading precomputed AATS and MINDIST")
        AA = pd.read_csv(os.path.join(output_dir, 'AA.csv.bz2'))
        MINDIST = pd.read_csv(os.path.join(output_dir, 'MINDIST.csv.bz2'))
    print('AATS obtained')
    # if already computed we can load the tables:
    AA = pd.read_csv(os.path.join(output_dir, 'AA.csv.bz2'))
    MINDIST = pd.read_csv(os.path.join(output_dir, 'MINDIST.csv.bz2'))
    ### Plot distribution of Pairwise Differences

    #### Distribution WITHIN model_namesories
    W = pd.DataFrame(columns=['stat', 'statistic', 'pvalue', 'label', 'comparaison'])

    plt.figure(figsize=(24, 12))
    plt.subplot(1, 2, 1)
    model_names = model_name_to_input_file.keys()
    for i, model_name in enumerate(model_names):
        subset = (np.load('{}/{}dist_{}_{}.npz'.format(output_dir, outFilePrefix, model_name, model_name)))['dist']
        if model_name == 'Real':
            subsetreal = subset
        sns.distplot(subset, hist=False, kde=True,
                     kde_kws={'linewidth': 3},  # 'bw':.02
                     label='{} ({} identical pairs)'.format(model_name, (subset == 0).sum()))

        sc = scs.wasserstein_distance(subsetreal, subset)
        W = W.append(
            {'stat': 'wasserstein', 'statistic': sc, 'pvalue': None, 'label': model_name, 'comparaison': 'within'},
            ignore_index=True)

    plt.title("Distribution of haplotypic pairwise difference within each dataset")
    plt.legend()
    subsetreal = None

    #### Distribution BETWEEN model_namesories
    plt.subplot(1, 2, 2)
    model_names = model_name_to_input_file.keys()
    for i, model_name in enumerate(model_names):
        subset = (np.load('{}/{}dist_{}_{}.npz'.format(output_dir, outFilePrefix, model_name, 'Real')))['dist']
        if model_name == 'Real':
            subsetreal = subset
        sns.kdeplot(subset, hist=False, kde=True,
                    kde_kws={'linewidth': 3},  # 'bw':.02
                    label='{} vs {} ({} identical pairs)'.format(model_name, 'Real', (subset == 0).sum()))

        sc = scs.wasserstein_distance(subsetreal, subset)
        W = W.append(
            {'stat': 'wasserstein', 'statistic': sc, 'pvalue': None, 'label': model_name, 'comparaison': 'between'},
            ignore_index=True)

    plt.title("Distribution of haplotypic pairwise difference between datasets")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "haplo_pairw_distrib.pdf".format("-".join(model_names))))

    scores = pd.concat([W])

    print(W)

    DISTmelt = MINDIST.melt(id_vars='model_name').rename(columns=str.title)
    g = sns.FacetGrid(DISTmelt, hue="Cat", height=7, col='Variable', hue_order=model_name_to_input_file.keys())
    # cut=0 : negative values have no meaning for distances, however be aware that this might accidently hide real picks at zero (due to copying for example)
    # check whether the full distribution is  similar or not (next cell)
    g.map(sns.distplot, "Value", hist=False, kde=True, kde_kws={'linewidth': 4, 'cut': 0})
    g.add_legend()
    plt.savefig(os.path.join(output_dir, "distrib_minimal_distances_cut.pdf"))

    DISTmelt = MINDIST.melt(id_vars='model_name').rename(columns=str.title)
    g = sns.FacetGrid(DISTmelt, hue="Cat", height=7, col='Variable', hue_order=model_name_to_input_file.keys())
    g.map(sns.distplot, "Value", hist=False, kde=True, kde_kws={'linewidth': 4})
    g.add_legend()
    plt.savefig(os.path.join(output_dir, "distrib_minimal_distances_full.pdf"))

    W = pd.DataFrame(columns=['stat', 'statistic', 'pvalue', 'label', 'comparaison'])
    for model_name in model_name_to_input_file.keys():
        for method in ['dTS', 'dST']:
            real = MINDIST[method][MINDIST.cat == 'Real']
            sc = scs.wasserstein_distance(real, MINDIST[method][MINDIST.cat == model_name])
            W = W.append(
                {'stat': 'wasserstein', 'statistic': sc, 'pvalue': None, 'label': model_name, 'comparaison': method},
                ignore_index=True)
    scores = pd.concat([scores, W])
    scores.to_csv(os.path.join(output_dir, "scores_pairwise_distances.csv"), index=False)

    plt.figure(figsize=(1.5 * len(model_names), 6))

    sns.barplot(x='Cat', y='Value', hue='Variable', palette=sns.color_palette('colorblind'),
                data=(AA.drop(columns=['PrivacyLoss', 'ref'], errors='ignore')).melt(id_vars='model_name').rename(
                    columns=str.title))
    plt.axhline(0.5, color='black')
    if 'Real_test' in AA.cat.values:
        plt.axhline(np.float(AA[AA.cat == 'Real_test'].AATS), color=sns.color_palette()[0], ls='--')
    plt.ylim(0, 1.1)
    plt.title("Nearest Neighbor Adversarial Accuracy on training (AATS) and its components")
    plt.savefig(os.path.join(output_dir, "AATS_scores.pdf"))

    Test = '_Test2'
    Train = ''  # means Training set is Real
    dfPL = plotPrivacyLoss(Train, Test, output_dir, color_palette, model_name_to_color)

    # Compute PL for the real dataset Test1
    # Useful if an RBM with alternative training scheme (cf paper) is in the list of models
    # Because Test1 served for initializing the RBM sampling in this case
    Test = '_Test2'
    Train = '_Test1'
    dfPL = plotPrivacyLoss(Train, Test, output_dir, color_palette, model_name_to_color)

    print(
        '************************************************************************\n*** Computation and plotting DIST/AATS DONE. Figures saved in {} ***\n************************************************************************'.format(
            output_dir))

    print("* Computing 3-point correlation for different gap values...")

    def get_counts(haplosubset, points):
        counts = np.unique(
            np.apply_along_axis(
                lambda x: ''.join(map(str, x[points])),
                # lambda x: ''.join([str(x[p]) for p in points]),
                0, haplosubset),
            return_counts=True)
        return (counts)

    def get_frequencies(counts):
        l = len(counts[0][0])  # haplotype length
        nind = np.sum(counts[1])
        f = np.zeros(shape=[2] * l)
        for i, allele in enumerate(counts[0]):
            f[tuple(map(int, allele))] = counts[1][i] / nind
        return f

    def three_points_cor(haplosubset, out='all'):
        F = dict()
        for points in [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]:
            strpoints = ''.join(map(str, points))
            F[strpoints] = get_frequencies(
                get_counts(haplosubset, points)
            )

        cors = [
            F['012'][a, b, c] - F['01'][a, b] * F['2'][c] - F['12'][b, c] * F['0'][a] - F['02'][a, c] * F['1'][b] + 2 *
            F['0'][a] * F['1'][b] * F['2'][c] for a, b, c in itertools.product(*[[0, 1]] * 3)]
        if out == 'mean':
            return (np.mean(cors))
        if out == 'max':
            return (np.max(np.abs(cors)))
        if out == 'all':
            return (cors)
        return (ValueError(f"out={out} not recognized"))

    # def mult_three_point_cor(haplo, extra_sample_info, model_name, picked_three_points):
    #    return [three_points_cor(haplo[np.ix_(snps,extra_sample_info.label==model_name)], out='all') for snps in picked_three_points]

    # set the seed so that the same real individual are subsampled (when needed)
    # to ensure consistency of the scores when adding a new model or a new sumstat
    np.random.seed(3)
    random.seed(3)

    # Compute 3 point correlations results for different datasets and different distances between SNPs

    # pick distance between SNPs at which 3point corr will be computed
    # (defined in nb of snps)
    # a gap of -9 means that snp triplets are chosen completely at random (not predefined distance)
    # for each category we randomly pick 'nsamplesets' triplets

    # if datasets have different nb of snps, for convenience we will sample
    # slightly more at the beginning of the chunk

    gap_vec = [1, 4, 16, 64, 256, 512, 1024, -9]
    nsamplesets = 1000
    min_nsnp = min([dat.shape[1] for dat in datasets.values()])
    cors_meta = dict()
    for gap in gap_vec:
        print(f'\n gap={gap} SNPs', end=' ')
        if gap < 0:
            # pick 3 random snps
            picked_three_points = [random.sample(range(min_nsnp), 3) for _ in range(nsamplesets)]
        else:
            try:
                # pick 3 successive snps spearated by 'gap' SNPs
                step = gap + 1
                picked_three_points = [np.asarray(random.sample(range(min_nsnp - 2 * step), 1)) + [0, step, 2 * step]
                                       for _
                                       in range(nsamplesets)]
            except:
                continue  # if there were not enough SNPs for this gap
        cors = dict()

        for model_name in model_name_to_input_file.keys():
            print(model_name, end=' ')
            # cors[model_name]=[three_points_cor(haplo[np.ix_(snps,extra_sample_info.label==model_name)], out='all') for snps in picked_three_points]
            cors[model_name] = [three_points_cor(datasets[model_name][:, snps].T, out='all') for snps in
                                picked_three_points]

        cors_meta[gap] = cors.copy()

    # print(cors_meta)

    with open(os.path.join(output_dir, "3pointcorr.pkl"), "wb") as outfile:
        pickle.dump(cors_meta, outfile)

    # Plot 3-point correlations results

    plt.figure(figsize=(2 * len(cors_meta), 7))
    # plt.figure(figsize=(figwi,figwi/2))
    for i, gap in enumerate((cors_meta).keys()):
        ax = plt.subplot(2, int(np.ceil(len(cors_meta) / 2)), int(i) + 1)
        cors = cors_meta[gap]
        real = list(np.array(cors['Real']).flat)
        lims = [np.min(real), np.max(real)]
        for key, val in cors.items():
            if key == 'Real': continue
            val = list(np.array(val).flat)
            plotreg(x=real, y=val, keys=['Real', key],
                    statname='Correlation', col=color_palette[key], ax=ax)
        if gap < 0:
            plt.title('3-point corr for random SNPs')
        else:
            plt.title(f'3-point corr for SNPs sep. by {gap} SNPs')

        plt.legend(fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3point_correlations.jpg'), dpi=300)  # can pick one of the format
    plt.savefig(os.path.join(output_dir, '3point_correlations.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, '3point_correlations.pdf'), dpi=300)

    # Same plot with axes limit fixed to (-0.1,0.1) for the sake of comparison

    plt.figure(figsize=(4 * len(cors_meta), 14))
    # plt.figure(figsize=(figwi,figwi/2))
    for i, gap in enumerate((cors_meta).keys()):
        ax = plt.subplot(2, int(np.ceil(len(cors_meta) / 2)), int(i) + 1)
        cors = cors_meta[gap]
        real = list(np.array(cors['Real']).flat)
        lims = [np.min(real), np.max(real)]
        for key, val in cors.items():
            if key == 'Real': continue
            val = list(np.array(val).flat)
            plotreg(x=real, y=val, keys=['Real', key],
                    statname='Correlation', col=color_palette[key], ax=ax)
            ax.set_xlim((-.1, .1))
            ax.set_ylim((-.1, .1))

        if gap < 0:
            plt.title('3-point corr for random SNPs')
        else:
            plt.title(f'3-point corr for SNPs sep. by {gap} SNPs')

        plt.legend(fontsize='small')
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, '3point_correlations_fixlim.pdf'), dpi=300)
    plt.savefig(os.path.join(output_dir, '3point_correlations_fixlim.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, '3point_correlations_fixlim.jpg'), dpi=300)

    print(
        '************************************************************************\n*** Computation and plotting 3-point cor DONE. Figures saved in {} ***\n************************************************************************'.format(
            output_dir))


def parse_args():
    parser = argparse.ArgumentParser(description='compare old models with new models')
    parser.add_argument('--number_of_samples', type=int, default=NUMBER_OF_EXPERIMENT_SAMPLES,
                        help='how many samples to take from each dataset')
    parser.add_argument('--compute_AATS', type=bool, default=True,
                        help='path to real extra data with classes file')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_EXPERIMENT_OUTPUT_DIR,
                        help='where put the output results')
    parser.add_argument('--experiment_name', type=str, default=DEFAULT_EXPERIMENT_NAME,
                        help='experiment name')
    parser.add_argument('--results_folder', type=str, default=SEQUENCE_RESULTS_PATH,
                        help='results path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    output_dir = os.path.join(args.output_dir, args.experiment_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    main(output_dir=output_dir, number_of_samples=args.number_of_samples,
         compute_AATS=args.compute_AATS, experiment_name=args.experiment_name, results_folder=args.results_folder)
