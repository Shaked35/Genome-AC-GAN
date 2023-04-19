import copy
import os
import os.path
import random

import matplotlib.backends.backend_pdf
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow
from matplotlib import cm as cm
from matplotlib import pyplot as plt
from scipy import stats as scs
from scipy.spatial import distance
from scipy.stats import pearsonr, linregress
from sklearn.decomposition import PCA
from tensorflow.python import enable_eager_execution
from tensorflow.python.keras.utils.np_utils import to_categorical

from utils.consts import *

try:
    import ot

    ot_loaded = True
except ModuleNotFoundError:
    ot_loaded = False
try:
    import statsmodels.api as sm

    sm_loaded = True
except ModuleNotFoundError:
    sm_loaded = False


def reverse_dict(d):
    reversed_dict = {}
    for key, value in d.items():
        reversed_dict[value] = key
    return reversed_dict


def init_gpus():
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    enable_eager_execution()
    if gpus:
        for gpu in gpus:
            tensorflow.config.experimental.set_memory_growth(gpu, True)
            tensorflow.config.experimental.set_visible_devices(gpus[0], "GPU")
            logical_gpus = tensorflow.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    else:
        print("No GPU device found")


def all_same(items):
    return len(set(items)) == 1


def plotreg(x, y, keys, statname, col, ax=None):
    """
    Plot for x versus y with regression scores and returns correlation coefficient

    Parameters
    ----------
    x : array, scalar
    y : array, scalar
    statname : str
        'Allele frequency' LD' or '3 point correlation' etc.
    col : str, color code
        color

    """

    lims = [np.min(x), np.max(x)]
    r, _ = pearsonr(x, y)
    if sm_loaded:
        reg = sm.OLS(x, y).fit()
    if ax is None:
        ax = plt.subplot(1, 1, 1)
    if len(x) < 100:
        alpha = 1
    else:
        alpha = .6
    ax.plot(x, y, label=f"{keys[1]}: cor={round(r, 2)}", c=col, marker='o', lw=0, alpha=alpha)
    ax.plot(lims, lims, ls='--', alpha=1, c='black')
    ax.set_xlabel(f'{statname} in {keys[0]}')
    ax.set_ylabel(f'{statname} in {keys[1]}')
    ax.legend()
    return r


def plotregquant(x, y, keys, statname, col, step=0.05, cumsum=False, ax=None):
    """
    Plot quantiles for x versus y (every step) with regression scores and returns correlation coefficient

    Parameters
    ----------
    x : array, scalar
    y : array, scalar
    statname : str
        'Allele frequency' LD' or '3 point correlation' etc.
    col : str, color code
        color
    step : float
        step between quantiles
    cumsum : boolean
        plot cumulative sum of quantiles instead

    Return
    ------
    r: float
        Pearson correlation coefficient

    """
    q = np.arange(0, 1, step=step)
    x = np.nanquantile(x, q)
    y = np.nanquantile(y, q)
    if cumsum:
        x = np.cumsum(x)
        y = np.cumsum(y)
    r = plotreg(x=x, y=y, keys=keys, statname=f'Quantiles {statname}', col=col, ax=ax)
    return r


def plotquant(x, y, keys, statname, col, step=0.05, cumsum=False, ax=None):
    """
    Plot quantiles for y (every step) with regression scores of x quant vs y quant

    Parameters
    ----------
    x : array, scalar
    y : array, scalar
    statname : str
        'Allele frequency' LD' or '3 point correlation' etc.
    col : str, color code
        color
    step : float
        step between quantiles
    cumsum : boolean
        plot cumulative sum of quantiles instead
    """

    q = np.arange(0, 1, step=step)
    x = np.nanquantile(x, q)
    y = np.nanquantile(y, q)
    if cumsum:
        x = np.cumsum(x)
        y = np.cumsum(y)
        cum = ' (cumsum)'
    else:
        cum = ''

    r, _ = pearsonr(x, y)
    _ = sm.OLS(x, y).fit()
    linregress(x, y)
    if ax is None:
        ax = plt.subplot(1, 1, 1)
    # ax.plot(q, y , label=f"{keys[1]}: cor={round(r,2)} slope={round(reg.params[0],2)}", c=col, marker='o', lw=1, alpha=.5)
    ax.plot(q, y, label=f"{keys[1]}: cor={round(r, 2)}", c=col, marker='o', lw=1, alpha=.5)
    ax.set_xlabel(f'Quantile')
    ax.set_ylabel(f'{statname}{cum} in {keys[1]}')
    ax.legend()
    return r


### LD related ###
def get_dist(posfname, kept_preprocessing='all', kept_snp='all', region_len_only=False):
    legend = pd.read_csv(posfname, sep=' ')
    snps_on_same_chrom = legend.id[0].split(':')[0] == legend.id[legend.shape[0] - 1].split(':')[
        0]  # are all SNPs on the same chromosome ?
    ALLPOS = np.array(legend.position)
    region_len = ALLPOS[-1] - ALLPOS[0]
    if region_len_only:
        return None, region_len, snps_on_same_chrom

    # Order below is important
    if kept_preprocessing != 'all':
        ALLPOS = ALLPOS[kept_preprocessing]
    if kept_snp != 'all':
        ALLPOS = ALLPOS[kept_snp]
    # compute matrix of distance between SNPS (nsnp x nsnp)
    dist = np.abs(ALLPOS[:, None] - ALLPOS)

    # flatten and take upper triangular
    flatdist = dist[np.triu_indices(dist.shape[0])]
    return flatdist, region_len, snps_on_same_chrom


def plotoneblock(A, REF=None, keys=[None, None], statname='LD', ax=None,
                 matshow_kws={'cmap': cm.get_cmap('viridis_r'), 'vmin': 0, 'vmax': 1}, suptitle_kws={}):
    if REF is None:
        keys = [keys[0]] * 2  # mirrored/symmetrical matrix, identical name for both axes
    else:
        if REF.shape != A.shape:
            print("Warning: plotblock: matrices of different sizes cannot be arranged",
                  " below and above the diagonal of a single plot.",
                  " Set REF to None in plotoneblock, ie mirror to True in plotLDblock")
            return
        # mask =  np.triu_indices(A.shape[0], k=0)
        A[np.triu_indices(A.shape[0], k=0)] = 0
        A = A.T + REF

    if ax is None: plt.subplot(1, 1, 1)
    imgplot = ax.matshow(A, **matshow_kws)
    plt.colorbar(imgplot, shrink=.65, ax=ax)
    # ax.set_xlabel(f'{statname} in {keys[0]} (above diagonal)') # or as title ?
    plt.title(f'{statname} in {keys[0]} (above diagonal)')  # or as title ?
    ax.set_ylabel(f'{statname} in {keys[1]} (below diagonal)')
    if len(suptitle_kws) > 0: plt.suptitle(**suptitle_kws)
    plt.tight_layout()


def plotLDblock(hcor_snp, left=None, right=None, ref='Real', mirror=False, diff=False, is_fixed=None, is_fixed_dic=None,
                suptitle_kws={}):
    """
    Parameters
    ----------
    hcor_snp : list
        list of matrices containing r**2 pairwise values
    left : int
        starting SNP, if None the region starts at the very first SNP
    right : int
        finishing SNP (excluded), if None the region encompasses the very last SNP
    mirror : bool
        if True print the symmetrical matrix for each category
        if False, print LD in one dataset versus the reference dataset (only if datasets have the same number of sites)
    diff : bool
        show A-REF
    is_fixed : np.array(bool)
        bool array indicating sites that are fixed in at least one of the dataset
    is_fixed_dic : dict(np.array(bool))
        dict containing for each dataset (Real, GAN, ...) a bool array indicating fixed sites
    """

    if diff:
        cmap = cm.get_cmap('RdBu_r')
        cmap.set_bad('gray')
        vmin, vmax = -1, 1
        mirror = True
    else:
        cmap = cm.get_cmap('viridis_r')  # cool'
        cmap.set_bad('white')
        ## cmap = cm.get_cmap('jet', 10) # jet doesn't have white color
        ## cmap.set_bad('gray') # default value is 'k'
        vmin, vmax = 0, 1

    if (not mirror) or diff:
        if is_fixed is not None:
            REF = hcor_snp[ref][np.ix_(~is_fixed, ~is_fixed)]
        elif is_fixed_dic is not None:
            REF = hcor_snp[ref][np.ix_(~is_fixed_dic[ref], ~is_fixed_dic[ref])]
        else:
            REF = hcor_snp[ref]
        REF = REF[left:right, left:right]  # full ref

    if mirror:
        triREF = None
    else:
        triREF = copy.deepcopy(REF)
        triREF[np.triu_indices(triREF.shape[0], k=0)] = 0  # keep only lower triangle

    for i, (cat, hcor) in enumerate(hcor_snp.items()):
        if is_fixed is not None:
            A = hcor[np.ix_(~is_fixed, ~is_fixed)]
        elif is_fixed_dic is not None:
            A = hcor[np.ix_(~is_fixed_dic[cat], ~is_fixed_dic[cat])]
        else:
            A = hcor

        A = copy.deepcopy(A[left:right, left:right])  # copy
        if diff:
            A = REF - A
        ax = plt.subplot(1, len(hcor_snp), i + 1)
        plotoneblock(A=A,
                     REF=triREF,
                     keys=[cat, ref],
                     statname='LD',
                     matshow_kws={'cmap': cmap, 'vmin': vmin, 'vmax': vmax},
                     ax=ax, suptitle_kws=suptitle_kws)
        # ax.set_title("SNP correlation in {} for SNPs {} to {} ;   RSS: {:.4f}".format(cat,a,b, RSS))
        # plt.tight_layout()


def datatransform(dat, to_minor_encoding=False, min_af=0, max_af=1):
    dat = np.array(dat)
    af = np.mean(dat, axis=0)
    if to_minor_encoding:
        dat = (dat + (af > 1 - af)) % 2
        af = np.mean(dat, axis=0)

    if (min_af > 0) | (max_af < 1):
        kept = (af >= min_af) & (af <= max_af)
        dat = dat[:, kept]
    else:
        kept = 'all'  # np.full(len(af), fill_value=True)
    return dat, kept


def plotPCAscatter(pcdf, method, orderedCat, output_dir):
    pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(output_dir, "PCA_scatter_compare_" + method + ".pdf"))
    if 'coupled_with' in pcdf.columns:
        pcdf = pcdf.query('label == coupled_with')
    g = sns.FacetGrid(pcdf, col="label", col_order=orderedCat)
    g.map(sns.scatterplot, 'PC1', 'PC2', alpha=.1)
    pdf.savefig()
    g = sns.FacetGrid(pcdf, col="label", col_order=orderedCat)
    g.map(sns.scatterplot, 'PC3', 'PC4', alpha=.1)
    pdf.savefig()
    g = sns.FacetGrid(pcdf, col="label", col_order=orderedCat)
    g.map(sns.scatterplot, 'PC5', 'PC6', alpha=.1)
    plt.tight_layout()
    pdf.savefig()
    pdf.close()


def plotPCAsuperpose(pcdf, method, orderedCat, output_dir, colpal):
    fig, axs = plt.subplots(nrows=3, ncols=len(orderedCat),
                            figsize=(len(orderedCat) * 3.2, 3 * 3.2), constrained_layout=True)
    # fig, axs = plt.subplots(nrows=3, ncols=len(orderedCat), figsize = (10, 10/len(orderedCat)))
    ext = 1
    for i, pcx in enumerate([0, 2, 4]):
        win = 0
        # compute x and y ranges to force same dimension for all methods
        pcs = pcdf.drop(columns=['label', 'coupled_with'], errors='ignore').values
        xlim = (np.min(pcs[:, pcx]) - ext, np.max(pcs[:, pcx]) + ext)
        ylim = (np.min(pcs[:, pcx + 1]) - ext, np.max(pcs[:, pcx + 1]) + ext)

        for cat in orderedCat:
            if 'coupled_with' in pcdf.columns:
                reals = (pcdf.label == 'Real') & (pcdf['coupled_with'] == cat)
            else:
                reals = (pcdf.label == 'Real')
            # if cat=='Real': continue
            ax = axs[i, win]

            ax.scatter(pcdf.values[reals, pcx],
                       pcdf.values[reals, pcx + 1], alpha=.4)
            if cat != 'Real':
                keep = (pcdf.label == cat)
                ax.scatter(pcdf.values[keep, pcx],
                           pcdf.values[keep, pcx + 1], alpha=.4, color=colpal[cat])
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel("PC{}".format(pcx + 1))  # make PC label starts at 1
            ax.set_ylabel("PC{}".format(pcx + 2))
            ax.set_title(cat)
            win += 1
    fig.suptitle(method)
    plt.savefig(os.path.join(output_dir, "PCA_allel_compare_models_" + method + ".pdf"))


def plotPCAdensity(pcdf, method, orderedCat, output_dir):
    pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(output_dir, "PCA_densities_compare_" + method + ".pdf"))
    g = sns.FacetGrid(pcdf, col="label", col_wrap=len(orderedCat), col_order=orderedCat)
    g.map(sns.kdeplot, 'PC1', 'PC2', cmap="RdBu", cbar=True)  # Reds_d")
    pdf.savefig()
    g = sns.FacetGrid(pcdf, col="label", col_wrap=len(orderedCat), col_order=orderedCat)
    g.map(sns.kdeplot, 'PC3', 'PC4', cmap="Reds_d", cbar=True)
    pdf.savefig()
    g = sns.FacetGrid(pcdf, col="label", col_wrap=len(orderedCat), col_order=orderedCat)
    g.map(sns.kdeplot, 'PC5', 'PC6', cmap="Reds_d", cbar=True)
    pdf.savefig()
    pdf.close()


def plotPCAallfigs(pcdf, method, orderedCat, output_dir, colpal):
    # plotPCAscatter(pcdf, method, orderedCat, output_dir)
    if method != 'independent_PCA':
        plotPCAsuperpose(pcdf, method, orderedCat, output_dir, colpal)
    plotPCAdensity(pcdf, method, orderedCat, output_dir)


def computePCAdist(pcdf, method, output_dir, stat='wasserstein', reg=1e-3):
    scores_df = pd.DataFrame(columns=['stat', 'statistic', 'label', 'PC', 'method'])
    print(stat)
    for key in pd.unique(pcdf.label):
        if 'coupled_with' in pcdf.columns:
            reals = (pcdf.label == 'Real') & (pcdf['coupled_with'] == key)
            gen = (pcdf.label == key) & (pcdf['coupled_with'] == key)
        else:
            reals = (pcdf.label == 'Real')
            gen = (pcdf.label == key)

        n = np.sum(gen)  # nb samples
        ncomp = pcdf.drop(columns=['label', 'coupled_with'], errors="ignore").shape[1]
        a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples

        for pc, colname in enumerate(pcdf.drop(columns=['label', 'coupled_with'], errors="ignore")):
            if pc > 1:  # we just need scores for pc1,pc2
                continue
            if stat == 'wasserstein':
                sc = scs.wasserstein_distance(pcdf[reals].iloc[:, pc], pcdf[gen].iloc[:, pc])
                row_data = {'stat': [stat], 'statistic': [sc], 'label': [key], 'PC': [pc + 1], 'method': [method]}
                scores_df = pd.concat([scores_df, pd.DataFrame(row_data, index=[0])], ignore_index=True)

            elif stat == 'wasserstein2D':
                if pc > 0:  # we just need the score for (pc1,pc2)
                    continue
                if not ot_loaded:  # in case the library is not available
                    pcdf.to_csv(os.path.join(output_dir, f'PCA_{method}.csv'))
                    return None
                if pc < ncomp - 1:
                    xs = pcdf[reals].iloc[:, pc:pc + 2].values
                    xt = pcdf[gen].iloc[:, pc:pc + 2].values
                    # loss matrix
                    M = ot.dist(xs, xt)
                    M /= M.max()
                    sc = ot.sinkhorn2(a, b, M, reg)[0]
                    print(key, sc)
                    new_row = pd.DataFrame(
                        {'stat': [stat], 'statistic': [sc], 'reg': [reg], 'label': [key], 'PC': [f'{pc + 1}-{pc + 2}'],
                         'method': [method]})
                    scores_df = pd.concat([scores_df, new_row], ignore_index=True)

            else:
                print(f'{stat} is not a recognized stat option')
                return None

    scores_df.to_csv(os.path.join(output_dir, f"{stat}_PCA_" + method + ".csv"))
    return scores_df


def plotPrivacyLoss(Train, Test, output_dir, colpal, allcolpal):
    if Train in ['Real', '_Real']:
        Train = ''

    if not os.path.exists(os.path.join(output_dir, f'AA{Train}.csv.bz2')) or not os.path.exists(
            os.path.join(output_dir, f'AA{Test}.csv.bz2')):
        print(
            f"Warning: at least one of the file {os.path.join(output_dir, f'AA{Train}.csv.bz2')} or {os.path.join(output_dir, f'AA{Test}.csv.bz2')} does not exist")
        dfPL = None
    else:
        AA_Test = pd.read_csv(os.path.join(output_dir, f'AA{Test}.csv.bz2'))  # AA_Test-Gen
        AA_Train = pd.read_csv(os.path.join(output_dir, f'AA{Train}.csv.bz2'))  # AA_Train-Gen
        PL = dict()
        for cat in AA_Train.cat:
            if cat not in AA_Test.cat.values: continue
            # Privacy Loss = Test AA - Train AA
            PL[cat] = np.float(AA_Test[AA_Test.cat == cat].AATS) - np.float(AA_Train[AA_Train.cat == cat].AATS)
        if len(PL) > 0:
            dfPL = pd.DataFrame.from_dict(PL, orient='index', columns=['PrivacyLoss'])
            dfPL['cat'] = dfPL.index
        dfPL.to_csv(os.path.join(output_dir, f'PL_Train{Train}_Test{Test}.csv'))

        colors = [colpal[key] if key in colpal else allcolpal[key] for key in dfPL.cat.values]
        sns.barplot(x='cat', y='PrivacyLoss', data=dfPL, palette=colors)
        plt.axhline(0, color='black')
        plt.title("Nearest Neighbor Adversarial Accuracy - Privacy Loss")
        plt.ylim(-.5, .5)
        plt.savefig(os.path.join(output_dir, f"PrivacyLoss_Train{Train}_Test{Test}.pdf"))

    return dfPL


def computeAAandDist(dat, samplelabel, categ=None, refCateg='Real', saveAllDist=False, output_dir='./',
                     outFilePrefix='',
                     reloadDTT=None):
    """
    Compute AATS scores
    Args:
        dat           pd.dataframe of the genetic data with individuals as rows, and SNPs as columns
        samplelabel   array of label ('Real', 'RBM_it690', etc) for each individual of dat (ie each line of the dataset)
        categ         list of category of datasets to investigate, eg ['Syn'] or ['SYn', 'RBM'].
                      Default: all categories available in dat

    Returns:
        a tuple (AA, closest_all)
        AA (pd.dataframe):    contains AATS score for each investigated category
        DIST (pd.dataframe):   contains the distances to closest neighbor for different categories and pairs (Real, Real) (Real, Syn)  (Syn, Real), (Syn, Syn)


    Example:
        AA, DIST = computeAATS(dat, samplelabel, ['Syn','RBM'])

    """

    if (refCateg is not None) and (refCateg not in np.unique(samplelabel)):
        print(f'Error: It is mandatory to have individuals labeled as your reference categ {refCateg}',
              ' in your dataset. You can change the reference label using the refCateg argument')
        return

    if categ is None:
        categ = np.unique(samplelabel)

    nReal = (samplelabel == refCateg).sum()
    AA = pd.DataFrame(columns=['cat', 'AATS', 'AAtruth', 'AAsyn', 'ref'])
    DIST = pd.DataFrame(columns=['cat', 'dTS', 'dST', 'dSS'])

    # reference-reference distance
    DTTfilename = f'{output_dir}/{reloadDTT}'
    if (reloadDTT is None) or (not os.path.exists(DTTfilename)):
        dAB = distance.cdist(dat.loc[samplelabel.isin([refCateg]), :],
                             dat.loc[samplelabel.isin([refCateg]), :], 'cityblock')
        if reloadDTT is not None:
            print(f"DTT was not yet saved: we computed it and saved it to {DTTfilename}")
            np.savez_compressed(DTTfilename, DTT=dAB)
    else:
        dAB = np.load(DTTfilename)['DTT']

    if saveAllDist:
        np.savez_compressed(f'{output_dir}/{outFilePrefix}dist_{refCateg}_{refCateg}',
                            dist=dAB[np.triu_indices(dAB.shape[0], k=1)])

    np.fill_diagonal(dAB, np.Inf)
    dTT = dAB.min(axis=1)
    DTMP = pd.DataFrame({'cat': [refCateg], 'dTS': [dTT], 'dST': [dTT], 'dSS': [dTT]})
    DIST = pd.concat([DIST, DTMP], ignore_index=True)

    for cat in categ:
        if cat == refCateg:
            continue

        ncat = (samplelabel == cat).sum()
        if ncat == 0:
            print(f'Warning: no individuals labeled {cat} were found in your dataset.',
                  ' Jumping to the following category')
            continue

        if ncat != nReal:
            print(f'Warning: nb of individuals labeled {cat} differs from the the nb in refCateg.',
                  ' Jumping to the following category')
        print(cat)
        dAB = distance.cdist(dat.loc[samplelabel.isin([cat]), :], dat.loc[samplelabel.isin([refCateg]), :], 'cityblock')
        if saveAllDist:
            np.savez_compressed(f'{output_dir}/{outFilePrefix}dist_{cat}_{refCateg}', dist=dAB.reshape(-1))

        dST = dAB.min(axis=1)  # dST
        dTS = dAB.min(axis=0)  # dTS
        dAB = distance.cdist(dat.loc[samplelabel.isin([cat]), :], dat.loc[samplelabel.isin([cat]), :], 'cityblock')
        if saveAllDist:
            np.savez_compressed(f'{output_dir}/{outFilePrefix}dist_{cat}_{cat}',
                                dist=dAB[np.triu_indices(dAB.shape[0], k=1)])

        np.fill_diagonal(dAB, np.Inf)
        dSS = dAB.min(axis=1)  # dSS
        n = len(dSS)
        AAtruth = ((dTS > dTT) / n).sum()
        AAsyn = ((dST > dSS) / n).sum()
        AATS = (AAtruth + AAsyn) / 2
        new_row = pd.DataFrame(
            {'cat': [cat], 'AATS': [AATS], 'AAtruth': [AAtruth], 'AAsyn': [AAsyn], 'ref': [refCateg]})
        AA = pd.concat([AA, new_row], ignore_index=True)
        DTMP = pd.DataFrame({'cat': [cat], 'dTS': [dTS], 'dST': [dST], 'dSS': [dSS]})
        DIST = pd.concat([DIST, DTMP], ignore_index=True)

    return AA, DIST


def init_dataset(hapt_genotypes_path: str = REAL_10K_SNP_1000G_PATH,
                 extra_data_path: str = REAL_EXTRA_DATA_PATH,
                 target_column: str = DEFAULT_TARGET_COLUMN,
                 minimum_samples: int = DEFAULT_MINIMUM_SAMPLES,
                 without_extra_data: bool = False):
    real_data = load_real_data(extra_data_path, hapt_genotypes_path, without_extra_data)
    relevant_columns = get_relevant_columns(real_data, [SAMPLE_COLUMN_NAME, target_column])
    real_data = real_data[relevant_columns]
    real_data = filter_samples_by_minimum_examples(minimum_samples, real_data, target_column)
    # Extract the features into a separate matrix
    X = real_data[list(set(relevant_columns) - {SAMPLE_COLUMN_NAME, target_column})].values
    x_train = extract_x_values(real_data, relevant_columns, target_column)
    class_to_id = order_class_ids(X, real_data, target_column)
    class_id_to_counts, uniques, y_train = extract_y_column(class_to_id, real_data, target_column)
    return (x_train, y_train), class_id_to_counts, len(uniques), class_to_id


def extract_y_column(class_to_id, real_data, target_column):
    y_train = pd.DataFrame(real_data[target_column].replace(class_to_id))
    uniques, counts = np.unique(y_train, return_counts=True)
    class_id_to_counts = dict(zip(uniques, counts))
    y_train = to_categorical(y_train, num_classes=len(uniques))
    return class_id_to_counts, uniques, y_train


def extract_x_values(real_data, relevant_columns, target_column):
    x_train = real_data[list(set(relevant_columns) - {SAMPLE_COLUMN_NAME, target_column, "class_id"})]
    x_train = x_train.values
    return x_train


def load_real_data(extra_data_path, hapt_genotypes_path, without_extra_data=False):
    if without_extra_data:
        return pd.read_csv(hapt_genotypes_path)
    df = pd.read_csv(hapt_genotypes_path, sep=' ', header=None)
    df[1] = df[1].str.replace("_A", "")
    df[1] = df[1].str.replace("_B", "")
    df = df.set_index(df[1])
    df_data = pd.read_csv(extra_data_path, sep='\t')
    df_data = df_data.set_index(df_data[SAMPLE_COLUMN_NAME])
    return df.join(df_data)


def get_relevant_columns(input_df: pd.DataFrame, input_columns: list[str]):
    output_columns = []
    for column_name, column_type in input_df.dtypes.to_dict().items():
        if column_name in input_columns or column_type == "int64":
            output_columns.append(column_name)
    return output_columns


def filter_samples_by_minimum_examples(minimum_samples, real_data, target_column):
    uniques, counts = np.unique(real_data[[target_column]], return_counts=True)
    class_name_to_counts = dict(zip(uniques, counts))
    class_name_to_counts = {k: v for k, v in class_name_to_counts.items() if v > minimum_samples}
    real_data = real_data[real_data[target_column].isin(list(class_name_to_counts.keys()))]
    return real_data


def order_class_ids(X, real_data, target_column):
    pca = PCA(n_components=8)
    X_pca = pca.fit_transform(X)
    class_to_pca = real_data[[target_column]]
    class_to_pca.loc[:, "pca_sum"] = np.sqrt(np.square(X_pca).sum(axis=1))
    sorted_by_pca = pd.DataFrame(class_to_pca.groupby(target_column).sum()["pca_sum"]).sort_values("pca_sum")
    class_to_id = {class_name: class_id for class_id, class_name in enumerate(sorted_by_pca.index)}
    return class_to_id


def init_analysis_args(output_dir, models_to_test):
    print("* Loading data (short script sumstats)...")
    model_name_to_input_file = {model_name: f"../{data['path']}" for model_name, data in models_to_test.items()}
    model_name_to_color = {model_name: data["color"] for model_name, data in models_to_test.items()}
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
    print(f"Figures will be saved in {output_dir}")
    if os.path.exists(output_dir):
        print('This directory exists, the following files might be overwritten:')
        print(os.listdir(output_dir))
    return model_name_to_input_file, model_name_to_color, color_palette


def load_analysis_data(number_of_samples: int, model_name_to_input_file: dict):
    transformations = {'to_minor_encoding': False, 'min_af': 0, 'max_af': 1}

    datasets, model_keep_all_snps, sample_info = dict(), dict(), dict()
    for model_name, file_path in model_name_to_input_file.items():
        print(model_name, "loaded from", file_path)
        model_sequences = pd.read_csv(file_path, sep=' ', header=None)
        if 'GS-AC-GAN' in model_name:
            model_sequences.columns = [column if column == 0 else column + 1 for column in model_sequences.columns]
            model_sequences.insert(0, 1, [f"AG{sample_id}" for sample_id in range(model_sequences.shape[0])])
        if model_sequences.shape[1] == 808:  # special case for a specific file that had an extra empty column
            model_sequences = model_sequences.drop(columns=model_sequences.columns[-1])
        if model_sequences.shape[0] > number_of_samples:
            model_sequences = model_sequences.drop(
                index=np.sort(
                    np.random.choice(np.arange(model_sequences.shape[0]),
                                     size=model_sequences.shape[0] - number_of_samples,
                                     replace=False))
            )
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
    print("Dictionary of datasets:", len(datasets))
    return extra_sample_info, sample_info, datasets, transformations, model_keep_all_snps


def build_allele_frequency(datasets):
    """
    Compute counts of "1" allele (could be derived, alternative or minor allele depending on the encoding)
    And check whether some sites are fixed
    matching_SNPs will be set to True if all datasets have the same nb of SNPs
    in this case we automatically consider that there can be a one-to-one comparison
    ie 1st SNP of generated datset should mimic the 1st real SNP and so on
    :param datasets: model to genotypes sequences
    :return:
    """

    is_fixed_dic, nindiv, sum_alleles_by_position, allele_frequency = dict(), dict(), dict(), dict()
    for model, genotypes_sequences in datasets.items():
        nindiv[model] = genotypes_sequences.shape[0]
        sum_alleles_by_position[model] = np.sum(genotypes_sequences, axis=0)
        allele_frequency[model] = sum_alleles_by_position[model] / nindiv[model]
        is_fixed_dic[model] = (np.sum(genotypes_sequences, axis=0) % nindiv[model] == 0)
    models = datasets.keys()
    is_fixed = np.vstack([is_fixed_dic[cat] for cat in models]).any(axis=0)
    return sum_alleles_by_position, allele_frequency, is_fixed
