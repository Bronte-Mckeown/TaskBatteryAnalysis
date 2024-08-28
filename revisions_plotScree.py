import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

def groupScree(data, group, filename, include_omni=False, order=None):
    '''This function generates combined Scree plots with slopes representing each level of a grouping variable.

    Parameters
    ----------
        data: pd.DataFrame, default=None
            The DataFrame containing the data for generating the Scree slope(s).

        group: str, default=None
            The grouping variable, the levels of which are used to generate each Scree slope.

        filename: str, default=None
            The intended name for the outputted .png file.

        include_omni: bool, default=False 
            Whether to include a Scree slope for the 'omnibus' solution representing the full dataset.

        order: list, default=None
            The specified ordering of the labels in the legend. If 'None', will sort labels alphabetically.

    Returns
    -------
        .png:
            The Scree plot with slopes for each level of a given grouping variable as well as the omnibus slope if include_omni is set to True.
    '''
    if group != None :
        #create unique list of names
        datasets = data[group].unique()

        #create a data frame dictionary to store your data frames
        DataFrameDict = {elem : pd.DataFrame() for elem in datasets}

        for key in DataFrameDict.keys():
            DataFrameDict[key] = data[:][data[group] == key]

        eigens = []
        for df in DataFrameDict:
            sample = DataFrameDict[df]
            sample = sample.drop(group, axis = 1)
            pca = PCA()
            sample_fit = pca.fit(sample)
            sample_eigens = pd.DataFrame(pca.explained_variance_)
            sample_eigens = sample_eigens.rename(columns={0: 'eigenvalue'})
            sample_eigens['dataset'] = df
            sample_eigens['compnum'] = sample_eigens.index + 1
            eigens.append(sample_eigens)

        screedf = pd.concat(eigens, axis = 0)


        data = data.drop([group], axis = 1)

    if include_omni :
        #define PCA model to use
        pca = PCA()

        #fit PCA model to data
        pca.fit(data)

        omni_eigens = pd.DataFrame(pca.explained_variance_)
        omni_eigens = omni_eigens.rename(columns={0:'eigenvalue'})
        omni_eigens['dataset'] = 'Omnibus'
        omni_eigens['compnum'] = omni_eigens.index + 1

        if group != None :
            screedf = pd.concat([screedf, omni_eigens], axis = 0)
        else :
            screedf = omni_eigens

    fig, ax = plt.subplots(figsize=(8,6))
    for label, df in screedf.groupby('dataset'):
        ax.plot(df.compnum, df.eigenvalue, alpha=1.0, lw=1.5, label=label)
    
    plt.legend()
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.xticks(list(range(1, data.shape[1] + 1)))

    handles, labels = plt.gca().get_legend_handles_labels()
    if order == None:
        if len(datasets) > 1:
            order = list(range(len(datasets)))
            plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
        else :
            order = [0]
    plt.savefig(str(filename) + '.png')
    plt.show()
    plt.close()

## Load relevant data
data = pd.read_csv('scratch/data/esq_demo_forPCA.csv')

data = data.filter(regex="_response")
data['dataset'] = 'Original'

## Generate Scree plot for just original dataset
groupScree(data=data, group='dataset', filename='scratch/results/revisions_originalScree')

## Load replication set and merge with original, designating a grouping variable.
data2 = pd.read_csv('scratch/data/esq_demo_replication.csv')

data2 = data2.filter(regex="_response")
data2['dataset'] = 'Replication'

fulldata = pd.concat([data, data2], axis = 0)

## Generate combined Scree plot with slopes for original and replication sets.
groupScree(data=fulldata, group="dataset", filename='scratch/results/revisions_combinedScree')