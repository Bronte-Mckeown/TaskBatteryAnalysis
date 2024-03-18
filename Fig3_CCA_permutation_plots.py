"Script to plot density plots for CCA permutation"

# import libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import rc

# read in data
null_corrs_df = pd.read_csv("scratch/results/cca/allTasks/nullcorrs_byPS_diff_pca.csv")
real_corrs_df = pd.read_csv("scratch/results/cca/allTasks/realcorrs.csv")

# set variables for saving out plots
n_comp = 5
num_iterations = 1000
type_shuff = "byPS_diff" 

# Set font size (this makes it 6ish in powerpoint)
plt.rcParams.update({'font.size': 5})

# Set figure size in inches (1 inch = 2.54 cm)
fig_width_cm = 2.5
fig_height_cm = 2.2
fig_width_inches = fig_width_cm / 2.54
fig_height_inches = fig_height_cm / 2.54

# loop over columns (i.e., CCA dimensions)
for col in range(n_comp-1):
    # Set figure size and DPI for each subplot
    fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches), dpi=1000)

    # plot real correlation as a dotted green line
    ax.axvline(x=real_corrs_df.iloc[0, col], color="green", linestyle="dotted", label="Real Correlation")
    # plot null correlations as a density plot (kde)
    sns.kdeplot(data=null_corrs_df.iloc[:, col], ax=ax, color="grey", fill=True, alpha=0.5, linewidth=0, label="Null Correlations")
    # set title
    ax.set_title("")
    # set x and y labels
    #ax.set(ylabel=None)
    ax.set_ylabel("Density")
    ax.set_xlabel("Correlation")
    # set x limits
    ax.set_xlim(-.1, .7)
    # remove legend
    ax.legend().set_visible(False)

    fig.tight_layout()

    # save figure
    fig.savefig(f"scratch/results/cca/allTasks/cca{col+1}_corrs_noRot_pca{n_comp}_{type_shuff}_{num_iterations}iters.png",
                bbox_inches = "tight", pad_inches=0.01,linewidth=0.5)

    # close the figure to free up memory
    plt.close(fig)

print("end")