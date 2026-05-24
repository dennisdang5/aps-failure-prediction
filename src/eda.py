import matplotlib.pyplot as plt
import seaborn as sns
import math

def compute_coefficient_of_variation(aps_df):
    coefficient_of_variation_CV = aps_df.std() / aps_df.mean()
    return coefficient_of_variation_CV.sort_values(ascending=False)

def plot_correlation_matrix(aps_df, figsize=(15,10)):
    plt.figure(figsize=figsize)
    sns.heatmap(aps_df.corr(), cmap='coolwarm', annot=False, vmin=-1, vmax=1)
    plt.title('APS Features Heatmap')
    plt.tight_layout()
    plt.show()


def get_top_cv_features(cv_series, n):
    return list(cv_series.head(n).index)

def plot_scatterplot(features_df, target_y, features_name, ncols=4, figsize=(15,10)):
    nrows = math.ceil(len(features_name) / ncols)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    ax = ax.flatten()
    for i, feature_name in enumerate(features_name):
        ax[i].scatter(target_y, features_df[feature_name])
        ax[i].set_xlabel('Class')
        ax[i].set_ylabel(feature_name)
        ax[i].set_title(feature_name)
        ax[i].set_yscale('symlog')

    for j in range(len(features_name), len(ax)):
        plt.delaxes(ax[j])

    plt.tight_layout()
    plt.show()

def plot_boxplot(features_df, target_y, features_name, ncols=4, figsize=(15,10)):
    nrows = math.ceil(len(features_name) / ncols)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    ax = ax.flatten()
    for i, feature_name in enumerate(features_name):
        sns.boxplot(x=target_y, y=features_df[feature_name], ax=ax[i])
        ax[i].set_xlabel('Class')
        ax[i].set_ylabel(feature_name)
        ax[i].set_title(feature_name)
        ax[i].set_yscale('symlog')
        ax[i].grid(False)
        ax[i].set_ylim(bottom=0)

    for j in range(len(features_name), len(ax)):
        plt.delaxes(ax[j])

    plt.tight_layout()
    plt.show()
