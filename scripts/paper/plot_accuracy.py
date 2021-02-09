'''
Plot accuracy
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import argparse

acc_vs_param = {
        "TRBA": { "Accuracy": 83.5, "Parameters": 49.6},
        "STAR-Net": { "Accuracy": 80.1, "Parameters": 48.9},
        "RARE": { "Accuracy": 81.2, "Parameters": 10.8},
        "Rosetta": { "Accuracy": 77.8, "Parameters": 44.3},
        "GCRNN": { "Accuracy": 77.5, "Parameters": 4.8},
        "R2AM": { "Accuracy": 78.4, "Parameters": 2.9},
        "CRNN": { "Accuracy": 76.1, "Parameters": 8.5},
        "Wordformer-Tiny": { "Accuracy": 80.1, "Parameters": 5.4},
        "Wordformer-Small": { "Accuracy": 82.3, "Parameters": 21.5},
        }

def plot_(data, title, ylabel="Accuracy", xlabel="Parameters", ncolors=9, vspace=0.2, is_scatter=True):
    plt.rc('font', size=12) 
    plt.rc('axes', titlesize=16)
    plt.rc('xtick', labelsize=14)

    fig, ax = plt.subplots()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    colors = sns.color_palette()[0:ncolors]
    markers = ['^', 's', 'o', 'D', '*', 'P', 'x', 'd', '+']

    x = np.arange(0,60,10)
    ax.set_xticks(x)

    i = 0
    labels = []
    has_label = False
    for key, val in data.items():
        label = key
        acc = val["Accuracy"]
        par = val["Parameters"]
        color = colors[i]

        ax.scatter(par, acc, marker=markers[i], s=100, label=label, color=color)
        ax.annotate(key, (par, acc))

        i = i + 1

    
    title = title.replace(" ", "_")
    title = title.replace("%", "")
    plt.savefig(title + ".png")
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wordformer results')
    parser.add_argument('--data',
                        default=None,
                        help='Data to plot')
    parser.add_argument('--vspace',
                        default=0.2,
                        type=float,
                        help='Vertical space in bar graph label')
    args = parser.parse_args()
    ylabel = "Accuracy (%)"
    xlabel = "Parameters (M)"

    title = "Accuracy vs Number of Parameters"
    data = acc_vs_param
    ncolors = 9
    is_scatter = True

    plot_(data=data, title=title, ylabel=ylabel, xlabel=xlabel, ncolors=ncolors, vspace=args.vspace, is_scatter=is_scatter)
