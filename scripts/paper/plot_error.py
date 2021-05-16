'''
Plot accuracy
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import argparse

acc = {
        "CRNN": { "delta" : [0.71,1.30,1.04,1.06,0.94,0.59,0.44,0.14]  , "std": [0.03, 0.26, 0.35, 0.19, 0.05, 0.15, 0.06, 0.02]},
        "R2AM": { "delta" : [1.17, 1.15, 1.48, 1.34, 1.40, 0.99, 0.78, 0.55]  , "std": [0.29, 0.06, 0.19, 0.06, 0.19, 0.02, 0.26, 0.08]},
        "GCRNN": { "delta" : [0.71, 1.30, 1.04, 1.06, 0.94, 0.59, 0.44, 0.14]  , "std": [0.03, 0.26, 0.35, 0.19, 0.05, 0.15, 0.06, 0.02]},
        "Rosetta": { "delta" : [0.99, 1.25, 2.10, 1.87, 1.57, 1.00, 1.01, 0.82]  , "std": [0.02, 0.09, 0.30, 0.11, 0.43, 0.11, 0.04, 0.04]},
        "RARE": { "delta" : [0.64, 1.01, 1.08, 1.35, 1.07, 0.76, 0.67, -0.12], "std": [0.08, 0.14, 0.10, 0.26, 0.13, 0.30, 0.05, 0.04]},
        "TRBA": { "delta" : [0.44, 0.74, 1.06, 0.79, 0.69, 0.59, 0.66, 0.10]  , "std": [0.18, 0.26, 0.15, 0.28, 0.46, 0.28, 0.49, 0.40]},
        }

def plot_(data, title="RandAug Grid Search", ylabel="Accuracy relative to baseline", xlabel="Number of augmentations"):
    plt.rc('font', size=14) 
    plt.rc('axes', titlesize=16)
    plt.rc('xtick', labelsize=14)

    fig, ax = plt.subplots()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    colors = sns.color_palette() + sns.color_palette("tab10") # [0:11]
    markers = ['^', 's', 'o', 'D', '*', 'P', 'x', 'd', 'v', '>', 'H']

    i = 0
    labels = []
    has_label = False
    xval =  [i for i in range(1,9)]
    for key, val in data.items():
        label = key
        delta = val["delta"]
        std = val["std"]
        color = colors[i]

        ax.errorbar(xval, delta, std, marker=markers[i], label=label, color=color, capsize=5)
        ax.legend()
        #xytext = (8, -5)
        #if "RARE" in label:
        #    xytext = (5, -15)
        
        #ax.annotate(key, (par, acc), xycoords='data',
        #            xytext=xytext, textcoords='offset points')

        i = i + 1

    #plt.plot(xval, yval, linewidth=2, color='teal')
    
    title = title.replace(" ", "_")
    title = title.replace("%", "")
    plt.savefig(title + ".png")
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wordformer results')
    parser.add_argument('--data',
                        default=None,
                        help='Data to plot')
    args = parser.parse_args()
    ncolors = 9

    plot_(data=acc)
