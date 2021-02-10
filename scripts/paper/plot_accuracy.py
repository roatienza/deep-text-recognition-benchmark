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
        "RARE": { "Accuracy": 81.2, "Parameters": 10.9},
        "Rosetta": { "Accuracy": 77.8, "Parameters": 44.3},
        "GCRNN": { "Accuracy": 77.5, "Parameters": 4.8},
        "R2AM": { "Accuracy": 78.4, "Parameters": 2.9},
        "CRNN": { "Accuracy": 76.1, "Parameters": 8.5},
        "ViTSTR-Tiny(Ours)": { "Accuracy": 80.1, "Parameters": 5.4},
        "ViTSTR-Small(Ours)": { "Accuracy": 82.3, "Parameters": 21.5},
        }

acc_vs_param_env = [ (2.9, 5.4, 10.9, 21.5, 49.6), (78.4, 80.1, 81.2, 82.3, 83.5)]

acc_vs_time = {
        "TRBA": { "Accuracy": 83.5, "Speed (msec/image)": 22.8},
        "STAR-Net": { "Accuracy": 80.1, "Speed (msec/image)": 8.8},
        "RARE": { "Accuracy": 81.2, "Speed (msec/image)": 18.8},
        "Rosetta": { "Accuracy": 77.8, "Speed (msec/image)": 5.3},
        "GCRNN": { "Accuracy": 77.5, "Speed (msec/image)": 11.2},
        "R2AM": { "Accuracy": 78.4, "Speed (msec/image)": 22.9},
        "CRNN": { "Accuracy": 76.1, "Speed (msec/image)": 3.7},
        "ViTSTR-Tiny(Ours)": { "Accuracy": 80.1, "Speed (msec/image)": 9.3},
        "ViTSTR-Small(Ours)": { "Accuracy": 82.3, "Speed (msec/image)": 9.5},
        }

acc_vs_time_env = [ (3.7, 5.3, 8.8, 9.5, 22.8), (76.1, 77.8, 80.1, 82.3, 83.5)]

acc_vs_flops = {
        "TRBA": { "Accuracy": 83.5, "FLOPS": 10.9},
        "STAR-Net": { "Accuracy": 80.1, "FLOPS": 10.7},
        "RARE": { "Accuracy": 81.2, "FLOPS": 2.0},
        "Rosetta": { "Accuracy": 77.8, "FLOPS": 10.0},
        "GCRNN": { "Accuracy": 77.5, "FLOPS": 1.8},
        "R2AM": { "Accuracy": 78.4, "FLOPS": 2.0},
        "CRNN": { "Accuracy": 76.1, "FLOPS": 1.4},
        "ViTSTR-Tiny(Ours)": { "Accuracy": 80.1, "FLOPS": 2.1},
        "ViTSTR-Small(Ours)": { "Accuracy": 82.3, "FLOPS": 8.4},
        }

acc_vs_flops_env  = [ (1.4, 1.8, 2.0, 8.4, 10.9), (76.1, 77.5, 81.2, 82.3, 83.5)]

def plot_(data, envelope, title, ylabel="Accuracy", xlabel="Parameters"):
    plt.rc('font', size=14) 
    plt.rc('axes', titlesize=16)
    plt.rc('xtick', labelsize=14)

    fig, ax = plt.subplots()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    colors = sns.color_palette()[0:9]
    markers = ['^', 's', 'o', 'D', '*', 'P', 'x', 'd', 'v']

    #if "FLOPS"
    #x = np.arange(0,60,10)
    #ax.set_xticks(x)

    i = 0
    labels = []
    has_label = False
    for key, val in data.items():
        label = key
        acc = val["Accuracy"]
        if "Parameters" in xlabel:
            par = val["Parameters"]
        else:
            par = val[xlabel]
        color = colors[i]

        ax.scatter(par, acc, marker=markers[i], s=100, label=label, color=color)
        xytext = (8, -5)
        if par == 44.3:
            xytext = (-35, -18)
        elif par == 48.9:
            xytext = (-45, -20)
        elif par == 49.6:
            xytext = (-25, -20)
        elif par == 10.9 and "FLOPS" in xlabel:
            xytext = (-24, -25)
        elif par == 10.9:
            xytext = (10, -5)
        elif par == 10.7:
            xytext = (-45, -20)
        elif par == 21.5 or par == 2.0:
            xytext = (10, -10)
        elif par == 8.8:
            xytext = (-75, -5)
        elif par == 22.8 or par == 22.9 or par == 8.4 or (par == 10.9 and "FLOPS" in xlabel) or par == 10.0:
            xytext = (-35, -20)
        elif par == 9.5:
            xytext = (5, -10)
        ax.annotate(key, (par, acc), xycoords='data',
                    xytext=xytext, textcoords='offset points')

        i = i + 1

    xval = envelope[0]
    yval = envelope[1]
    plt.plot(xval, yval, linewidth=2, color='orange')
    
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
    ncolors = 9

    if args.data == "time":
        xlabel = "Speed (msec/image)"
        title = "Accuracy vs Msec per Image"
        data = acc_vs_time
        envelope = acc_vs_time_env
    elif args.data == "flops":
        xlabel = "FLOPS"
        title = "Accuracy vs FLOPS"
        data = acc_vs_flops
        envelope = acc_vs_flops_env
    else:
        xlabel = "Parameters (M)"
        title = "Accuracy vs Number of Parameters"
        data = acc_vs_param
        envelope = acc_vs_param_env

    plot_(data=data, envelope=envelope, title=title, ylabel=ylabel, xlabel=xlabel)
