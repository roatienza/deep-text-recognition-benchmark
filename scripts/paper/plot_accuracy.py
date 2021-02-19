'''
Plot accuracy
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import argparse

acc_vs_param = {
        "TRBA": { "Accuracy": 84.3, "Parameters": 49.6},
        "STAR-Net": { "Accuracy": 81.8, "Parameters": 48.9},
        "RARE": { "Accuracy": 81.9, "Parameters": 10.9},
        "Rosetta": { "Accuracy": 78.2, "Parameters": 44.3},
        "GCRNN": { "Accuracy": 78.3, "Parameters": 4.8},
        "R2AM": { "Accuracy": 78.4, "Parameters": 2.9},
        "CRNN": { "Accuracy": 76.7, "Parameters": 8.5},
        "ViTSTR-Tiny\n(Ours)": { "Accuracy": 80.0, "Parameters": 5.4},
        "ViTSTR-Small\n(Ours)": { "Accuracy": 82.5, "Parameters": 21.5},
        "ViTSTR-Tiny\n+Aug(Ours)": { "Accuracy": 82.3, "Parameters": 5.4},
        "ViTSTR-Small+Aug\n(Ours)": { "Accuracy": 83.7, "Parameters": 21.5},
        }

acc_vs_param_env = [[ (2.9, 5.4, 10.9, 21.5, 49.6), (78.4, 80.0, 81.9, 82.5, 84.3)],  
                    [ (2.9, 5.4, 21.5, 49.6), (78.4, 82.3, 83.7, 84.3)] ]

acc_vs_time = {
        "TRBA": { "Accuracy": 84.3, "Speed (msec/image)": 22.8},
        "STAR-Net": { "Accuracy": 81.8, "Speed (msec/image)": 8.8},
        "RARE": { "Accuracy": 81.9, "Speed (msec/image)": 18.8},
        "Rosetta": { "Accuracy": 78.2, "Speed (msec/image)": 5.3},
        "GCRNN": { "Accuracy": 78.3, "Speed (msec/image)": 11.2},
        "R2AM": { "Accuracy": 78.4, "Speed (msec/image)": 22.9},
        "CRNN": { "Accuracy": 76.7, "Speed (msec/image)": 3.7},
        "ViTSTR-Tiny(Ours)": { "Accuracy": 80.0, "Speed (msec/image)": 9.3},
        "ViTSTR-Small(Ours)": { "Accuracy": 82.5, "Speed (msec/image)": 9.5},
        "ViTSTR-Tiny\n+Aug(Ours)": { "Accuracy": 82.3, "Speed (msec/image)": 9.3},
        "ViTSTR-Small+Aug(Ours)": { "Accuracy": 83.7, "Speed (msec/image)": 9.5},
        }

acc_vs_time_env = [ 
                    [ (3.7, 5.3, 8.8, 9.5, 22.8), (76.7, 78.2, 81.8, 82.5, 84.3)],
                    [ (3.7, 5.3, 8.8, 9.3, 9.5, 22.8), (76.7, 78.2, 81.8, 82.5, 83.7, 84.3)],
                ]

acc_vs_flops = {
        "TRBA": { "Accuracy": 84.3, "FLOPS": 10.9},
        "STAR-Net": { "Accuracy": 81.8, "FLOPS": 10.7},
        "RARE": { "Accuracy": 81.9, "FLOPS": 2.0},
        "Rosetta": { "Accuracy": 78.2, "FLOPS": 10.0},
        "GCRNN": { "Accuracy": 78.3, "FLOPS": 1.8},
        "R2AM": { "Accuracy": 78.4, "FLOPS": 2.0},
        "CRNN": { "Accuracy": 76.7, "FLOPS": 1.4},
        "ViTSTR-Tiny(Ours)": { "Accuracy": 80.0, "FLOPS": 2.1},
        "ViTSTR-Small(Ours)": { "Accuracy": 82.5, "FLOPS": 8.4},
        "ViTSTR-Tiny\n+Aug(Ours)": { "Accuracy": 82.3, "FLOPS": 2.1},
        "ViTSTR-Small\n+Aug(Ours)": { "Accuracy": 83.7, "FLOPS": 8.4},
        }

acc_vs_flops_env  = [ 
        [(1.4, 1.8, 2.0, 8.4, 10.9), (76.7, 78.3, 81.9, 82.5, 84.3)],
        [(1.4, 1.8, 2.0, 2.1, 8.4, 10.9), (76.7, 78.3, 81.9, 82.3, 83.7, 84.3)]
        ]

def plot_(data, envelope, title, ylabel="Accuracy", xlabel="Parameters"):
    plt.rc('font', size=14) 
    plt.rc('axes', titlesize=16)
    plt.rc('xtick', labelsize=14)

    fig, ax = plt.subplots()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    colors = sns.color_palette() + sns.color_palette("tab10") # [0:11]
    markers = ['^', 's', 'o', 'D', '*', 'P', 'x', 'd', 'v', '>', 'H']

    #if "FLOPS"
    #x = np.arange(0,60,10)
    #ax.set_xticks(x)
    isparam = True if "Parameters" in xlabel else False
    isspeed = True if "Speed" in xlabel else False
    isflops = True if "FLOPS" in xlabel else False

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
        if isparam: 
            if "GCRNN" in label:
                xytext = (5, -15)
            elif "R2AM" in label:
                xytext = (5, 5)
            elif "Rosetta" in label:
                xytext = (-15, 15)
            elif "RARE" in label:
                xytext = (5, -15)
            elif "STAR" in label:
                xytext = (-45, -25)
            elif "TRBA" in label:
                xytext = (-25, -25)
            elif "Aug" in label and "Small" in label:
                xytext = (-50, 0)
            elif "Aug" in label and "Tiny" in label:
                xytext = (-30, 25)
            elif "Small" in label:
                xytext = (0, -35)
            elif "Tiny" in label:
                xytext = (5, -20)
        elif isspeed:
            if "STAR" in label:
                xytext = (5, -15)
            elif "R2AM" in label:
                xytext = (-25, 10)
            elif "TRBA" in label:
                xytext = (-35, -25)
            elif "Tiny" in label and "Aug" in label:
                xytext = (-90, -10)
            elif "Small" in label and "Aug" in label:
                xytext = (-40, 15)
        elif isflops:
            if "RARE" in label:
                xytext = (5, -15)
            elif "Rosetta" in label:
                xytext = (-35, -25)
            elif "R2AM" in label:
                xytext = (5, 5)
            elif "STAR" in label:
                xytext = (-45, -25)
            elif "TRBA" in label:
                xytext = (-55, 0)
            elif "GCRNN" in label:
                xytext = (0, -20)
            elif "Tiny" in label and "Aug" in label:
                xytext = (-30, 15)
            elif "Small" in label and "Aug" in label:
                xytext = (-90, 0)
            elif "Small" in label:
                xytext = (-90, -20)
        
        ax.annotate(key, (par, acc), xycoords='data',
                    xytext=xytext, textcoords='offset points')

        i = i + 1

    xval = envelope[0][0]
    yval = envelope[0][1]
    plt.plot(xval, yval, linewidth=2, color='orange')
    
    xval = envelope[1][0]
    yval = envelope[1][1]
    plt.plot(xval, yval, linewidth=2, color='teal')
    
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
