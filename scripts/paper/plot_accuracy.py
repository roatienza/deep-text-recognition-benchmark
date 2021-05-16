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
        "ViTSTR-Tiny\n(Ours)": { "Accuracy": 80.3, "Parameters": 5.4},
        "ViTSTR-Small\n(Ours)": { "Accuracy": 82.6, "Parameters": 21.5},
        "ViTSTR-\nTiny+Aug\n(Ours)": { "Accuracy": 82.1, "Parameters": 5.4},
        "ViTSTR-Small\n+Aug(Ours)": { "Accuracy": 84.2, "Parameters": 21.5},
        "ViTSTR-Base\n(Ours)": { "Accuracy": 83.7, "Parameters": 85.8},
        "ViTSTR-Base+Aug(Ours)": { "Accuracy": 85.2, "Parameters": 85.8},
        }

acc_vs_param_env = [[ (2.9, 5.4, 10.9, 21.5, 49.6, 85.8), (78.4, 80.3, 81.9, 82.6, 84.3, 83.7)],  
                    [ (2.9, 5.4, 21.5, 85.8), (78.4, 82.1, 84.2, 85.2)] ]

acc_vs_time = {
        "TRBA": { "Accuracy": 84.3, "Speed (msec/image)": 22.8},
        "STAR-Net": { "Accuracy": 81.8, "Speed (msec/image)": 8.8},
        "RARE": { "Accuracy": 81.9, "Speed (msec/image)": 18.8},
        "Rosetta": { "Accuracy": 78.2, "Speed (msec/image)": 5.3},
        "GCRNN": { "Accuracy": 78.3, "Speed (msec/image)": 11.2},
        "R2AM": { "Accuracy": 78.4, "Speed (msec/image)": 22.9},
        "CRNN": { "Accuracy": 76.7, "Speed (msec/image)": 3.7},
        "ViTSTR-Tiny(Ours)": { "Accuracy": 80.3, "Speed (msec/image)": 9.3},
        "ViTSTR-Small(Ours)": { "Accuracy": 82.6, "Speed (msec/image)": 9.5},
        "ViTSTR-Tiny+Aug": { "Accuracy": 82.1, "Speed (msec/image)": 9.3},
        "ViTSTR-Small\n+Aug(Ours)": { "Accuracy": 84.2, "Speed (msec/image)": 9.5},
        "ViTSTR-Base(Ours)": { "Accuracy": 83.7, "Speed (msec/image)": 9.8},
        "ViTSTR-Base+Aug(Ours)": { "Accuracy": 85.2, "Speed (msec/image)": 9.8},
        }

acc_vs_time_env = [ 
                    [ (3.7, 9.8, 22.8), (76.7, 83.7, 84.3)],
                    #[ (3.7, 5.3, 8.8, 9.8, 22.8), (76.7, 78.2, 81.8, 83.7, 84.3)],
                    [ (3.7, 9.8, 22.8), (76.7, 85.2, 84.3)],
                    #[ (3.7, 5.3, 8.8, 9.5, 9.8, 22.8), (76.7, 78.2, 81.8, 84.2, 85.2, 84.3)],
                ]

acc_vs_flops = {
        "TRBA": { "Accuracy": 84.3, "GFLOPS": 10.9},
        "STAR-Net": { "Accuracy": 81.8, "GFLOPS": 10.7},
        "RARE": { "Accuracy": 81.9, "GFLOPS": 2.0},
        "Rosetta": { "Accuracy": 78.2, "GFLOPS": 10.0},
        "GCRNN": { "Accuracy": 78.3, "GFLOPS": 1.8},
        "R2AM": { "Accuracy": 78.4, "GFLOPS": 2.0},
        "CRNN": { "Accuracy": 76.7, "GFLOPS": 1.4},
        "ViTSTR-Tiny(Ours)": { "Accuracy": 80.3, "GFLOPS": 1.3},
        "ViTSTR-Small(Ours)": { "Accuracy": 82.6, "GFLOPS": 4.6},
        "ViTSTR\n-Tiny\n+Aug\n(Ours)": { "Accuracy": 82.1, "GFLOPS": 1.3},
        "ViTSTR-Small\n+Aug(Ours)": { "Accuracy": 84.2, "GFLOPS": 4.6},
        "ViTSTR-Base\n(Ours)": { "Accuracy": 83.7, "GFLOPS": 17.6},
        "ViTSTR-Base+Aug(Ours)": { "Accuracy": 85.2, "GFLOPS": 17.6},
        }

acc_vs_flops_env  = [ 
        [(1.3, 2.0, 4.6, 10.9, 17.6), (80.3, 81.9, 82.6, 84.3, 83.7)],
        [(1.3, 4.6, 17.6), (82.1, 84.2, 85.2)]
        ]

def plot_(data, envelope, title, ylabel="Accuracy", xlabel="Parameters"):
    plt.rc('font', size=14) 
    plt.rc('axes', titlesize=16)
    plt.rc('xtick', labelsize=14)

    fig, ax = plt.subplots()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    colors = sns.color_palette() + sns.color_palette("tab10") # [0:11]
    markers = ['^', 's', 'o', 'D', '*', 'P', 'x', 'd', 'v', '>', 'H', '1', '2']

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
                xytext = (5, 5)
            elif "RARE" in label:
                xytext = (5, -15)
            elif "STAR" in label:
                xytext = (10, -10)
            elif "TRBA" in label:
                xytext = (-25, -25)
            elif "Aug" in label and "Small" in label:
                xytext = (-30, 10)
            elif "Aug" in label and "Tiny" in label:
                xytext = (-25, 15)
            elif "Aug" in label and "Base" in label:
                xytext = (-180, 0)
            elif "Small" in label:
                xytext = (10, -25)
            elif "Tiny" in label:
                xytext = (10, -20)
            elif "Base" in label:
                xytext = (-75, -30)
        elif isspeed:
            if "STAR" in label:
                xytext = (-10, -20)
            elif "R2AM" in label:
                xytext = (-25, 10)
            elif "TRBA" in label:
                xytext = (-35, -25)
            elif "Tiny" in label and "Aug" in label:
                xytext = (5, -5)
            elif "Small" in label and "Aug" in label:
                xytext = (-100, -10)
            elif "Base" in label and "Aug" in label:
                xytext = (5, 0)
            elif "Base" in label:
                xytext = (5, -12)
            elif "Small" in label:
                xytext = (5, 0)
        elif isflops:
            if "RARE" in label:
                xytext = (5, -15)
            elif "Rosetta" in label:
                xytext = (-35, -25)
            elif "R2AM" in label:
                xytext = (5, 5)
            elif "STAR" in label:
                xytext = (5, -15)
            elif "TRBA" in label:
                xytext = (-15, -20)
            elif "GCRNN" in label:
                xytext = (0, -20)
            elif "Tiny" in label and "Aug" in label:
                xytext = (-15, 10)
            elif "Small" in label and "Aug" in label:
                xytext = (-35, 10)
            elif "Small" in label:
                xytext = (5, -10)
            elif "Base" in label and "Aug" in label:
                xytext = (-180, 0)
            elif "Base" in label:
                xytext = (-80, -30)
        
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
    parser = argparse.ArgumentParser(description='ViTSTR')
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
        xlabel = "GFLOPS"
        title = "Accuracy vs GFLOPS"
        data = acc_vs_flops
        envelope = acc_vs_flops_env
    else:
        xlabel = "Parameters (M)"
        title = "Accuracy vs Number of Parameters"
        data = acc_vs_param
        envelope = acc_vs_param_env

    plot_(data=data, envelope=envelope, title=title, ylabel=ylabel, xlabel=xlabel)
