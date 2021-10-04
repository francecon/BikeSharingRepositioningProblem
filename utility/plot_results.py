# -*- coding: utf-8 -*-
# import numpy as np
from matplotlib import pyplot
# import matplotlib.pyplot as plt


def plot_comparison_hist(values, labels, colors, title, x_label, y_label):
    """Method to plot histograms used to verify in-sample and out-of-sample stability.

    Args:
        values (list): list of objective function values for exact solutions and heuristics solutions
        labels (list): list of labels of the plot
        colors (list): list of colors
        title (string): title of the plot
        x_label (string): label of the x-axis
        y_label (string): label of the y-axis
    """
    for i, item in enumerate(values):
        pyplot.hist(item, color=colors[i], bins=25, alpha=0.5, label=labels[i])
    pyplot.title(title)
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)
    pyplot.legend(loc='upper left')
    pyplot.savefig(f"./results/{title}.png")
    pyplot.close()
