# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 11:08:40 2021

@author: Francesco Conforte
"""
import pandas as pd
import matplotlib.pyplot as plt


def plot_opt_num_scenarios():
    """Method to plot the graph used to choose which is the optimum cardinality for a Scenario Tree
    """
    file = pd.read_csv("./results/optimum_num_scenarios.csv",index_col=0,header=0)
    distributions = ["norm","uni","expo"]
    
    markers = ["s","o","x"]
    
    for i,row in enumerate(file.values):
        plt.plot(range(100,1001,100),row,marker=markers[i],linewidth=2.5)
    plt.legend(distributions,fontsize=12)
    plt.xlabel("Cardinality of the scenario tree",fontsize=12)
    plt.ylabel("Objective function value (€)",fontsize=12)
    plt.xticks(range(100,1001,100),fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    plt.savefig(f"./results/Optimum number of scenarios.png")



