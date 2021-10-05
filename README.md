<p align="right">
<img src="fig/polito_logo.png" alt="Logo" width="190" height="110">
</p>

# **Stochastic optimization models for a bike-sharing repositioning problem (BRP)**

This work is a project developed for the Operational Research course of ICT4SS at Politecnico di Torino.
The main purpose was to develop stochastic optimization models to solve a bike-sharing repositioning problem (BRP). 

## Abstract
The aim of this work is to study the problem of bike repositioning (BRP) that bike-sharing service providers have to face with. The purpose is to help the management of a 
fleet of bikes over a set of bike-stations, each with a fifixed capacity and stochastic demand. Being the problem very complex, the scope of the project is restricted by the following assumptions: 

1. Users pick up a bike at a station
   and drop them at a different one. Trips happen all at once.

2. Transshipment of bikes among stations
   is performed at the end of the day, so as to be ready for the next day trips.

The proposed solutions for these problems are: 
1. a two-stage stochastic optimization model based on GUROBI Exact Solver
2. a heuristic model based on the Progressive Hedging (PH) algorithm. 

The focus of our work was to find a better solution for bike sharing systems with bigger dimensions than the one shown in the work done by Maggioni et al. in https://www.sciencedirect.com/science/article/abs/pii/S0377221718311044?via%3Dihub.

The main strength of our PH heuristic method lays in the ability to find solutions with hypothetical real scenarios having a large number of bike stations considered in the computation. This, indeed, turned out to be not feasible with the used exact solver.

To see the full work please see the [documentation](https://github.com/AndreaMinardi/2021_ORTA_Bike_Sharing/blob/main/2021_ORTA_BikeSharing.pdf).

## Code Organization
The code is written in Python and organized in several folders. To launch the program, it is important to launch the [main.py](https://github.com/AndreaMinardi/2021_ORTA_Bike_Sharing/blob/main/main.py) file. 
From the terminal, run:

```shell
python3 main.py [-n <n_scenarios>] [-d <distribution_>]
```



1. [-n <n_scenarios>] represents the cardinality of the scenario tree you want to use (by default = 500)
2.  [-d <distribution_>] represents the distribution you want to use to build scenarios (choose among "norm" for Gaussian, "uni"  for Uniform, "expo" for exponential)

The main.py contains a menù through which you can:
1. solve the problem with the **Exact Solver** GUROBI;
2. solve the problem with the **Progressive Hedging** algorithm;
3. find the initial pair of values of penalty and alpha for PH;
4. compute the Value of Stochastic Solution (**VSS**) and the Expected Value of Perfect Information (**EVPI**);
5. perform the **in-sample stability**;
6. perform the **out-of-sample stability**;
7. find the **optimum number of scenarios**.




### Built With
* [Gurobi Optimization](https://www.gurobi.com/)
* [Python3](https://www.python.org/download/releases/3.0/)


### Developers
(c) 2021, [Francesco Conforte](https://github.com/FrancescoConforte), [Paolo De Santis](https://github.com/paolodesa), [Andrea Minardi](https://github.com/AndreaMinardi), [Jacopo Braccio](https://github.com/jacopobr).

