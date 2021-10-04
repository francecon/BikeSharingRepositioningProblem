# -*- coding: utf-8 -*-
import numpy as np
import utility.OD_matrix_gen as gen
import utility.OD_matrix_realistic as gen_real

class Sampler:
    """Class representing the sampler of scenarios.
    It has two methods:
        1. sample_ev() to sample the scenario to solve the EV problem
        2. sample_stoch() to sample "n_scenarios" 
    """
    def __init__(self):
        pass

    def sample_ev(self, instance, n_scenarios, distribution):
        demand = self.sample_stoch(instance, n_scenarios, distribution)
        return np.around(np.average(demand, axis=2))

    def sample_stoch(self, instance, n_scenarios, distribution):
        generator = gen.Generator(n_scenarios, instance.n_stations, distribution)
        return generator.scenario_res
