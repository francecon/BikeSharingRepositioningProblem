#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from simulator.instance import Instance
from simulator.tester import Tester
from solver.BikeSharing import BikeSharing
from heuristic.heuristics import ProgressiveHedging
from solver.sampler import Sampler
from utility.plot_results import plot_comparison_hist
from utility.plot_optimum_number_of_scenarios import plot_opt_num_scenarios
import csv
import sys
import getopt

np.random.seed(5)  

if __name__ == '__main__':
    log_name = "./logs/main.log"
    logging.basicConfig(
        filename=log_name,
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO, datefmt="%H:%M:%S",
        filemode='w'
    )

    fp = open("./etc/bike_share_settings.json", 'r')
    sim_setting = json.load(fp)
    fp.close()

    sam = Sampler()

    inst = Instance(sim_setting)
    dict_data = inst.get_data()


    # Reward generation
    n_scenarios = 500
    distribution = "norm"

    help_str = 'main.py -n <n_scenarios> -d <distribution>'
    try:
        opts, args = getopt.getopt(
            sys.argv[1:], "hn:d:", ["n_scenarios=", "distribution="])
    except getopt.GetoptError:
        print(help_str)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_str)
            sys.exit()
        elif opt in ("-n", "--n_scenarios"):
            n_scenarios = int(arg)
        elif opt in ("-d", "--distribution"):
            if (arg in ("norm", "uni", "expo")):
                distribution = arg
            else:
                print("Choose a distribution among norm, uni and expo")
                sys.exit()

    demand_matrix = sam.sample_stoch(
        inst,
        n_scenarios=n_scenarios,
        distribution=distribution
    )
    
    def solve_exact():
        """
        Here we compute the exact solution with GUROBI. We then save results in a csv file.
        It contains a table with columns: 
        1. Method (Exact),
        2. Objective function value in euro
        3. Computational Time
        4. First stage solution (number of bikes to put per stations at the beginning of each day)
        """
        print("EXACT METHOD")
        file_output = open(
            "./results/exact_method.csv",
            "w"
        )
        file_output.write("method, of, time, sol\n")
        prb = BikeSharing()
        of_exact, sol_exact, comp_time_exact, _ = prb.solve(
            inst,
            demand_matrix,
            n_scenarios,
        )
        file_output.write("{}, {}, {}, {}\n".format(
            "exact", of_exact, comp_time_exact, ' '.join(str(e) for e in sol_exact)
        ))
        print("To see results, open the file: /results/exact_method.csv")

    
    def solve_heu():
        """
        heuristic solution using the Progressive Hedging algorithm. We then save the solution in a csv file.
        It contains a table with columns: 
        1. Method (Heuristics),
        2. Objective function value in euro
        3. Computational Time
        4. Number of iterations
        5. First stage solution (number of bikes to put per stations at the beginning of each day)
        """
        print("HEURISTIC METHOD")
        file_output = open(
            "./results/heu_method.csv",
            "w"
        )
        file_output.write("method, of, time, iterations, sol\n")
        heu = ProgressiveHedging()
        of_heu, sol_heu, comp_time_heu, ite = heu.solve(
            inst,
            demand_matrix,
            n_scenarios
        )
        file_output.write("{}, {}, {}, {}, {}\n".format(
            "heu", of_heu, comp_time_heu, ite, ' '.join(str(e) for e in sol_heu)
        ))
        print("To see results, open the file: /results/heu_method.csv")

    # ########################################################
    # SEARCH FOR GOOD VALUES OF PENALTY AND ALPHA
    # ########################################################
    def search_penalty_alpha():
        """
        Method to find the initial pair of values of penalty and alpha for PH algorithm.
        """
        print("SEARCHING FOR GOOD VALUES OF PENALTY AND ALPHA")
        file_output = open(
            "./results/search_penalty_alpha.csv",
            "w"
        )
        file_output.write("method, of, time, rho, alpha, iterations, sol\n")
        heu = ProgressiveHedging()
        for r in np.arange(10, 110, 30):
            for alpha in [1.1, 10, 100]:
                print("TRYING WITH ALPHA=: ", alpha, "AND PENALTY=", r)
                of_heu, sol_heu, comp_time_heu, ite = heu.solve(
                    inst,
                    demand_matrix,
                    n_scenarios,
                    round(r, 1),
                    round(alpha, 1)
                )
                file_output.write("{}, {}, {}, {}, {}, {}, {}\n".format(
                    "heu", of_heu, comp_time_heu, r, alpha, ite, ' '.join(str(e) for e in sol_heu)
                ))

        file_output.close()
        print("To see results, open the file: /results/search_penalty_alpha.csv")

    
    def vss_evpi():
        """Method to compute the Value Of The Stochastic Solution (VSS) and Expected value of 
        perfect information (EVPI). Here the Recourse Problem (RP) and the Expected Value Problem (EVV)
         are solved to compute the VSS as: VSS = EVV-RP as well as the Wait and See Problem (WS) is 
         solved to compute the EVPI as EVPI = RP - WS.
        """
        
        # #########################################################
        # RECOURSE PROBLEM
        # #########################################################
        """
        Here we make a first stage decision and then we solve the second stage problems one per each scenario
        and average their solutions in order to compute the objective function value of the RECOURSE PROBLEM. 
        """
        demand_RP = sam.sample_stoch(
            inst,
            n_scenarios=n_scenarios,
            distribution=distribution
        )
        test = Tester()

        prb = BikeSharing()
        of_exact, sol_exact, comp_time_exact, _ = prb.solve(
            inst,
            demand_matrix,
            n_scenarios,
            verbose=True
        )

        ris_RP = test.solve_second_stages(
            inst,
            sol_exact,
            n_scenarios,
            demand_RP
        )
        
        #take the expected value over all the scenarios (mean because scenarios are assumed equiprobable)
        RP = np.mean(ris_RP) 
        
    
        # #########################################################
        # EXPECTED VALUE PROBLEM and the VALUE OF THE STOCHASTIC SOLUTION
        # #########################################################
        """
        The Scenarios are all blend together and only a scenario given by the average of them is considered.
        The resulting solution is clearly suboptimal but allows us to understand how 
        much we can gain from the fact that we consider the stochasticity with respect
        to not considering it at all and so, just considering the Expected Value of the demand.
        """
        # take the average scenario
        EV_demand_matrix = sam.sample_ev(
            inst,
            n_scenarios=n_scenarios,
            distribution=distribution
        )

        # Solve the Expected Value (EV) Problem and save the EV solution
        of_EV, sol_EV, comp_time_EV = prb.solve_EV(
            inst,
            EV_demand_matrix,
            verbose=True
        )

        # Sample new scenarios
        demand_EV = sam.sample_stoch(
            inst,
            n_scenarios=n_scenarios,
            distribution=distribution
        )

        # use the EV solution as the first stage solution 
        # for the stochastic program and compute the expected value of 
        # the objective function over several scenarios
        ris_EV = test.solve_second_stages(
            inst,
            sol_EV,
            n_scenarios,
            demand_EV
        )

        EEV = np.average(ris_EV)
        
        print("\nRecourse problem solution (RP)", RP)
        print("\nEV solution (EV): ", of_EV)
        print("\nExpected result of EV solution (EEV): ", EEV)
        print("\nValue of the Stochastic Solution (VSS = EEV-RP):", EEV-RP)


        # ##########################################################
        # WAIT AND SEE and the EXPECTED VALUE OF PERFECT INFORMATION
        # ##########################################################
        """
        Considering each of the scenarios separately and solving the first stage problems
        with full knowledge of the scenario is going to unfold. This is useful to understand
        what is the actual value of "knowing the future" and being able to adapt the first 
        stage variables to the possible demand. 
        """
        WS_demand = sam.sample_stoch(
            inst,
            n_scenarios=n_scenarios,
            distribution=distribution
        )
        ris2, WS_sol = test.solve_wait_and_see(
            inst,
            n_scenarios,
            WS_demand
        )
        print("\nWait and see solution (WS): ", WS_sol)
        print("\nExpected value of perfect information (EVPI = RP-WS): ", RP-WS_sol)

    # ##########################################################
    # IN SAMPLE STABILITY ANALYSIS
    # ##########################################################
    def test_in_sample(n_scenarios=500, n_rep=20):
        """
        Here we analyze the in sample stability for our scenario tree generation method.
        This requirement guarantees that whichever scenario tree we choose, the optimal
        value of the objective function reported by the model itself is (approximately) 
        the same. We evaluate different solutions on "n_rep" generated trees with cardinality
        "n_scenarios" and if the results are similar, we have in sample stability.
        """
        test = Tester()
        prb = BikeSharing()
        heu = ProgressiveHedging()
        print("IN SAMPLE STABILITY ANALYSIS")
        
        print("EXACT MODEL START...")
        in_samp_exact = test.in_sample_stability(prb, sam, inst, n_rep, n_scenarios, distribution)

        print("HEURISTIC MODEL START...")
        in_samp_heu = test.in_sample_stability(heu, sam, inst, n_rep, n_scenarios, distribution)

        plot_comparison_hist(
            [in_samp_exact, in_samp_heu],
            ["exact", "heuristic"],
            ['red', 'blue'], "In Sample Stability",
            "Objective Function value (€)", "Occurrences"
        )

        rows = zip(in_samp_exact, in_samp_heu)
        with open("./results/in_stability.csv", "w") as f:
            writer = csv.writer(f)
            f.write("in_samp_exact, in_samp_heu\n")
            for row in rows:
                writer.writerow(row)

    # ##########################################################
    # OUT OF SAMPLE STABILITY ANALYSIS
    # ##########################################################
    def test_out_sample(n_scenarios_first = 500, n_scenarios_second = 500, n_rep = 20):
        """
        The out-of-sample stability test investigates whether a scenario generation method,
        with the selected sample size, creates scenario trees that provide optimal solutions 
        that give approximately the same optimal value as when using the true probability distribution.
        """
        test = Tester()
        prb = BikeSharing()
        heu = ProgressiveHedging()
        print("OUT OF SAMPLE STABILITY ANALYSIS")
        
        print("EXACT MODEL START...")
        out_samp_exact = test.out_of_sample_stability(prb, sam, inst, n_rep, n_scenarios_first, n_scenarios_second)
        
        print("HEURISTIC MODEL START...")
        out_samp_heu = test.out_of_sample_stability(heu, sam, inst, n_rep, n_scenarios_first, n_scenarios_second)

        plot_comparison_hist(
            [out_samp_exact, out_samp_heu],
            ["exact", "heuristic"],
            ['red', 'blue'], "Out of Sample Stability",
            "Objective Function value (€)", "Occurrences"
        )

        rows = zip(out_samp_exact, out_samp_heu)
        with open("./results/out_stability.csv", "w") as f:
            writer = csv.writer(f)
            f.write("out_samp_exact, out_samp_heu\n")
            for row in rows:
                writer.writerow(row)
    # ##########################################################
    # FIND THE OPTIMUM NUMBER OF SCENARIOS
    # ##########################################################
    def optimum_num_scenarios():
        """
        Method to find the optimum number of scenarios. 
        The in-sample analysis is performed for an increasing cardinality of Scenario Trees 
        and for 3 different distributions: Exponential, Uniform and Normal.
        
        Results are then saved in a csv to plot easly them and see where the 
        objective function value is pretty constant.

        """
        test = Tester()
        prb = BikeSharing()

        n_rep = 15 #number of times we solve the problem
        obj_values_distr = []
        distributions = ["norm","uni","expo"]
        for distr in distributions:
            obj_values = []
            for i in range(100,1001,100):
                obj_values.append(np.mean(test.in_sample_stability(prb, sam, inst, n_rep, i,distr)))
            obj_values_distr.append(obj_values)
        with open("./results/optimum_num_scenarios.csv", "w") as f:
            writer = csv.writer(f)
            f.write("distribution,100,200,300,400,500,600,700,800,900,1000\n")
            for i, val in enumerate(obj_values_distr):
                writer.writerow([distributions[i]]+val)    
            f.close()

        plot_opt_num_scenarios()
        
        
    # ##########################################################
    # MENU'
    # ##########################################################    
    
    while True:
        try:
            option = input("What do you want to do? (Options: solve_exact, solve_heu, search_penalty_alpha, vss_evpi, test_in_sample, test_out_sample, opt_scenarios or exit)\n")
            if (option == "solve_exact"):
                solve_exact()
            elif (option == "solve_heu"):
                solve_heu()
            elif (option == "search_penalty_alpha"):
                search_penalty_alpha()
            elif (option == "vss_evpi"):
                vss_evpi()
                break
            elif (option == "test_in_sample"):
                inp = input("Choose the number of Scenario Trees you want to build\n")
                test_in_sample(n_rep=int(inp))
            elif (option == "test_out_sample"):
                inp = input("Choose the number of Scenario Trees you want to build\n")
                test_out_sample(n_rep=int(inp))
            elif (option == "opt_scenarios"):
                optimum_num_scenarios()
            elif (option == "exit"):
                break
            else:
                print("Unsupported operation, please check the command")
        except KeyboardInterrupt:
            break
        