# -*- coding: utf-8 -*-
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB


class Tester():
    """Class containing different useful methods:
        1. solve_second_stage() to solve the second stage problem
        2. solve_wait_and_see() to solve the WS problem used to compute the EVPI
        3. in_sample_stability() to analyze the in-sample stability of the scenario generation method
        4. out_of_sample_stability() to analyze the out-of-sample stability of the scenario generation method
    """
    def __init__(self):
        pass

    def solve_second_stages(self, 
        inst, sol, n_scenarios, demand_matrix
    ):
        ans = []
        dict_data = inst.get_data()
        obj_fs = 0
        n_stations = inst.n_stations
        stations = range(n_stations)
        for i in stations:
            obj_fs += dict_data["procurement_cost"] * sol[i]
        
        
        for s in range(n_scenarios):
            problem_name = "SecondStagePrb"
            model = gp.Model(problem_name)
            
            ### Variables
            
            I_plus = model.addVars(
                n_stations,
                lb=0,
                vtype=GRB.INTEGER,
                name='I+'
            )

            I_minus = model.addVars(
                n_stations, n_stations,
                lb=0,
                vtype=GRB.INTEGER,
                name='I-'
            )

            O_plus = model.addVars(
                n_stations,
                lb=0,
                vtype=GRB.INTEGER,
                name='O+'
            )

            O_minus = model.addVars(
                n_stations,
                lb=0,
                vtype=GRB.INTEGER,
                name='O-'
            )

            T_plus = model.addVars(
                n_stations,
                lb=0,
                vtype=GRB.INTEGER,
                name='T+'
            )

            T_minus = model.addVars(
                n_stations,
                lb=0,
                vtype=GRB.INTEGER,
                name='T-'
            )


            tau = model.addVars(
                n_stations, n_stations,
                lb=0,
                vtype=GRB.INTEGER,
                name='tau'
            )

            beta = model.addVars(
                n_stations, n_stations,
                lb=0,
                vtype=GRB.INTEGER,
                name='beta'
            )

            rho = model.addVars(
                n_stations, n_stations,
                lb=0,
                vtype=GRB.INTEGER,
                name='rho'
            )

            
            ### Objective Function
        
            obj_funct = gp.quicksum(
                (
                    dict_data['stock_out_cost'][i]*gp.quicksum(I_minus[i, j] for j in stations) +
                    dict_data["time_waste_cost"][i]*O_minus[i] +
                    gp.quicksum(dict_data['trans_ship_cost'][i][j]*tau[i, j]  for j in stations)
                ) for i in stations
            )

            model.setObjective(obj_funct, GRB.MINIMIZE)
            
            ### Costraints
            
            for i in stations:
                for j in stations:
                    model.addConstr(
                        beta[i, j] == demand_matrix[i, j, s] - I_minus[i, j],
                        f"rented_bikes_number"
                    )
                
            for i in stations:
                model.addConstr(
                    I_plus[i] - gp.quicksum(I_minus[i,j] for j in stations) == sol[i] - gp.quicksum(demand_matrix[i, j, s] for j in stations),
                    f"surplus_shortage_balance"
                )

            for i in stations:
                model.addConstr(
                    O_plus[i] - O_minus[i] == dict_data['station_cap'][i] - sol[i] + gp.quicksum(beta[i,j] for j in stations) - gp.quicksum(beta[j, i] for j in stations),
                    f"residual_overflow_balance"
                )

            for i in stations:
                model.addConstr(
                    gp.quicksum(rho[i,j] for j in stations) == O_minus[i],
                    f"redir_bikes_eq_overflow"
                )

            for i in stations:
                model.addConstr(
                    gp.quicksum(rho[j,i] for j in stations) <= O_plus[i],
                    f"redir_bikes_not_resid_cap"
                )

            for i in stations:
                model.addConstr(
                    T_plus[i] - T_minus[i] == dict_data['station_cap'][i] - O_plus[i] + gp.quicksum(rho[j,i] for j in stations) - sol[i],
                    f"exceed_failure_balance"
                )

            for i in stations:
                model.addConstr(
                    gp.quicksum(tau[i,j] for j in stations) == T_plus[i],
                    f"tranship_equal_excess"
                )

            for i in stations:
                model.addConstr(
                    gp.quicksum(tau[j,i] for j in stations) <= T_minus[i],
                    f"tranship_equal_failure"
                )


            model.update()
            model.setParam('OutputFlag', 0)
            model.setParam('LogFile', './logs/gurobi.log')
            model.optimize()
            ans.append(obj_fs + model.getObjective().getValue())

        return ans


    def solve_wait_and_see(self, 
        inst, n_scenarios, demand_matrix
    ):
        ans = []
        dict_data = inst.get_data()
        n_stations = inst.n_stations
        stations = range(n_stations)
        
        
        for s in range(n_scenarios):
            problem_name = "Wait_and_See_BikeShare"
            model = gp.Model(problem_name)
            
            ### Variables
            X = model.addVars(
                n_stations,
                lb=0,
                vtype=GRB.INTEGER,
                name='X'
            )
            
            I_plus = model.addVars(
                n_stations,
                lb=0,
                vtype=GRB.INTEGER,
                name='I+'
            )

            I_minus = model.addVars(
                n_stations, n_stations,
                lb=0,
                vtype=GRB.INTEGER,
                name='I-'
            )

            O_plus = model.addVars(
                n_stations,
                lb=0,
                vtype=GRB.INTEGER,
                name='O+'
            )

            O_minus = model.addVars(
                n_stations,
                lb=0,
                vtype=GRB.INTEGER,
                name='O-'
            )

            T_plus = model.addVars(
                n_stations,
                lb=0,
                vtype=GRB.INTEGER,
                name='T+'
            )

            T_minus = model.addVars(
                n_stations,
                lb=0,
                vtype=GRB.INTEGER,
                name='T-'
            )


            tau = model.addVars(
                n_stations, n_stations,
                lb=0,
                vtype=GRB.INTEGER,
                name='tau'
            )

            beta = model.addVars(
                n_stations, n_stations,
                lb=0,
                vtype=GRB.INTEGER,
                name='beta'
            )

            rho = model.addVars(
                n_stations, n_stations,
                lb=0,
                vtype=GRB.INTEGER,
                name='rho'
            )
            ### Objective Function
            obj_funct = dict_data["procurement_cost"] * gp.quicksum(X[i] for i in stations)
            
            obj_funct += gp.quicksum(
                (
                    dict_data['stock_out_cost'][i]*gp.quicksum(I_minus[i, j] for j in stations) +
                    dict_data["time_waste_cost"][i]*O_minus[i] +
                    gp.quicksum(dict_data['trans_ship_cost'][i][j]*tau[i, j]  for j in stations)
                ) for i in stations
            )

            model.setObjective(obj_funct, GRB.MINIMIZE)
            
            ### Costraints
            
            for i in stations:
                model.addConstr(
                    X[i] <= dict_data['station_cap'][i],
                    f"station_bike_limit"
                )

            for i in stations:
                for j in stations:
                    model.addConstr(
                        beta[i, j] == demand_matrix[i, j, s] - I_minus[i, j],
                        f"rented_bikes_number"
                    )
                
            for i in stations:
                model.addConstr(
                    I_plus[i] - gp.quicksum(I_minus[i,j] for j in stations) == X[i] - gp.quicksum(demand_matrix[i, j, s] for j in stations),
                    f"surplus_shortage_balance"
                )

            for i in stations:
                model.addConstr(
                    O_plus[i] - O_minus[i] == dict_data['station_cap'][i] - X[i] + gp.quicksum(beta[i,j] for j in stations) - gp.quicksum(beta[j, i] for j in stations),
                    f"residual_overflow_balance"
                )

            for i in stations:
                model.addConstr(
                    gp.quicksum(rho[i,j] for j in stations) == O_minus[i],
                    f"redir_bikes_eq_overflow"
                )

            for i in stations:
                model.addConstr(
                    gp.quicksum(rho[j,i] for j in stations) <= O_plus[i],
                    f"redir_bikes_not_resid_cap"
                )

            for i in stations:
                model.addConstr(
                    T_plus[i] - T_minus[i] == dict_data['station_cap'][i] - O_plus[i] + gp.quicksum(rho[j,i] for j in stations) - X[i],
                    f"exceed_failure_balance"
                )

            for i in stations:
                model.addConstr(
                    gp.quicksum(tau[i,j] for j in stations) == T_plus[i],
                    f"tranship_equal_excess"
                )

            for i in stations:
                model.addConstr(
                    gp.quicksum(tau[j,i] for j in stations) <= T_minus[i],
                    f"tranship_equal_failure"
                )


            model.update()
            model.setParam('OutputFlag', 0)
            model.setParam('LogFile', './logs/gurobi.log')
            model.optimize()
            ans.append(model.getObjective().getValue())

        WS = np.average(ans)
        return ans, WS



    def in_sample_stability(self, problem, sampler, instance, n_repetitions, n_scenarios_sol, distribution):
        ans = [0] * n_repetitions
        for i in range(n_repetitions):
            print("Scenario Tree: ", i)
            a = time.time()
            reward = sampler.sample_stoch(
                instance,
                n_scenarios=n_scenarios_sol,
                distribution=distribution
            )
            of, sol, comp_time, ite = problem.solve(
                instance,
                reward,
                n_scenarios_sol
            )
            b = time.time()
            print("Time spent:", b-a)
            ans[i] = of
        return ans
    
    def out_of_sample_stability(self, problem, sampler, instance, n_repetitions, n_scenarios_sol, n_scenarios_out):
        ans = [0] * n_repetitions
        for i in range(n_repetitions):
            print("Scenario Tree: ", i)
            a = time.time()
            reward= sampler.sample_stoch(
                instance,
                n_scenarios=n_scenarios_sol
            )
            of, sol, comp_time, ite = problem.solve(
                instance,
                reward,
                n_scenarios_sol
            )
            reward_out = sampler.sample_stoch(
                instance,
                n_scenarios=n_scenarios_out
            )
            profits = self.solve_second_stages(
                instance, sol,
                n_scenarios_out, reward_out
            )
            b = time.time()
            print("Time spent:", b-a)
            ans[i]=np.mean(profits)
            
        return ans
