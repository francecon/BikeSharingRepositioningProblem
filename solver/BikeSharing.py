# -*- coding: utf-8 -*-
import time
import logging
import gurobipy as gp
from gurobipy import GRB
import numpy as np


class BikeSharing():
    """Class representing the Exact Solver GUROBI.
    It has methods:
       1. solve() to solve the deterministic problem optimally
       2. solve_EV() to solve the Expected Value problem
    """
    def __init__(self):
        pass

    def solve(
        self, instance, demand_matrix, n_scenarios, time_limit=None,
        gap=None, verbose=False
    ):
        dict_data = instance.get_data()
        n_stations = dict_data['n_stations']
        stations = range(n_stations)
        
        scenarios = range(n_scenarios)

        problem_name = "Bike_Sharing"
        logging.info("{}".format(problem_name))
        # logging.info(f"{problem_name}")

        model = gp.Model(problem_name)
        X = model.addVars(
            n_stations,
            lb=0,
            vtype=GRB.INTEGER,
            name='X'   
        )
        I_plus = model.addVars(
            n_stations, n_scenarios,
            lb=0,
            vtype=GRB.INTEGER,
            name='I+'
        )

        I_minus = model.addVars(
            n_stations, n_stations, n_scenarios,
            lb=0,
            vtype=GRB.INTEGER,
            name='I-'
        )

        O_plus = model.addVars(
            n_stations, n_scenarios,
            lb=0,
            vtype=GRB.INTEGER,
            name='O+'
        )

        O_minus = model.addVars(
            n_stations, n_scenarios,
            lb=0,
            vtype=GRB.INTEGER,
            name='O-'
        )

        T_plus = model.addVars(
            n_stations, n_scenarios,
            lb=0,
            vtype=GRB.INTEGER,
            name='T+'
        )

        T_minus = model.addVars(
            n_stations, n_scenarios,
            lb=0,
            vtype=GRB.INTEGER,
            name='T-'
        )


        tau = model.addVars(
            n_stations, n_stations, n_scenarios,
            lb=0,
            vtype=GRB.INTEGER,
            name='tau'
        )

        beta = model.addVars(
            n_stations, n_stations, n_scenarios,
            lb=0,
            vtype=GRB.INTEGER,
            name='beta'
        )

        rho = model.addVars(
            n_stations, n_stations, n_scenarios,
            lb=0,
            vtype=GRB.INTEGER,
            name='rho'
        )

        
          
        obj_funct = dict_data["procurement_cost"] * gp.quicksum(X[i] for i in stations)

        obj_funct += gp.quicksum(
            (
                dict_data['stock_out_cost'][i]*gp.quicksum(I_minus[i, j, s] for j in stations) +
                dict_data["time_waste_cost"][i]*O_minus[i, s] +
                gp.quicksum(dict_data['trans_ship_cost'][i][j]*tau[i, j, s]  for j in stations)
            )
            for i in stations for s in scenarios
        )/(n_scenarios + 0.0)
        model.setObjective(obj_funct, GRB.MINIMIZE)       


        for i in stations:
            model.addConstr(
                X[i] <= dict_data['station_cap'][i],
                f"station_bike_limit"
            )
            
        for i in stations:
            for j in stations:
                for s in scenarios:
                    model.addConstr(
                        beta[i, j, s] == demand_matrix[i, j, s] - I_minus[i, j, s],
                        f"rented_bikes_number"
                    )
                    
        for i in stations:
            for s in scenarios:
                model.addConstr(
                    I_plus[i, s] - gp.quicksum(I_minus[i,j,s] for j in stations) == X[i] - gp.quicksum(demand_matrix[i, j, s] for j in stations),
                    f"surplus_shortage_balance"
                )
                
        for i in stations:
            for s in scenarios:
                model.addConstr(
                    O_plus[i, s] - O_minus[i, s] == dict_data['station_cap'][i] - X[i] + gp.quicksum(beta[i,j,s] for j in stations) - gp.quicksum(beta[j, i ,s] for j in stations),
                    f"residual_overflow_balance"
                )
                
        for i in stations:
            for s in scenarios:
                model.addConstr(
                    gp.quicksum(rho[i,j,s] for j in stations) == O_minus[i, s],
                    f"redir_bikes_eq_overflow"
                )
                
        for i in stations:
            for s in scenarios:
                model.addConstr(
                    gp.quicksum(rho[j,i,s] for j in stations) <= O_plus[i, s],
                    f"redir_bikes_not_resid_cap"
                )
                
        for i in stations:
            for s in scenarios:
                model.addConstr(
                    T_plus[i, s] - T_minus[i, s] == dict_data['station_cap'][i] - O_plus[i, s] + gp.quicksum(rho[j,i,s] for j in stations) - X[i],
                    f"exceed_failure_balance"
                )
                
        for i in stations:
            for s in scenarios:
                model.addConstr(
                    gp.quicksum(tau[i,j,s] for j in stations) == T_plus[i, s],
                    f"tranship_equal_excess"
                )
                
        for i in stations:
            for s in scenarios:
                model.addConstr(
                    gp.quicksum(tau[j,i,s] for j in stations) <= T_minus[i, s],
                    f"tranship_equal_failure"
                )
                    
        model.update()
        if gap:
            model.setParam('MIPgap', gap)
        if time_limit:
            model.setParam(GRB.Param.TimeLimit, time_limit)
        if verbose:
            model.setParam('OutputFlag', 1)
        else:
            model.setParam('OutputFlag', 0)
        model.setParam('LogFile', './logs/gurobi.log')
        # model.write("./logs/model.lp")

        start = time.time()
        model.optimize()
        end = time.time()
        comp_time = end - start
        
        sol = [0] * dict_data['n_stations']
        of = -1
        if model.status == GRB.Status.OPTIMAL:
            for i in stations:
                grb_var = model.getVarByName(
                    f"X[{i}]"
                )
                sol[i] = grb_var.X
            of = model.getObjective().getValue()
        
        
        
        return of, sol, comp_time, None



    def solve_EV(
        self, instance, demand_matrix, time_limit=None,
        gap=None, verbose=False
    ):
        dict_data = instance.get_data()
        n_stations = dict_data['n_stations']
        stations = range(n_stations)
        
        problem_name = "Bike_Sharing_EV"
        logging.info("{}".format(problem_name))
        # logging.info(f"{problem_name}")

        model = gp.Model(problem_name)
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

                
        obj_funct = dict_data["procurement_cost"] * gp.quicksum(X[i] for i in stations)

        obj_funct += gp.quicksum(
            (
                dict_data['stock_out_cost'][i]*gp.quicksum(I_minus[i, j] for j in stations) +
                dict_data["time_waste_cost"][i]*O_minus[i] +
                gp.quicksum(dict_data['trans_ship_cost'][i][j]*tau[i, j]  for j in stations)
            )
            for i in stations)
        model.setObjective(obj_funct, GRB.MINIMIZE)


        for i in stations:
            model.addConstr(
                X[i] <= dict_data['station_cap'][i],
                f"station_bike_limit"
            )
            
        for i in stations:
            for j in stations:
                model.addConstr(
                    beta[i, j] == demand_matrix[i, j] - I_minus[i, j],
                    f"rented_bikes_number"
                )
                    
        for i in stations:
            model.addConstr(
                I_plus[i] - gp.quicksum(I_minus[i,j] for j in stations) == X[i] - gp.quicksum(demand_matrix[i, j] for j in stations),
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
        if gap:
            model.setParam('MIPgap', gap)
        if time_limit:
            model.setParam(GRB.Param.TimeLimit, time_limit)
        if verbose:
            model.setParam('OutputFlag', 1)
        else:
            model.setParam('OutputFlag', 0)
        model.setParam('LogFile', './logs/gurobi.log')
        # model.write("./logs/model.lp")

        start = time.time()
        model.optimize()
        end = time.time()
        comp_time = end - start
        
        sol = [0] * dict_data['n_stations']
        of = -1
        if model.status == GRB.Status.OPTIMAL:
            for i in stations:
                grb_var = model.getVarByName(
                    f"X[{i}]"
                )
                sol[i] = grb_var.X
            of = model.getObjective().getValue()
        return of, sol, comp_time