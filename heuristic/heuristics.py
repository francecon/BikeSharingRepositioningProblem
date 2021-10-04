# -*- coding: utf-8 -*-
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB

class ProgressiveHedging():
    """Class representing the PH heuristics method.
    It has two methods:
        1. DEP_solver() to solve the deterministic problem with Augmented Relaxation
        2. solve() is the actual Progressive Hedging algorithm
    """
    def __init__(self):
        pass

    def DEP_solver(self, dict_data, scenario_input, TGS, lambd=0, pen_rho=0, iteration=0):
        problem_name = "DEP"
        model = gp.Model(problem_name)
        n_stations = dict_data['n_stations']
        stations = range(n_stations)
                
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
        model.update()
        sol = [X[i] for i in stations]


        ## Objective Function
        obj_funct = dict_data["procurement_cost"] * gp.quicksum(X[i] for i in stations)
        
        obj_funct += gp.quicksum(
            (
                dict_data['stock_out_cost'][i]*gp.quicksum(I_minus[i, j] for j in stations) +
                dict_data["time_waste_cost"][i]*O_minus[i] +
                gp.quicksum(dict_data['trans_ship_cost'][i][j]*tau[i, j]  for j in stations)
            ) for i in stations
        )
        if iteration!= 0:
            relax = np.dot(lambd.T,(np.array(sol)-TGS))
            penalty = (pen_rho/2)*(np.dot((np.array(sol)-TGS),(np.array(sol)-TGS).T))
            
            obj_funct += relax
        
            obj_funct += penalty


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
                    beta[i, j] == scenario_input[i, j] - I_minus[i, j],
                    f"rented_bikes_number"
                )
            
        for i in stations:
            model.addConstr(
                I_plus[i] - gp.quicksum(I_minus[i,j] for j in stations) == X[i] - gp.quicksum(scenario_input[i, j]for j in stations),
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
        model.optimize()
        sol = [0] * dict_data['n_stations']
        of = -1
        if model.status == GRB.Status.OPTIMAL:
            for i in stations:
                grb_var = model.getVarByName(
                    f"X[{i}]"
                )
                sol[i] = grb_var.X
            of = model.getObjective().getValue()
        return of, np.array(sol)
    

    def solve(
        self, instance, scenarios, n_scenarios, rho = 70, alpha=100
    ):
        ans = []
        of_array = []
        dict_data = instance.get_data()

        # temporary global solution (initialize the TGS for the first iteration)
        TGS = 0

        # max iterations
        maxiter = 100

        #iteration
        k=0

        #lagrangian multiplier
        lam = np.zeros((n_scenarios, dict_data['n_stations']))
        start = time.time()
        # solve the base problem for each of the solutions 
        # (solve the problem for the first time in order to initialize the first stage solution at the zero-th iteration)
        # For each scenario, solve the mono-scenario problem
        for i, s in enumerate(np.rollaxis(scenarios, 2)):
            of, sol = self.DEP_solver(dict_data, s, TGS, lam[i], rho)
            of_array.append(of)
            ans.append(np.array(sol))
        x_s_arrays = ans
        
        # compute temporary global solution for first iteration
        TGS = np.average(x_s_arrays, axis=0).astype(int)
        lam = rho*(x_s_arrays - TGS)

        for k in range(1, maxiter+1):

            if ( np.all(abs(x_s_arrays-TGS) == 0) ):
                break

   
            x_s_arrays = []
            of_array = []
            ans = []

            # solve monoscenario problems
            for i, s in enumerate(np.rollaxis(scenarios, 2)):
                of, sol = self.DEP_solver(dict_data, s, TGS, lam[i], rho, k)
                of_array.append(of)
                ans.append(np.array(sol))
            x_s_arrays = ans

            # compute temporary global solution for first iteration
            TGS = np.average(x_s_arrays, axis=0).astype(int)
            

            # update the multipliers
            lam = lam + rho*(x_s_arrays - TGS)
            rho = alpha*rho

        end = time.time()
        comp_time = end - start

        sol_x = TGS
        of = np.average(of_array, axis=0)
        return of, sol_x, comp_time, k
