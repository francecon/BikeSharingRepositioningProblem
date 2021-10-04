# -*- coding: utf-8 -*-
import logging
import numpy as np

np.random.seed(42)
class Instance():
    def __init__(self, sim_setting):

        logging.info("starting simulation...")
        
        self.n_stations = sim_setting['n_stations']
        
        #c
        self.procurement_cost = sim_setting['procurement_cost'] 
        
        #v_i
        self.stock_out_cost = [sim_setting['stock_out_cost']] * self.n_stations
        
        #w_i
        self.time_waste_cost = [sim_setting['time_waste_cost']] * self.n_stations
        
        #t_ij
        self.trans_ship_cost = [[sim_setting['trans_ship_cost'] for x in range(self.n_stations)] for y in range(self.n_stations)] 
        
        # k_i
        self.stations_cap = np.around(np.random.uniform(
                                sim_setting['station_max_cap'],
                                sim_setting['station_min_cap'],
                                size=self.n_stations
                                )
                            )

        logging.info(f"stations_number: {self.n_stations}")
        logging.info(f"procurement_costs: {self.procurement_cost}")
        logging.info(f"stock_out_costs: {self.stock_out_cost}")
        logging.info(f"time_waste_cost: {self.time_waste_cost}")
        logging.info(f"Transshipment_cost: {self.trans_ship_cost}")
        logging.info(f"stations_capacity: {self.stations_cap}")
        logging.info("simulation end")

    def get_data(self):
        logging.info("getting data from instance...")
        return {
            "procurement_cost": self.procurement_cost,
            "stock_out_cost": self.stock_out_cost,
            "time_waste_cost": self.time_waste_cost,
            "trans_ship_cost": self.trans_ship_cost,
            "n_stations": self.n_stations,
            "station_cap": self.stations_cap
        }
