# Machine sensor readings model
# (c) CK Tham & N Sharma, ECE NUS
# E-mail: eletck@nus.edu.sg

# Note: generated values are normalised to between 0.0 and 1.0
# Modalities: volt, rotate, pressure, vibration
# Original ranges:
# volt:      min 97.33360378, max 255.1247173
# rotate:    min 138.4320753, max 695.0209844
# pressure:  min 51.23710577, max 185.9519977
# vibration: min 14.877054,   max 76.7910723

import numpy as np
import pickle

class Machine():
    def __init__(self):
        """
        Initialize the machine emulation
        """
        # GMM models for generation of data, and limits to convert to observations
        with open("models/gmms.pkl", 'rb') as file:
            dt_data = pickle.load(file)
            self.GMs = dt_data['GMMs']    # GMMs for data generation
            self.limits = dt_data['limits'] # list of 25th, 50th and 75th percentiles for each telemetry reading

    def readSensors(self):
        if self.curr_state in [0,1,2,3]:    #deterioration
            return self.GMs[self.curr_state].sample(n_samples=1)[0][0]
        
        elif self.curr_state in [8,9]:  #failure
            return self.GMs[4].sample(n_samples=1)[0][0]
        
        else:   # in maintenance
            return np.array([-1.0, -1.0, -1.0, -1.0])
