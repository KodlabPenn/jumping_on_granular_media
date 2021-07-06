import numpy as np
import matplotlib.pyplot as plt

import sand_goldman
import robot_models
import os
import matplotlib.pyplot as plt
import sys
import json
from matplotlib import animation

robot = robot_models.robot()
dt = 0.000001

def motor_model(current):
	nMots = 2
	l1 = robot_models.l1
	Tstall_leg = nMots*Tstall_motor/l1
	torque = kt*current
	if torque > Tstall_leg:
		torque = Tstall_leg
	    
	return torque

def calc_joules(xs, ths, kls, bls):

	# Motor constants from Kenneally et al. 2018
	kt = 0.09549296586 	#                	Units = Nm/A
	Rm = 0.186 			# Motor resistance 	Units = Ohms

	inv_kt = 1.0/kt
	nomleglen = robot.leg.nomLegLen

	all_rs = xs[:,0]-xs[:,2]
	all_drs = xs[:,1]-xs[:,3]
	forces = [-kl*r - bl*dr for (r, dr, kl, bl) in zip(all_rs, all_drs, kls, bls)]

	torques = [f/robot.leg.D_inv_kinematics(r+nomleglen) for (r, f) in zip(all_rs, forces)]

	raw_currents = [t*inv_kt for t in torques]

	currents = raw_currents

	toasterpower = [2.0*Rm*((0.5*c)**2) for c in currents]
	avgtoasterpower = np.mean(toasterpower)

	mechanicalpower = [dth*t for (dth, t) in zip(ths[:,1], torques)]
	avgmechpower = np.mean(mechanicalpower)

	toaster = dt*avgtoasterpower*len(toasterpower)
	mechanical = dt*avgmechpower*len(mechanicalpower)

	joules = toaster+mechanical

	energy_vals = dict()
	energy_vals['joules'] = joules
	energy_vals['toaster'] = toaster
	energy_vals['mechanical'] = mechanical

	#print('Calc cost to jump joules: '+str(joules))
	
	return energy_vals