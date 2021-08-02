'''
jump_once.py
Written by Sonia Roberts, soro@seas.upenn.edu
Last updated 20210705
Based on Matlab simulation used to derive active damping controller in
Roberts & Koditschek (2018) 

This Python script is a discrete time simulation of a robot jumping on 
granular media using the Goldman lab model. 

The robot is modeled as a virtual linear spring with a compression-
extension controller which changes the spring stiffness at the bottom of
stance.

The ground is modeled as a nonlinear spring using the added mass model from
Aguilar & Goldman (2016). There are three forces: A nonlinear compression
spring force, a quadratic damping force with a depth-dependent coefficient,
and a force resulting from the acceleration of the added mass of grains
moving with the foot. All of these forces are only active as the robot's
foot moves into the ground. There are no restoring forces.

The physical robot has a control loop of 1kHz, but I advise running the
simulation with a dt at least as small as 10^-6. To prevent the robot from 
gaining an unfair advantage in simulation as opposed to the real world, the 
robot can only change its stiffness or damping coefficient every 0.001s,
regardless of the timestep chosen for simulation. 

For each of the inputs, I recommend a range of values that should work 
reasonably well and give an explanation of the physical meaning of the 
variable in question, though your mileage may vary if you modify too many
parameters simultaneously: 
R: 0.02 - 0.05        Radius of robot foot 
kl_e: 500 - 900       Extension stiffness 
xdot: -1.5 to -0.5    Initial velocity of robot body and toe           
AD_FLAG: True/False   Use active damping?
bl_c: 0.5-25          Damping of the virtual spring during compression 
bl_e: 0.5-25          Damping of the virtual spring during extension 
kl_c: 100-300         Stiffness of the virtual spring during compression 
max_runtime: 0.5-1.0  The maximum simulation time 
dt: 0.000001          The time step length for the simulation 
Rho: 1000.0           Density of the grains (poppyseeds)
Phi: 0.50-0.63        Volume fraction of the granular media 
kg_scale: 0.75-2      A scaling factor for the ground's spring force 
bg_scale: 0.5-4       A scaling factor for the ground's damping force

The simulation gives a dict simulation_results as output. This dict 
contains the following: 
joules      (float)   The cost of the jump in joules
toaster     (float)   The energy lost to heat over the course of the jump
mechanical  (float)   The electrical energy converted to mechanical energy
apex        (float)   The apex of the robot during flight
body        (list)    The positions of the robot body during the simulation
toe         (list)    The positions of the robot's toe during the simulation

This file needs the following files in the same folder:
* sand_goldman.py
* calc_cost_to_jump.py
* robot_models.py

I recommend calling this function from a script in another file that tests 
the range of parameters you are interested in. An example would be:

import jump_once
import matplotlib.pyplot as plt

for (ls,R) in zip(['k:','k--','k-'],[0.03, 0.04, 0.05]):
    simulation_results = jump_once.jump_once(R=R)
    ts = [i*0.000001 for i in range(len(simulation_results['body']))]
    plt.plot(ts,simulation_results['body'],ls)
    plt.plot(ts,simulation_results['toe'],ls)
plt.plot([0,max(ts)],[0,0],'k')
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.title('Height of jumps, varying foot size')
plt.legend(['R = 0.03', 'R = 0.04', 'R = 0.05'])
plt.show()

'''

from math import *
import os
import matplotlib.pyplot as plt
import numpy as np
import sand_goldman 
import json
import sys
import time 
import calc_cost_to_jump
import robot_models

def jump_once(R=0.03, kl_e=650, bl_AD=0, xdot=-1.0, AD_FLAG=False, bl_c=0.5, bl_e=5, kl_c=250, max_runtime=0.9, dt=0.000001, Rho=1000.0, Phi = 0.57, data_folder='experiment_data', kg_scale=1, bg_scale=1, SAVE_DATA=False, fname='test'):
    robot = robot_models.robot()
    sand = sand_goldman.sand(R=R,Phi=Phi,Rho=Rho)
    sand.m0 = robot.mf

    x_0 = [0, xdot, 0, xdot]

    start = time.time()

    BODPOS = 0
    BODVEL = 1
    BODACC = 0
    TOEPOS = 2
    TOEVEL = 3
    TOEACC = 1

    mb = robot.mb # body mass

    g = 9.81

    # The amount of time that the simulation should run for before timing out. 
    # Determined by max_runtime (above). 
    tspan = np.arange(0.0,max_runtime,dt)
    nTimeSteps = len(tspan)

    xs = np.zeros((nTimeSteps,4))       # Positions and velocities of body and foot
    ths = np.zeros((2*nTimeSteps,2))    # Theta refers to the motor angles
    kgs = np.zeros((nTimeSteps,1))      # Stiffness of ground
    bgs = np.zeros((nTimeSteps,1))      # Damping of ground
    mfs = np.zeros((nTimeSteps,1))      # Mass of foot 
    kls = np.zeros((nTimeSteps,1))      # Stiffness of leg
    bls = np.zeros((nTimeSteps,1))      # Damping of leg
    ddxs = np.zeros((nTimeSteps,2))     # Acceleration of body and foot
    Fbfs = np.zeros((nTimeSteps,1))     # Force from leg spring
    c2e = np.zeros((nTimeSteps,3))      # compToExt

    # The ground can only compress, and has no restoring forces, but the robot leg 
    # needs to go through two rounds of pushing on the ground: compression and 
    # extension. This value is the "height" of the ground in the world frame 
    # coordinates, where the world frame has zero height at the initial ground 
    # height. 
    groundzero = 0.0
    groundzeros = []

    # Initial kinetic energy
    kin_0 = 0.5*mb*x_0[BODVEL]**2+0.5*robot.mf*x_0[TOEVEL]**2
    ens = [] # Energy

    # Last timestep at which the robot was able to change its stiffness or damping
    # coefficient. Regardless of what timestep size the simulation is run at, the 
    # robot cannot change its controller faster than every 0.001s.
    lastControlT = 0.0

    # The leg stiffness and damping values must be initialized because the robot 
    # cannot act more frequently than every 0.001s. 
    kl_n = kl_c # Leg stiffness at timestep n = 0 is the "compression" stiffness
    bl_n = bl_c   # Leg damping at timestep n = 0 is the natural damping of the leg

    # Manually input the data for the first timestep
    ts = 0
    x_n = x_0
    th_0 = robot.leg.inv_kinematics(robot.leg.nomLegLen)
    ths[0,0] = robot.leg.inv_kinematics(xs[0,BODPOS]-xs[0,TOEPOS]+robot.leg.nomLegLen)
    ths[0,1] = 0

    bg_n = bg_scale*sand.damping(0) # Assume xf at ts = 0 is 0
    mf_n = robot.mf
    kg_n = 0

    # Calculate forces from the leg spring connecting body and foot, and the ground
    Fbf = kl_n*(x_n[BODPOS]-x_n[TOEPOS]) + bl_n*(x_n[BODVEL]-x_n[TOEVEL])
    Fg = kg_n*x_n[TOEPOS] + np.sign(x_n[TOEVEL])*(bg_n*x_n[TOEVEL]**2)

    # Acceleration for this time step is based on the last timestep
    ddxs[ts,BODACC] = -Fbf/mb - g;
    ddxs[ts,TOEACC] = Fbf/mf_n -Fg/mf_n - g;

    # Calculate the positions and velocities at this timestep using velocity and 
    # acceleration from the last time step
    xs[ts,BODPOS] = x_n[BODPOS] + dt*x_n[BODVEL]
    xs[ts,TOEPOS] = x_n[TOEPOS] + dt*x_n[TOEVEL]
    xs[ts,BODVEL] = x_n[BODVEL] + dt*ddxs[ts,BODACC]
    xs[ts,TOEVEL] = x_n[TOEVEL] + dt*ddxs[ts,TOEACC]

    # These are the ground damping, added foot mass, and ground stiffness values 
    # for this foot position and velocity, which will be used in the next timestep
    # to calculate forces exerted by the ground
    absxf = abs(xs[ts,TOEPOS])
    bgs[ts] = bg_scale*sand.damping(absxf)
    mfs[ts] = sand.mass(absxf) + robot.mf
    kgs[ts] = kg_scale*sand.spring(absxf)/absxf

    ts = 1

    print('Entering stance mode')

    while ts < nTimeSteps:
        if ts%100000==0:
            print('Time: '+str(round(dt*ts,2))+'/'+str(round(dt*nTimeSteps,2)))
        
        x_n = xs[ts-1,:]
        th_n = ths[ts-1,:]
        
        bg_n = bgs[ts-1]
        mf_n = mfs[ts-1]
        kg_n = kgs[ts-1]
        
        # The robot acts based on its state in the previous timestep, and can only 
        # act every 0.001s regardless of how frequently the physics models update
        if ts*dt - lastControlT > 0.001:
            lastControlT = ts*dt
            
            # Check: Compression --> extension? 
            # Switch from compression to extension spring if either: 
            # 1) The robot is starting to jump up: 
            #   a) The leg length is increasing AND
            #   b) The toe is below the ground surface AND
            #   c) The body velocity is positive, OR
            # 2) The leg length has gone below 0.105m, where 0.1 is the minimum 
            # possible leg length. The robot will crash if its leg length goes 
            # below 0.1. 
            compToExt = (x_n[BODVEL]-x_n[TOEVEL]>0 and x_n[TOEPOS] < 0 and x_n[BODVEL]>0) or x_n[BODPOS]-x_n[TOEPOS]+robot.leg.nomLegLen<0.105
            
            kl_n = ~compToExt*kl_c + compToExt*kl_e
            bl_n = ~compToExt*bl_c + compToExt*bl_e + compToExt*AD_FLAG*bl_AD*abs(x_n[TOEVEL])*(x_n[TOEVEL]<0)

            c2e[ts,0] = (x_n[BODVEL]-x_n[TOEVEL]>0 and x_n[TOEPOS] < 0 and x_n[BODVEL]>0)
            c2e[ts,1] = x_n[BODPOS]-x_n[TOEPOS]+robot.leg.nomLegLen<0.105
            c2e[ts,2] = compToExt
        
        kls[ts] = kl_n
        bls[ts] = bl_n

        # This prevents the ground from giving restoring force. If the toe lifts 
        # above this position, it will be considered to be in "flight" phase. 
        if x_n[TOEPOS] < groundzero: groundzero = x_n[TOEPOS]
        
        # The equations of motion determine the forces and therefore accelerations 
        # during each timestep. 
        # The motors behave like virtual torsional springs. Therefore a change of 
        # coordinates through the kinematics of the leg linkage is required in 
        # order to describe the forces at the toe. 
        #F_mots = kl_n*(th_n[0]-th_0) + bl_n*th_n[1]
        #r = x_n[0]-x_n[2]+robot.leg.nomLegLen
        
        # Cap the forces from the leg at 70 Nm
        # Stall torque of 1 motor: 3.5
        # Length of first leg link: 0.1
        # 3.5*2/0.1 = 70
        Fbf = kl_n*(x_n[BODPOS]-x_n[TOEPOS]) + bl_n*(x_n[BODVEL]-x_n[TOEVEL])
        if (Fbf > 70.0):
            Fbf = 70.0
        if (Fbf < -70):
            Fbf = -70
        Fbfs[ts] = Fbf
        Fg = kg_n*x_n[TOEPOS] + np.sign(x_n[TOEVEL])*(bg_n*x_n[TOEVEL]**2)

        ddxs[ts,BODACC] = -Fbf/mb - g;
        ddxs[ts,TOEACC] = Fbf/mf_n -Fg/mf_n - g;        
        
        xs[ts,BODPOS] = x_n[BODPOS] + dt*x_n[BODVEL]
        xs[ts,TOEPOS] = x_n[TOEPOS] + dt*x_n[TOEVEL]
        xs[ts,BODVEL] = x_n[BODVEL] + dt*ddxs[ts,BODACC]
        xs[ts,TOEVEL] = x_n[TOEVEL] + dt*ddxs[ts,TOEACC]

        # If the robot's toe position is at or below the current "ground zero" with
        # negative velocity, then we use the granular media model to determine the
        # stiffness and damping of the ground and the foot mass. Otherwise, the
        # stiffness and damping coefficients of the ground are zero, and the mass 
        # of the foot is initial mass of the foot alone. 
        if (xs[ts,TOEPOS] < groundzero) and (xs[ts,TOEVEL] < 0):
            absxf = abs(xs[ts,TOEPOS])
            bgs[ts] = bg_scale*sand.damping(absxf)
            mfs[ts] = sand.mass(absxf) + robot.mf
            kgs[ts] = kg_scale*sand.spring(absxf)/absxf
        else:
            bgs[ts] = 0
            mfs[ts] = robot.mf
            kgs[ts] = 0

        r = xs[ts,BODPOS]-xs[ts,TOEPOS]+robot.leg.nomLegLen
        dr = xs[ts,BODVEL]-xs[ts,TOEVEL]
        ths[ts,0] = robot.leg.inv_kinematics(r)
        ths[ts,1] = robot.leg.D_inv_kinematics(r)*dr
        
        if kl_n*(x_n[BODPOS]-x_n[TOEPOS]) + bl_n*(x_n[BODVEL]-x_n[TOEVEL]) > mf_n*g:
            xs = xs[0:ts-1]
            kgs = kgs[0:ts-1]
            bgs = bgs[0:ts-1]
            mfs = mfs[0:ts-1]
            kls = kls[0:ts-1]
            ddxs = ddxs[0:ts-1]
            bls = bls[0:ts-1]
            #ths = ths[0:ts-1]
            break
        
        ts = ts + 1
        
        groundzeros.append(groundzero)

    if (ts < nTimeSteps):
        # Remove the zeros from the kg, bg, and mf arrays
        for i in range(kgs.size):
            if (kgs[i] == 0):
                kgs[i] = kgs[i-1]
                bgs[i] = bgs[i-1]
                mfs[i] = mfs[i-1]

        print('Entering flight mode')

        x_flight = np.zeros((nTimeSteps,4))
        x_flight[0] = xs[ts-2]

        tsf = 1;

        while tsf < nTimeSteps:
            x_n = x_flight[tsf-1]
            x_flight[tsf,BODPOS] = x_n[BODPOS] + dt*x_n[BODVEL]
            x_flight[tsf,TOEPOS] = x_n[TOEPOS] + dt*x_n[TOEVEL]
            x_flight[tsf,BODVEL] = x_n[BODVEL] + dt*(-kl_n*(x_n[BODPOS]-x_n[TOEPOS])/mb-g)
            x_flight[tsf,TOEVEL] = x_n[TOEVEL] + dt*((kl_n*(x_n[BODPOS]-x_n[TOEPOS])+bl_n*(x_n[BODVEL]-x_n[TOEVEL]))/robot.mf-g)
            
            if x_flight[tsf,TOEPOS] < 0 and x_flight[tsf,BODVEL] < 0 and x_flight[tsf,TOEVEL] < 0:
                x_flight = x_flight[1:tsf]
                break
            tsf = tsf + 1

        xs = np.vstack((xs,x_flight[0:-1]))
        mfs = np.vstack((mfs,robot.mf*np.ones((tsf-2,1))))
        kls = np.vstack((kls,kl_c*np.ones((tsf-2,1))))
        bls = np.vstack((bls,bl_n*np.ones((tsf-2,1))))
        kgs = np.vstack((kgs,np.zeros((tsf-2,1))))
        bgs = np.vstack((bgs,np.zeros((tsf-2,1))))
        times = np.zeros((len(bls),1))

        for (i,t) in zip(range((ts-1),len(bls)-1),tspan[(ts-1):len(bls)-1]):
            times[i] = tspan[i]
        for i in range(len(bls)):
            r = xs[i,0]-xs[i,2]+robot.leg.nomLegLen
            dR = xs[i,1]-xs[i,3]
            ths[i,0] = robot.leg.inv_kinematics(r)
            ths[i,1] = robot.leg.D_inv_kinematics(r)*dR
        ths = ths[0:len(bls),:]

        ###############################################################################
        # Write the current experiment to a text file so that the data can be         #
        # visualized or used for further experiments as desired                       #
        ###############################################################################
        if not os.path.isdir(data_folder):
            os.makedirs(data_folder)

        dataToSave = np.hstack((times,xs,ths,mfs,kls,kgs,bls,bgs))
        dataToSaveList = dataToSave.tolist()
        if (SAVE_DATA):
            print('Saving data')
            with open(data_folder+'/'+fname,'w') as write_file:
                json.dump(dataToSaveList,write_file)
    #print("total time")
    #print(time.time() - start)
    apex = max(xs[:,BODPOS]+robot.leg.nomLegLen)
    energy_vals = calc_cost_to_jump.calc_joules(xs, ths, kls, bls)
    joules = energy_vals['joules']
    print('joules: '+str(joules))
    print('apex: '+str(apex))
    print('depth: '+str(min(xs[:,TOEPOS])))
    print('')

    simulation_results = energy_vals
    simulation_results['apex'] = apex
    simulation_results['body'] = np.array([xbodpos+robot.leg.nomLegLen for xbodpos in xs[:,BODPOS]])
    simulation_results['toe'] = xs[:,TOEPOS]
    
    # Uncomment these two lines if you want the simulation results to include velocity of body and toe
    simulation_results['bodvel'] = xs[:,BODVEL]
    simulation_results['toevel'] = xs[:,TOEVEL]
    simulation_results['Fbfs'] = Fbfs
    
    return simulation_results