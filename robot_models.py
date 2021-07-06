'''
class motor(), a model of the T-motor U-8 series motors
class leg(), a model of a single two-motor leg with a simple four-bar linkage

Written by Sonia Roberts, soro@seas.upenn.edu
Last updated 20200401
'''

from math import *
import matplotlib.pyplot as plt
import numpy as np

class motor():
    def __init__(self):
        self.Rm = 0.186 # Ohms; motor resistance
        self.kt = 0.09549296586 # Nm/A; motor torque constant
        self.kv = 100 # rpm/V; motor voltage constant
        self.tau_s = 3.5 # Nm; stall torque
        self.drag = self.kt*self.kv/(self.Rm**2) # electromechanical drag
    
    # Based on physical experiments we suspect that the stall torque is vastly 
    # underestimated, so for now this function is ignored
    def capForce(self,F,nMots):
        if np.abs(F) > self.tau_s*nMots:
            F = np.sign(F)*self.tau_s*nMots
        return F

class leg():
    def __init__(self):
        self.nMots = 2 # Number of motors
        self.l1 = 0.1  # Length of first leg link
        self.l2 = 0.2  # Length of second leg link
        self.nomLegLen = 0.27 # Nominal leg length = rest length of spring
        self.motor = motor() # Instance of class motor (above)
    
    # function inv_kinematics calculates the rotation of the motor using the 
    # inverse kinematics of the leg linkage. 
    # The input r indicates the radius of the leg, which is the same as its 
    # length in the rest of the code and write-up used here. 
    # (The Ghost SDK and Minitaur literature typically use polar coordinates to
    # refer to the toe position.)
    def inv_kinematics(self,r):
        l1 = self.l1
        l2 = self.l2
        numerator = (l1**2)-(l2**2)+r**2
        denominator = 2*l1*r
        return np.pi - np.arccos(numerator/denominator)

    def kinematics(self,beta):
        #alpha = -beta -np.pi
        alpha = -beta + np.pi
        return self.l1*np.cos(alpha)+np.sqrt((self.l2**2)-(self.l1**2)*np.sin(alpha)**2)
    
    def D_inv_kinematics(self,r):
        l1 = self.l1
        l2 = self.l2
        numerator = -l1**2 + l2**2 + r**2
        sqrtNumerator = ((l1**2)-(l2**2)+(r**2))**2
        sqrtDenominator = 4*(l1**2)*r**2
        denominator = (2*l1*r**2)*np.sqrt(1-(sqrtNumerator/sqrtDenominator))
        return numerator/denominator
        
    def D_kinematics(self,beta):
        l1 = self.l1
        l2 = self.l2
        #firstterm = l1*np.sin(beta)
        firstterm = -l1*np.sin(beta)
        numerator = -(l1**2)*np.cos(beta)*np.sin(beta)
        denominator = np.sqrt((l2**2)-(l1**2)*np.sin(beta)**2)
        return firstterm + numerator/denominator
    
    # function force calculates the force at the toe, given an input command 
    # (gain) and a leg position (r). The nominal leg length nomLegLen is used 
    # to calculate the amount of deflection. 
    def force(self,r,gain):
        # Calculate the force at the motor using the inverse kinematics, the 
        # commanded gain, and leg deflection from the toe
        beta = self.inv_kinematics(r)
        commandedForceAtToe = gain*(r-self.nomLegLen)
        commandedForceAtMotor = self.l1*commandedForceAtToe     
        # The actual force at the motor is determined by the motor stall torque
        #actualForceAtMotor = self.motor.capForce(commandedForceAtMotor,self.nMots)
        actualForceAtMotor = commandedForceAtMotor
        # Calculate the actual force at the toe using forward kinematics
        #actualForceAtToe = actualForceAtMotor*self.D_kinematics(beta)
        actualForceAtToe = commandedForceAtMotor/self.l1
        return actualForceAtToe

class robot():
    def __init__(self):
        '''
        The mass of the hopper in total while connected to the string pot is
        ~2kg. Modeling the leg as a massless rod with a point mass on the
        end at the foot, the mass of the foot can reasonably be assumed to
        be <= 10% of the mass of the robot's body, because the dynamics of
        the foot don't really affect the dynamics of the body when the robot
        is in flight. 
        '''
        self.mb = 1.75      # The mass of the robot body, mb > 0
        self.mf = 0.175     # The mass of the foot, mf > 0 
        self.leg = leg()