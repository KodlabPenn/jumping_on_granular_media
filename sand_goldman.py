'''
class sand, the Aguilar-Goldman model

Written by Sonia Roberts, soro@seas.upenn.edu
Last updated 20190618

The model describes the ground reaction force of sand in response to intrusion by a flat circular foot of radius R. 
The constants set in the initialization function are physical quantities that the physicists have painstakingly modeled. 
There are three components to the model: 
* sand.stiffness, a function of intrusion depth 
* sand.dissipation, a function of intrusion depth and velocity of intrusion 
* sand.mass, the added mass of the sand grains moving with the foot, a function of intrusion depth
'''

from math import *
import numpy as np

class sand():
    def __init__(self, R=0.03, Phi=0.57,Rho=1000.0):
        if R is None:
            R = 0.03
        
        self.R = R               # Radius of robot foot
        self.m0 = 0.1            # kg of toe
        
        self.Theta = 1.0472      # Theta in radians (60 from Hubicki et al.) -- shear band angle
        self.cotTheta = cos(self.Theta)/sin(self.Theta)
        #self.Phi = 0.57          # compaction used in Goldman lab experiments
        self.Phi = Phi
        self.Sigma = 0.12*10**6  # "depth dependent resistive stress"
        self.ksh = 1600.0        # "penetration stiffness"
        self.cg = 2.7            # "surrounding mass scaling factor"
        #self.Rho = 1000.0        # grain density for poppyseeds -- kg/m^3
        self.Rho = Rho
        self.b = 17.2            # tuned empirically, taken from Hubicki et al. -- "inertial drag scaling factor"
        self.C = 1.0             # mentioned as tuned empirically but couldn't find a value; constant "extra" grains moving with shearing cone
        self.Mu = 2.0            # tuned empirically, taken from Hubicki et al. -- recruitment rate
        
        self.initZc(self.R)
    
    # Set the "critical depth" value zc at which the added mass cone is fully formed
    def initZc(self,R):
        self.zc = self.R*tan(self.Theta)/self.Mu
    
    # Return the force*mass due to the stiffness of the sand
    def stiffness(self,z):
        return self.spring(z)*z
    
    # Return the force*mass due to the dissipation of kinetic energy into the sand
    def dissipation(self,z,dz):
        return self.damping(z)*dz**2
    
    # Return the mass of the toe, including both its initial mass and the mass of the added grains
    def mass(self,z):
        z_orig = z
        if (z > self.zc):
            z_orig = z
            z = self.zc
        phirhomupi = self.Phi*self.Rho*self.Mu*pi;
        thirdmuzsqrtansqrth = ((self.Mu*z)**2)/(3.*(tan(self.Theta)**2))
        Rmuztanth = self.R*self.Mu*z/tan(self.Theta)
        ma = phirhomupi*z*(self.R**2+thirdmuzsqrtansqrth-Rmuztanth)
        return ma + self.C*z_orig

    def Saflat(self,z):
        if z > self.zc:
            z = self.zc
        return pi*(self.R**2)*z + ((pi*(self.Mu**2)*(z**3))*self.cotTheta**2)/3.0 - (pi*self.R*self.Mu*(z**2))*self.cotTheta
    
    def Sacone(self,z):
        if z > self.zc:
            z = self.zc
        constinteg = (pi*self.R**2)/cos(self.Theta);
        Aflatinteg = (1./cos(self.Theta))*self.Saflat(z);
        return constinteg + Aflatinteg
    
    def dmadz(self,z):
        return self.Phi*self.Rho*self.Mu*self.Aflat(z) + self.C
    
    def Aflat(self,z):
        if z > self.zc:
            z = self.zc
        mutanth = (self.Mu/tan(self.Theta))
        twoRmutanth = 2*self.R*self.Mu/tan(self.Theta)
        return pi*(self.R**2+(z*mutanth)**2-z*twoRmutanth)
    
    def damping(self,z):
        return self.dampScale(z)*self.b*self.cg*self.dmadz(z)
    
    def spring(self,z):
        return (self.ksh/(pi*self.R**2))*self.Saflat(z) + self.stiffScale(z)*self.Sigma*self.Sacone(z)*z

    def stiffScale(self,z):
        return 1

    def dampScale(self,z):
        return 1

# Uncomment to debug        
# sand = sand()
# print(sand.R)
# print(sand.zc)

# depths = np.arange(0.0,0.02,0.001)
# stiffs = []
# for depth in depths:
    # stiffs.append(sand.stiffness(depth))
# plt.plot(depths,stiffs)
# plt.title('Depth and stiffness')
# plt.show()