###############################################################################
# simulation_visualization.py                                                 #
# Written by Sonia Roberts, soro@seas.upenn.edu                               #
# Last updated 20210802                                                       #
# Useful for debugging and comparing results for different controllers        #
#                                                                             #
# To run this file:                                                           #
# 1) Install ffmpeg and ensure you are using Python 2.7                       #
# 2) Pick the experiment you want to visualize in folder "experiment_data"    #
# 3) Run "python simulation_visualization.py <experiment_filename>"           #
# 4) If you want to display the video before saving it, run                   #
#     "python simulation_visualization.py <experiment_filename> foo"          #
#     (where "foo" is any extra text)                                         #
#                                                                             #
###############################################################################
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import sys
import json
import robot_models
from matplotlib import animation

video_folder='experiment_videos'

robot = robot_models.robot()

if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    fname = 'test'

if ('R_' in fname):
    R_ind = fname.find('R_')
    R_val_string = fname[R_ind+2:fname.find('__')].replace('_','.')
    R = float(R_val_string)

#execfile('simulation_parameters.py')

print('Visualizing experiment '+fname)

# Load the data saved in the barebones simulation
ff = open('experiment_data/'+fname,'r')
data_list = json.load(ff)
data = np.asarray(data_list)

ts = data[:,0]     # time
xs = data[:,1:5]   # states of robot body mass and foot, [xb dxb xf dxf]
ths = data[:,5:7]  # angle of torsional leg spring
mfs = data[:,7]    # foot masses
kls = data[:,8]    # leg stiffnesses
kgs = data[:,9]    # ground stiffnesses
bls = data[:,10]   # leg damping values

dt = ts[1]-ts[0]
framesSlice = int(0.006/dt)

nTimeSteps = np.size(kls)
allTimeSteps = range(nTimeSteps)

# Set up the figure which will be used to create the animation later
fig = plt.figure()
ax = plt.axes(xlim=(-0.4, 0.4), ylim=(-.1, .4))
robot_body, = ax.plot([], [], 'bo', ms=30)
robot_foot, = ax.plot([], [], 'ro', ms=10)
robot_leftlinkage_link1, = ax.plot([], [], 'k-', lw=4)
robot_leftlinkage_link2, = ax.plot([], [], 'k-', lw=4)
robot_rightlinkage_link1, = ax.plot([], [], 'k-', lw=4)
robot_rightlinkage_link2, = ax.plot([], [], 'k-', lw=4)
ground, = ax.plot([], [], 'k-', lw = 2)

# Initialize all of the lines and points that will be plotted later; needed for
# matplotlib.animation.FuncAnimation to make a video
def init():
    robot_body.set_data([],[])
    robot_foot.set_data([],[])
    robot_leftlinkage_link1.set_data([],[])
    robot_leftlinkage_link2.set_data([],[])
    robot_rightlinkage_link1.set_data([],[])
    robot_rightlinkage_link2.set_data([],[])
    ground.set_data([],[])
    return robot_body, robot_foot, robot_leftlinkage_link1, robot_leftlinkage_link2, \
                        robot_rightlinkage_link1, robot_rightlinkage_link2, ground, 

# This function tells matplotlib.animation.FuncAnimation what to draw at each 
# timestep i by setting the data for each of the lines defined in init()
def animate(i):
    # We don't want to take every single frame: dt <= 1*10^-6!
    j = i*framesSlice
    horiz = 0
    robot_x = xs[j,0]+robot.leg.nomLegLen
    foot_x = xs[j,2]
    robot_body.set_data(horiz,robot_x)
    
    # Use geometry to find the correct positions of the leg link endpoints
    leftlink_knee_horiz = -robot.leg.l1*np.sin(ths[j,0])
    if ths[j,0]>np.pi/2.:
        leftlink_knee_vert = -(np.sqrt((robot.leg.l1**2)-(leftlink_knee_horiz**2)))
    else:
        leftlink_knee_vert = (np.sqrt((robot.leg.l1**2)-(leftlink_knee_horiz**2)))
    robot_leftlinkage_link1.set_data([0,leftlink_knee_horiz],[robot_x,robot_x+leftlink_knee_vert])
    robot_leftlinkage_link2.set_data([leftlink_knee_horiz,horiz],[robot_x+leftlink_knee_vert,foot_x])
    robot_rightlinkage_link1.set_data([0,-leftlink_knee_horiz],[robot_x,robot_x+leftlink_knee_vert])
    robot_rightlinkage_link2.set_data([-leftlink_knee_horiz,horiz],[robot_x+leftlink_knee_vert,foot_x])
    ground.set_data([-0.5, 0.5],[0, 0])
    robot_foot.set_data(horiz,foot_x)
    return robot_body, robot_foot, robot_leftlinkage_link1, robot_leftlinkage_link2, \
                        robot_rightlinkage_link1, robot_rightlinkage_link2, ground, 

# FuncAnimation takes in the animation function and an initializer function, 
# and loops the video in a figure window until the user closes it. The video
# will also be saved to a .mp4 using a filename based on the name of the data
# file visualized. The "blit" flag tells the animation function whether to 
# completely redraw the animation every time, or re-use whatever parts of the 
# background and previously drawn lines it can. This saves a lot of time. 
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=int(np.floor(nTimeSteps/framesSlice)), interval=20, blit=True)
#anim.save('/'+video_folder+'/'+fname+'_video.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
anim.save(video_folder+'/'+fname+'_video.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()