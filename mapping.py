from  load_data import *
from p2_utils import *
import numpy as np
import os
import cv2
import math
import time
from matplotlib import pyplot as plt
import time
from tqdm import tqdm
from scipy.special import softmax

def init_map():
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -20  #meters
    MAP['ymin']  = -20
    MAP['xmax']  =  20
    MAP['ymax']  =  20 
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
    MAP['log_odds'] = np.zeros((MAP['sizex'],MAP['sizey'])) #DATA TYPE: char or int8
    MAP['robo_state'] = [0,0,0]
    
    return MAP

def slam(lidar_data,odometry_data,MAP):

    ############## Initializations ##############
    num_pose = 1081#len(lidar_data[0]['delta_pose'])
    plot_en = 1
    num_particles = 20
    part_states = np.zeros((3,num_particles))
    part_corr = np.zeros(num_particles)
    part_wghts = np.ones(num_particles) / num_particles
    

    ############### Sanity ##############
    np_lidar_delta_pose = np.zeros((3,len(lidar_data)))
    np_lidar_scan = np.zeros((num_pose,len(lidar_data)))
    for i in range(len(lidar_data)):
        np_lidar_delta_pose[:,i] = lidar_data[i]['delta_pose'][0].T
        np_lidar_scan[:,i] = lidar_data[i]['scan']


    xy_noise_pwr = 1#1e3
    yaw_noise_pwr =1# 1e-7
    start_time = time.time()
    cur_scan = 0
    ###### Initial Scan ######

    cur_pose = np_lidar_delta_pose[:,cur_scan]
    MAP = update_and_map(np_lidar_scan[:,cur_scan],cur_pose,MAP,odometry_data['head_angles'][:,cur_scan],update_log_odds = True)

    #for cur_scan in range(1,np_lidar_scan.shape[1]):
    for cur_scan in tqdm(range(1,np_lidar_scan.shape[1])):
    #for cur_scan in range(1,np_lidar_scan.shape[1]):
        #MAP,xs0,ys0 = update_and_map(np_lidar_scan[:,cur_scan],np_lidar_delta_pose[:,cur_scan],MAP,odometry_data['head_angles'][:,cur_scan])
        #### Update the Map based on Lidar Reading #####
        cur_pose = cur_pose + np_lidar_delta_pose[:,cur_scan]
        for particle in range(num_particles):
            ######## Predict #########
            noise = xy_noise_pwr * np.random.normal(0,0.1,2)#Particular noise to x,y
            noise = np.hstack((noise,yaw_noise_pwr * np.random.normal(0,0.1,1))) #Particular noise to x,y
            part_states[:,particle] = (noise + cur_pose).T
            ######## Update Weights#########
            MAP,part_corr[particle] = update_and_map(np_lidar_scan[:,cur_scan],part_states[:,particle],MAP,odometry_data['head_angles'][:,cur_scan])
            part_wghts[particle] = part_wghts[particle] * np.exp(part_corr[particle])
        
        
        ####### ReSampling ######


        ####### Update Map with best particle ########
        best_particle = np.argmax(softmax(part_corr)) ## Do we really need softmax???
        #best_particle = np.argmax(part_corr)
        part_wghts /= np.sum(part_wghts)

        if(cur_scan%5000 ==0):
            MAP = update_and_map(np_lidar_scan[:,cur_scan],part_states[:,best_particle],MAP,odometry_data['head_angles'][:,cur_scan],update_log_odds = True,plot_en = 1)
        else:
            MAP = update_and_map(np_lidar_scan[:,cur_scan],part_states[:,best_particle],MAP,odometry_data['head_angles'][:,cur_scan],update_log_odds = True,plot_en = 0)

        #### Predict the next step based on Odometry ########
        #MAP['robo_xloc'].append(MAP['robo_xloc'][-1] + np_lidar_delta_pose[0,cur_scan] + noise_pwr * np.random.randn(1))
        #MAP['robo_yloc'].append(MAP['robo_yloc'][-1] + np_lidar_delta_pose[1,cur_scan] + noise_pwr * np.random.randn(1))
        #if(cur_scan == 0):
        #    xs_leg,ys_leg = xs0,ys0

    print("Time taken",time.time()-start_time)


    ######## Update #########


        


def update_and_map(ranges,pose,MAP,head_angles,update_log_odds=False,plot_en=0):
    #dataIn = io.loadmat("lidar/train_lidar0.mat")
    angles = np.array([np.arange(-135,135.25,0.25)*np.pi/180.]).T
    ranges = ranges.reshape((ranges.shape[0],1))
    #ranges = np.double(dataIn['lidar'][0][110]['scan'][0][0]).T
    
    # take valid indices
    valid_range = np.logical_and((ranges < 30),(ranges> 0.1))
    ranges = ranges[valid_range]
    angles = angles[valid_range]
    
    # xy position in the sensor frame
    xs0 = np.array([ranges*np.cos(angles)])
    ys0 = np.array([ranges*np.sin(angles)])
    scan_ranges = np.vstack((xs0,ys0))
    dummy = np.vstack((xs0*0,ys0*0))
    scan_ranges = np.vstack((scan_ranges,dummy))
    world = convert2world_frame(scan_ranges,pose,head_angles) 
    xs0 = world[0,:].reshape(1,world.shape[1])
    ys0 = world[1,:].reshape(1,world.shape[1])
    # convert position in the map frame here 
     
    #Y = np.concatenate([np.concatenate([xs0,ys0],axis=0),np.zeros(xs0.shape)],axis=0)
    
    # convert from meters to cells
    xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    
    # build an arbitrary map 
    indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
    MAP['map'][xis[0][indGood[0]],yis[0][indGood[0]]]=1
      
    x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
    y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map
    
    x_range = np.arange(-0.2,0.2+0.05,0.05)
    y_range = np.arange(-0.2,0.2+0.05,0.05)
    
    map_shape = MAP['map'].shape

    #correlation = mapCorrelation(map_threshold,x_im,y_im,Y,x_range,x_range)
    ################# Plot ################
    if(plot_en):
          fig = plt.figure(figsize=(18,6))
          #plot original lidar points
          ax1 = fig.add_subplot(121)
          #plt.plot(xs0,ys0,'.k')
          #plt.plot(xis,yis,'.k')

          #plt.plot(xs_leg,ys_leg,'.k')
          #plt.scatter(0,0,s=30,c='r')

          #plt.scatter(MAP['robo_xloc'],MAP['robo_yloc'],s=0.01,c='r')
          plt.scatter(MAP['robo_state'][0],MAP['robo_state'][1],s=10,c='r')
          #robo_x += np_lidar_delta_pose[0,cur_scan]
          #robo_y += np_lidar_delta_pose[1,cur_scan]

          #plt.xlabel("x")
          #plt.ylabel("y")
          #plt.title("Laser reading (red being robot location)")
          #plt.axis('equal')

          #plot map
          map_threshold = np.where(MAP['log_odds'] > 0, np.ones(map_shape),np.zeros(map_shape))
          ax2 = fig.add_subplot(122)
          plt.imshow(map_threshold,cmap="hot")
          plt.title("Occupancy map")
          
          #plot correlation
          #ax3 = fig.add_subplot(133,projection='3d')
          #X, Y = np.meshgrid(np.arange(0,9), np.arange(0,9))
          #ax3.plot_surface(X,Y,c,linewidth=0,cmap=plt.cm.jet, antialiased=False,rstride=1, cstride=1)
          #plt.title("Correlation coefficient map")

          plt.show()


    #################################################
    
    ######## Update Log-Odds #########
    if(update_log_odds == True):
        free_cells = np.zeros(map_shape)
        occupied_cells = np.zeros(map_shape)
        #x_cur,y_cur = 0,0
        pose = pose.astype(np.int16)
        pose_x = np.ceil((pose[0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        pose_y = np.ceil((pose[1] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
      
        for scan in range(xis.shape[1]):
            if(yis[0,scan] >  MAP['sizey'] or xis[0,scan] >  MAP['sizex']) :
                pass
            else:
              free_cells = cv2.line(free_cells,(pose_x,pose_y),(xis[0,scan],yis[0,scan]),color = 1,thickness=1)
              occupied_cells[yis[0,scan],xis[0,scan]] = 1
              #free_cells = cv2.line(free_cells,(pose_y,pose_x),(yis[0,scan],xis[0,scan]),color = 255,thickness=1)
              #occupied_cells[xis[0,scan],yis[0,scan]] = 1
        #occupied_cells[pose_y,pose_x] = 1
      
        sensor_confidence = np.full(map_shape,np.log(9))
        MAP["log_odds"] += 2 * occupied_cells * sensor_confidence #Because we are subtracting one value at occupied free cell
        MAP["log_odds"] = MAP['log_odds'] - free_cells * sensor_confidence  #Because we are subtracting one value at occupied free cell
        MAP['robo_state'] = pose
        return MAP
    
    
    #TODO: Cap log odds, Threshold Map also
    ####### Perform Correlation #######
    
    
    Y = np.vstack((xs0,ys0))
      
    #correlation = mapCorrelation(MAP['map'],x_im,y_im,Y,x_range,y_range)
    map_threshold = np.where(MAP['log_odds'] > 0, np.ones(map_shape),np.zeros(map_shape))
    #print(map_threshold.shape)
    #print("range",np.min(MAP['log_odds']),np.max(MAP['log_odds']))

    plot_en = 0
    if(plot_en == 10):
        plt.subplot(121)
        plt.imshow(MAP['log_odds'],cmap='gray')
        plt.subplot(122)
        plt.imshow(map_threshold)
        plt.show()
    #correlation = mapCorrelation(map_threshold,x_im,y_im,Y,x_range,y_range)
    correlation = mapCorrelation(map_threshold,x_im,y_im,Y,x_range,x_range)
    #Y = np.vstack((yis,xis))
    #correlation = mapCorrelation(map_threshold,y_im,x_im,Y,y_range,x_range)
    
    return MAP,correlation

      #MAP['log_odds'] = np.logical_or(MAP['log_odds'],cv2.line(empty_img,start,end)



  
#  c = mapCorrelation(MAP['map'],x_im,y_im,Y[0:3,:],x_range,y_range)
#  
#  c_ex = np.array([[3,4,8,162,270,132,18,1,0],
#		  [25  ,1   ,8   ,201  ,307 ,109 ,5  ,1   ,3],
#		  [314 ,198 ,91  ,263  ,366 ,73  ,5  ,6   ,6],
#		  [130 ,267 ,360 ,660  ,606 ,87  ,17 ,15  ,9],
#		  [17  ,28  ,95  ,618  ,668 ,370 ,271,136 ,30],
#		  [9   ,10  ,64  ,404  ,229 ,90  ,205,308 ,323],
#		  [5   ,16  ,101 ,360  ,152 ,5   ,1  ,24  ,102],
#		  [7   ,30  ,131 ,309  ,105 ,8   ,4  ,4   ,2],
#		  [16  ,55  ,138 ,274  ,75  ,11  ,6  ,6   ,3]])
#		  
#  if np.sum(c==c_ex) == np.size(c_ex):
#	  print("...Test passed.")
#  else:
#	  print("...Test failed. Close figures to continue tests.")	
#
#  
def polar2cart(lidar_data):
    import math
    theta_temp = np.arange(0, math.radians(271),math.radians(270/1080))
    theta = theta_temp[:lidar_data.shape[0]]
    return np.vstack((lidar_data * np.cos(theta),lidar_data * np.sin(theta)))

def convert2world_frame(lidar_scan,lidar_pose,head_angles):
    '''
    This takes in the lidar frame reading and computes the world frame readings
    Ideally the z axis should not change
    '''

    #Pose from lidar to head
    #print("Pose is ",lidar_pose)
    lid2head_pose = np.hstack((np.eye(3),np.array([0,0,0.15]).reshape(3,1)))
    lid2head_pose = np.vstack((lid2head_pose,np.array([0,0,0,1]).reshape(1,4)))

    #Pose from head to body
    yaw,pit = head_angles
    rot_yaw = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])#.astype(float)
    rot_pit = np.array([[np.cos(pit),0,np.sin(pit)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])#.astype(float)
    head2body_rot = rot_yaw @ rot_pit
    head2body_pose = np.hstack((head2body_rot,np.array([0,0,0.33]).reshape(3,1)))
    head2body_pose = np.vstack((head2body_pose,np.array([0,0,0,1]).reshape(1,4)))

    #Pose from body to world

    yaw = lidar_pose[2]
    body2world_rot = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])#.astype(float)
    body2world_pose = np.hstack((body2world_rot,np.array([lidar_pose[0],lidar_pose[1],0.93]).reshape(3,1)))
    body2world_pose = np.vstack((body2world_pose,np.array([0,0,0,1]).reshape(1,4)))
      

    #Total pose
    tot_pose = body2world_pose @ head2body_pose @ lid2head_pose
    return tot_pose @ lidar_scan



if __name__ == '__main__':
    lidar_data = get_lidar("lidar/train_lidar0")
    #Pose is already in world frame and scan has to be shifted to world frame
    print("Read Lidar Data")

    MAP = init_map()
    print("Map intialized")
    #lidar_data = sorted(lidar_data.items(),key = lambda k:  k['t'][0])
    #print("Sorted Lidar Data")
    odometry_data = get_joint("joint/train_joint0")
    print("Loaded Odometry data")

    slam(lidar_data,odometry_data,MAP)

