from  load_data import get_joint, get_lidar
from p2_utils import mapCorrelation
import numpy as np
from scipy import io
import os
import cv2
import math
import time
from matplotlib import pyplot as plt
import time
from tqdm import tqdm
from scipy.special import softmax
import csv


###### Global Configurations #####
resample_thrs = 0.5
lidar_confidence = 9
log_thrs = 100
num_true_events = 8
x_noise_pwr = 2 * 0.001757 #1e3
y_noise_pwr = 2 * 0.00199 
yaw_noise_pwr = 2 * 0.00109
num_particles = 20
plot_freq = 500
load_prev_map = False
map_x = 20
map_y = 20
map_res = 0.05
best_particle = 0


def init_map():
    MAP = {}
    MAP['res']   = map_res #meters
    MAP['xmin']  = -map_x  #meters
    MAP['ymin']  = -map_y
    MAP['xmax']  =  map_x
    MAP['ymax']  =  map_y 
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['log_odds'] = np.zeros((MAP['sizex'],MAP['sizey'])) #DATA TYPE: char or int8
    MAP['x_loc'] = []
    MAP['y_loc'] = []
    MAP['theta'] = []
    if(load_prev_map == True):
        print("Loading Previous Logs")
        MAP['log_odds'] = np.load('images/part_20_threshold_50/map_log_odds.npy')
    
    return MAP

def slam(lidar_data,odometry_data,MAP):


    ############## Initializations ##############
    num_pose = 1081#len(lidar_data[0]['delta_pose'])
    plot_en = 1
    part_states = np.zeros((3,num_particles))
    part_corr = np.zeros(num_particles)
    part_wghts = np.ones(num_particles) / num_particles
    

    ############### Sanity ##############
    np_lidar_delta_pose = np.zeros((3,len(lidar_data)))
    np_lidar_scan = np.zeros((num_pose,len(lidar_data)))
    for i in range(len(lidar_data)):
        np_lidar_delta_pose[:,i] = lidar_data[i]['delta_pose'][0].T
        np_lidar_scan[:,i] = lidar_data[i]['scan']


    start_time = time.time()
    cur_scan = 0
    particle_ids = np.arange(num_particles)
    ###### Initial Scan and Log Odds Update ######

    cur_pose = np_lidar_delta_pose[:,cur_scan]
    scan_time = lidar_data[cur_scan]['t'][0][0]
    poss_odom_values = np.where(odometry_data['ts'][0,:] > scan_time)
    cur_odom = poss_odom_values[0][0]
    MAP = update_and_map(np_lidar_scan[:,cur_scan],cur_pose,MAP,odometry_data['head_angles'][:,cur_odom],update_log_odds = True)


    for cur_scan in tqdm(range(1,np_lidar_scan.shape[1])):
        #### Update the Map based on Lidar Reading #####
        scan_time = lidar_data[cur_scan]['t'][0][0]
        poss_odom_values = np.where(odometry_data['ts'][0,:] > scan_time)
        try:
            cur_odom = poss_odom_values[0][0]
        except:
            poss_odom_values = np.where(odometry_data['ts'][0,:] < scan_time)
            cur_odom = poss_odom_values[0][-1]

        for particle in particle_ids:
            ######## Predict #########
            noise = np.random.normal(0,x_noise_pwr,1)#Particular noise to x
            noise = np.hstack((noise, np.random.normal(0,y_noise_pwr,1))) #Particular noise to y
            noise = np.hstack((noise, np.random.normal(0,yaw_noise_pwr,1))) #Particular noise to yaw
            part_states[:,particle] = noise + part_states[:,particle] + np_lidar_delta_pose[:,cur_scan]
            ######## Update Weights#########
            MAP,part_corr[particle] = update_and_map(np_lidar_scan[:,cur_scan],part_states[:,particle],MAP,odometry_data['head_angles'][:,cur_odom])
        ####### Update Map with best particle ########
        corr_softmax = softmax(part_corr)
        part_wghts = part_wghts * corr_softmax
        part_wghts /= np.sum(part_wghts)
        assert np.sum(part_wghts) > 0.99999
        best_particle = np.argmax(corr_softmax) 

#         if(cur_scan == (np_lidar_scan.shape[1]-1)):
        if((cur_scan % plot_freq == 0) or (cur_scan == np_lidar_scan.shape[1]-1)) :
            MAP = update_and_map(np_lidar_scan[:,cur_scan],part_states[:,best_particle],MAP,odometry_data['head_angles'][:,cur_odom],update_log_odds = True,plot_en = 1,cur_scan=cur_scan)
        else:
            MAP = update_and_map(np_lidar_scan[:,cur_scan],part_states[:,best_particle],MAP,odometry_data['head_angles'][:,cur_odom],update_log_odds = True,plot_en = 0)

        ####### ReSampling ######
        N_eff = 1/(np.linalg.norm(part_wghts) ** 2)
        if(N_eff < resample_thrs * num_particles):
            part_wghts = np.ones(num_particles) / num_particles
            particle_ids = np.random.choice(num_particles,num_particles,part_wghts.squeeze)
            part_states_T = part_states.T
            part_states = part_states_T[particle_ids].T

    ######## Logging Data #########
    with open("images/part_20_threshold_50/configurations.csv",'w',newline='') as f:
        writer = csv.writer(f)
        a = ['Number of Particles', str(num_particles)]
        writer.writerow(a)
        a = ['Noise x,y and yaw is ', str(x_noise_pwr), str(y_noise_pwr),str(yaw_noise_pwr)]
        writer.writerow(a)
        a = ['Resampling Threshold ', str(resample_thrs)]
        writer.writerow(a)
        a = ['Lidar confidence ', str(lidar_confidence)]
        writer.writerow(a)
        a = ['Log Accumulation Threshold ', str(log_thrs)]
        writer.writerow(a)
        a = ['Log Map Threshold ', str(num_true_events)]
        writer.writerow(a)
        a = ['Code run time', str(time.time()-start_time)]
        writer.writerow(a)
        a = ['Map Resolution x,y and resolution', str(map_x),str(map_y),str(map_res)]
        writer.writerow(a)
        np.save('images/part_20_threshold_50/map_log_odds',MAP['log_odds'])




        
def update_and_map(ranges,pose,MAP,head_angles,update_log_odds=False,plot_en=0,cur_scan=0):
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

    ######## Drop Points that have negative z value in world frame ########
    world_T = world.T
    world_corrected = world_T[np.where(world[2,:] > 0)]
    world_corrected = world_corrected.T

    xs0 = world_corrected[0,:].reshape(1,world_corrected.shape[1])
    ys0 = world_corrected[1,:].reshape(1,world_corrected.shape[1])
    # convert position in the map frame here 
     
    
    # convert from meters to cells
    xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    
      
    x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
    y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map
    
    ### 9 X 9 Correlation Matrix ###
    x_range = np.arange(-map_res * 4,map_res * 4 + map_res,map_res)
    y_range = np.arange(-map_res * 4,map_res * 4 + map_res,map_res)
    
    map_shape = MAP['log_odds'].shape

    
    ######## Update Log-Odds #########
    pix_thrs = -num_true_events * np.log(lidar_confidence) ### Use this pixel only it was detected as free more than two times
    if(update_log_odds == True):
        free_cells = np.zeros(map_shape)
        occupied_cells = np.zeros(map_shape)
        ################## Bug of the year was found here ##############
        pose_x = np.ceil((pose[0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        pose_y = np.ceil((pose[1] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1

      
        for scan in range(xis.shape[1]):
            if(yis[0,scan] >  MAP['sizey'] or xis[0,scan] >  MAP['sizex']) :
                pass
            else:
            
              free_cells = cv2.line(free_cells,(pose_x,pose_y),(xis[0,scan],yis[0,scan]),color = 1,thickness=1)
              occupied_cells[yis[0,scan],xis[0,scan]] = 1

      
        sensor_confidence = np.full(map_shape,np.log(lidar_confidence))
        MAP["log_odds"] += 2 * occupied_cells * sensor_confidence #Because we are subtracting one value at occupied free cell
        MAP["log_odds"] = MAP['log_odds'] - free_cells * sensor_confidence  #Because we are subtracting one value at occupied free cell
        MAP["log_odds"] = np.where(MAP["log_odds"] > log_thrs, np.full(map_shape,log_thrs),MAP["log_odds"])
        MAP["log_odds"] = np.where(MAP["log_odds"] < -log_thrs, np.full(map_shape,-log_thrs),MAP["log_odds"])
        MAP['x_loc'].append(pose_x)
        MAP['y_loc'].append(pose_y)
        MAP['theta'].append(pose[2])

    ################# Plot ################
    plot_en_ = False
    if(plot_en):
          fig = plt.figure(figsize=(18,10))
          map_threshold = np.zeros((map_shape[0],map_shape[1],3))
          map_threshold[:,:,0] = np.where(MAP['log_odds'] <= pix_thrs,  np.full(map_shape,1),np.zeros(map_shape))
          map_threshold[:,:,1] = np.where(MAP['log_odds'] <= pix_thrs,  np.full(map_shape,1),np.zeros(map_shape)) 
          map_threshold[:,:,2] = np.where(MAP['log_odds'] <= pix_thrs,  np.full(map_shape,1),np.zeros(map_shape)) 
          plt.scatter(MAP['x_loc'],MAP['y_loc'],s=0.1,c='r')
          plt.scatter(pose_x,pose_y,s=1,c='b')

          img_name = '/datasets/home/94/594/adkulkar/sensing/SLAM/images/part_20_threshold_50/' + 'img_' + str(cur_scan) + '.png'
          title_name = "Occupancy Map" + "Scan Number : " + str(cur_scan)
          plt.title(title_name)
          plt.xlabel("x")
          plt.ylabel("y")
          plt.imshow(map_threshold)
          plt.savefig(img_name)
          plt.show()
          
    if(update_log_odds == True):
        return MAP
    
    ####### Perform Correlation #######
    
    Y = np.vstack((xs0,ys0))
      
    map_threshold = np.where(MAP['log_odds'] >= -pix_thrs, np.ones(map_shape),np.zeros(map_shape))
    correlation = mapCorrelation(map_threshold,x_im,y_im,Y,x_range,x_range)
    
    return MAP,correlation
  
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
      

    body_pose = head2body_pose @ lid2head_pose @ lidar_scan
    
    ###### Filtering Out Ground Points ######
    body_pose_T = body_pose.T
    body_pose_corrected = body_pose_T[np.where(body_pose[2,:] > 0)]
    body_pose_corrected = body_pose_corrected.T
    
    return body2world_pose @ body_pose_corrected 




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

