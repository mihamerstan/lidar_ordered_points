import numpy as np
import torch
import torchvision
import torch.nn.functional as F

import pandas as pd
from laspy.file import File
from pickle import dump, load

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as udata
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler


# INPUT PARAMETERS
file_dir = "../../lidar/Data/parking_lot/"
filename = "first_returns_modified_164239.pkl"
# Training Data parameters
scan_line_gap_break = 7000 # threshold over which scan_gap indicates a new scan line
min_pt_count = 1730 # in a scan line, otherwise line not used
max_pt_count = 2000 # in a scan line, otherwise line not used
num_scan_lines = 100 # to use as training set
starting_line=1000
val_split = 0.2
seq_len = 16

# Angle range of considered points (deg /0.006)
starting_angle = 4500
ending_angle = -4500

# points in between scan lines
stride_inline = 2
stride_across_lines = 1

# Note: x_scaled, y_scaled, and z_scaled MUST be the first 3 features and miss_pts_before MUST be the last feature
feature_list = [
    'x_scaled',
    'y_scaled',
    'z_scaled',
    'scan_line_idx',
    'scan_angle_deg',
    'abs_scan_angle_deg',
    'miss_pts_before'
]

def add_missing_pts(first_return_df):
    # Create a series with the indices of points after gaps and the number of missing points (no max)
    miss_pt_ser = first_return_df[first_return_df['miss_pts_before']>0]['miss_pts_before']
    # miss_pts_arr is an array of zeros that is the dimensions [num_missing_pts,cols_in_df]
    miss_pts_arr = np.zeros([int(miss_pt_ser.sum()),first_return_df.shape[1]])
    # Create empty series to collect the indices of the missing points
    indices = np.ones(int(miss_pt_ser.sum()))

    # Fill in the indices, such that they all slot in in order before the index
    i=0
    for index, row in zip(miss_pt_ser.index,miss_pt_ser):
        new_indices = index + np.arange(row)/row-1+.01
        indices[i:i+int(row)] = new_indices
        i+=int(row)
    # Create a Dataframe of the indices and miss_pts_arr
    miss_pts_df = pd.DataFrame(miss_pts_arr,index=indices,columns = first_return_df.columns)
    miss_pts_df['mask'] = [0]*miss_pts_df.shape[0]
    miss_pts_df['miss_pts_before'] = -1
    # Fill scan fields with NaN so we can interpolate them
    for col in ['scan_angle','scan_angle_deg']:
        miss_pts_df[col] = [np.NaN]*miss_pts_df.shape[0]
    # Concatenate first_return_df and new df
    full_df = first_return_df.append(miss_pts_df, ignore_index=False)
    # Resort so that the missing points are interspersed, and then reset the index
    full_df = full_df.sort_index().reset_index(drop=True)
    # Interpolate the scan angles
    full_df[['scan_angle','scan_angle_deg']] = full_df[['scan_angle','scan_angle_deg']].interpolate()
    # Fill miss_pts_before with -1 so infilled points can be identified
    
    return full_df

def create_scan_line_tensor(file_dir=file_dir,filename=filename,\
		feature_list = feature_list, min_pt_count = min_pt_count, \
		max_pt_count=max_pt_count, num_scan_lines = num_scan_lines, \
		starting_line = starting_line, starting_angle=starting_angle, \
		ending_angle=ending_angle):
	first_return_df = pd.read_pickle(file_dir+filename)

	# miss_pts_before is the count of missing points before the point in question (scan gap / 5 -1)
	first_return_df['miss_pts_before'] = round((first_return_df['scan_gap']/-5)-1)
	first_return_df['miss_pts_before'] = [max(0,pt) for pt in first_return_df['miss_pts_before']]
	print("miss pts min: ", first_return_df['miss_pts_before'].min())
	print("miss pts max: ", first_return_df['miss_pts_before'].max())
	# Add 'mask' column, set to one by default
	first_return_df['mask'] = [1]*first_return_df.shape[0]

	# Add abs_scan_angle_deg as a feature
	first_return_df['abs_scan_angle_deg'] = abs(first_return_df['scan_angle_deg'])

	# Number of points per scan line
	scan_line_pt_count = first_return_df.groupby('scan_line_idx').count()['gps_time']

	# Remove scan lines outside the point count range from first_return_df
	valid_scan_line_idx = scan_line_pt_count[(scan_line_pt_count>min_pt_count) * (scan_line_pt_count<max_pt_count)].index

	# Only the points that are in valid scan lines
	first_return_df_valid = first_return_df[first_return_df['scan_line_idx'].isin(valid_scan_line_idx)]

	# Fill in missing points``
	first_return_df_valid = add_missing_pts(first_return_df_valid)

	# Now remove lines that don't have 1730 points between -27 and 27 degrees
	# Number of points per scan line
	scan_line_pt_count = first_return_df_valid.groupby('scan_line_idx').count()['gps_time']

	# Remove scan lines outside the point count range from first_return_df
	valid_scan_line_idx = scan_line_pt_count[scan_line_pt_count>min_pt_count].index

	# Only the points that are in valid scan lines
	first_return_df_valid = first_return_df_valid[first_return_df_valid['scan_line_idx'].isin(valid_scan_line_idx)]

	# Indices for the point closes to starting_angle in each scan line
	starting_idx = [abs(first_return_df_valid[first_return_df_valid['scan_line_idx']==line_idx] \
	     ['scan_angle']-starting_angle).idxmin() for line_idx in first_return_df_valid['scan_line_idx'].unique()]

	# Remove the nan idx corresponding to zero scan line
	starting_idx = [x for x in starting_idx if str(x) != 'nan']

	# Create Tensor
	scan_line_tensor = torch.randn([len(starting_idx),1737,len(feature_list)])
	# Loop thru scan lines
	for line,line_idx in enumerate(starting_idx):
	        # Fill the appropriate line in scan_line_tensor
	        name = first_return_df_valid.iloc[line_idx].name
	        try:
	            scan_line_tensor[line,:,:] = torch.Tensor(first_return_df_valid.loc\
	                                      [name:name+1736][feature_list].values)
	        except RuntimeError:
	            print("Not enough points in line {}".format(line))
	return scan_line_tensor


######################################################################################################
#### 2D Baseline Functions

# Outer Loop Functions
def find_top_left_mask(im_m):
    # Given a [7,64,64] mask matrix, identify the xy coordinates of the top-left point of the mask
    for j in np.arange(im_m.shape[1]): # Loop thru rows, find first masked row
        idx = np.arange(im_m.shape[2])[im_m[0,j,:]==0]
        if len(idx)>0:
            mask_top_row = j
            mask_first_col = idx.min()
#             print("Mask starts at ({},{})".format(mask_top_row,mask_first_col))
            break
    return mask_top_row,mask_first_col


# Inner Loop Functions
def surrounding(im_o,im_m,pt,n):
    ''' Gather surrounding points that are not masked
    im_o - masked, clean matrix
    im_m - mask matrix
    pt - [x,y] of grid point to use
    n - number of points in each direction to consider as surrounding
    '''
    surrounding_pts = im_o[:,pt[0]-n:pt[0]+n+1,pt[1]-n:pt[1]+n+1]
    surrounding_no_mask = im_m[0,pt[0]-n:pt[0]+n+1,pt[1]-n:pt[1]+n+1] != 0
    return surrounding_pts, surrounding_no_mask


def surrounding_grid(surrounding_pts,surrounding_no_mask):
    ''' Reshape surrounding_pts into list of [x_grid,y_grid,x,y,z] arrays
        surrounding_pts - all the surrounding points, regardless of mask
        surrounding_no_mask - True/False matrix indicating which points are masked
        mat_pt_list - list of 1x5 numpy arrays
    '''
    mat = surrounding_pts
    mat_pt_list = []
    for i in range(surrounding_pts.shape[1]):
        for j in range(surrounding_pts.shape[2]):
            if surrounding_no_mask[i,j]:
                mat_pt_list.append([i,j,surrounding_pts[0,i,j].item(), \
                                        surrounding_pts[1,i,j].item(), \
                                        surrounding_pts[2,i,j].item()])
                # mat_pt_list is [x_grid,y_grid,x,y,z] for each point
    return mat_pt_list

def plane_fit(mat_pt_list):
    raw_points = np.array(mat_pt_list).T
    points = raw_points.T - raw_points.mean(axis=1)
    norm_vector_dict = {} # x_norm, y_norm, z_norm
    pt_on_plane_dict = {} # x_pt, y_pt, z_pt

    dim_dict = {"x":2,"y":3,"z":4}
    for key in dim_dict:
        # SVD for each xyz dim at a time
        svd = np.linalg.svd(points[:,[0,1,dim_dict[key]]].T)
        norm_vector = np.transpose(svd[0])[2]    
        norm_vector_dict[key] = norm_vector
        dist_from_plane = np.dot(points[0,[0,1,dim_dict[key]]],norm_vector)
        proj_on_norm = dist_from_plane*np.array([norm_vector]).T
        pt_on_plane = raw_points[[0,1,dim_dict[key]],0] - proj_on_norm[:,0]
        pt_on_plane_dict[key] = pt_on_plane
    return norm_vector_dict, pt_on_plane_dict
        
def interpolate_grid(norm_vector_dict,pt_on_plane_dict,n):
    # Find the target point on the plane
    # dim_dict = {"x":2,"y":3,"z":4}
    pt = [n,n] # Interpolated point is at center of grid
    interp = {}
    for key in ['x','y','z']:
        a,b,c = norm_vector_dict[key]
        x0,y0,z0 = pt_on_plane_dict[key]
        interp[key] = (1/c)*(a*(x0 - pt[0])+b*(y0-pt[1])+c*z0)    
    return interp

def inner_interp_loop(im_o,im_m,pt,n):
    # Input: For a point pt, interpolate based on surrounding points
    surrounding_pts, surrounding_no_mask = surrounding(im_o,im_m,pt,n)
    mat_pt_list = surrounding_grid(surrounding_pts,surrounding_no_mask)
    # Fit plane for each dimension to mat_pt_list 
    norm_vector_dict, pt_on_plane_dict = plane_fit(mat_pt_list)
    interp = interpolate_grid(norm_vector_dict,pt_on_plane_dict,n)
    # Update im_c and im_m with new point
    im_o[:3,pt[0],pt[1]] = torch.Tensor([interp[k] for k in ['x','y','z']])
    im_m[:3,pt[0],pt[1]] = torch.Tensor([1.,1.,1.])
    
    return im_o, im_m

def outer_interp_loop(c,m,mask_pts_per_seq,n=2):
    # Note this only works for consecutive missing points
    out = c*m
    for im_idx in range(c.shape[0]): # loop thru the batch
        im_c = c[im_idx]
        im_m = m[im_idx].clone()
        im_o = out[im_idx]
        # Circle in algorithm
        # Given an im_c and im_m, identify mask location, interpolate all points
        # Set surrounding distance n

        start_pt = find_top_left_mask(im_m)

        for ml in np.arange(mask_pts_per_seq,0,-2):    
            if ml == 1:
                pt = start_pt[0],start_pt[1]
                im_o, im_m = inner_interp_loop(im_o,im_m,pt,n)
            # Top row
            for j in np.arange(start_pt[1],start_pt[1]+ml-1):
                i_c = start_pt[0]
                pt = [i_c,j]
                im_o, im_m = inner_interp_loop(im_o,im_m,pt,n)
            # Right col
            for i in np.arange(start_pt[0],start_pt[0]+ml-1):
                j_c = start_pt[1]+ml-1
                pt = [i,j_c]
                im_o, im_m = inner_interp_loop(im_o,im_m,pt,n)
            # Bottom row
            for j in np.arange(start_pt[1]+ml-1,start_pt[1],-1):
                i_c = start_pt[0]+ml-1
                pt = [i_c,j]
                im_o, im_m = inner_interp_loop(im_o,im_m,pt,n)
            # Left col
            for i in np.arange(start_pt[0]+ml-1,start_pt[0],-1):
                j_c = start_pt[1]
                pt = [i,j_c]
                im_o, im_m = inner_interp_loop(im_o,im_m,pt,n)
            # Move the starting point in one ring
            start_pt = [start_pt[i]+1 for i in [0,1]]
        out[im_idx] = im_o
    return out