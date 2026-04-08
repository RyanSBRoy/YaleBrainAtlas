import numpy as np
import pandas as pd
from collections.abc import Iterable
import pyvista as pv
import trimesh
import torch
import scipy
from scipy.spatial import cKDTree
from collections import Counter, defaultdict
import pickle
from itertools import compress
import numbers
import os
import sys

from YaleBrainAtlas.src.attributes import BrainAttribute, MapProxy, ListProxy, TensorProxy, ArrayProxy

INTRINSIC_TYPES = (numbers.Number, str)
root = os.path.dirname(os.path.abspath(__file__))

class Tract:
    def __init__(self, name, yba, root=root):
        super().__setattr__('name', name)
        super().__setattr__('yba', yba)

        # We store the path to the data, but we don't load it
        super().__setattr__('_data_path', os.path.join(root, '..', 'data', f'WhiteMatterTracts/{name}/'))
        
    def __getattr__(self, name):
        #Lazy loader: Only loads files or runs spatial math when called.
        if not os.path.exists(self._data_path):
            raise FileNotFoundError("Tract Path Does Not Exist, please initialize tract with root directory")
        #GEOMETRY
        if name == 'poly':
            val = pv.read(f'{self._data_path}polydata.vtp')
            super().__setattr__('poly', val)
            return val
            
        if name == 'tract_tree':
            val = cKDTree(self.poly.points)
            super().__setattr__('tract_tree', val)
            return val

        #COORDINATES AND OFFSETS
        if name in ('coords', 'offsets', 'affine_mni'):
            with open(f'{self._data_path}tractData.pkl', 'rb') as file:
                tract_data = pickle.load(file)
                super().__setattr__('affine_mni', tract_data['affine_mni'])
                super().__setattr__('coords', tract_data['points'])
                super().__setattr__('offsets', tract_data['offsets'])
            return self.__dict__[name]

        #SPATIAL CALCULATION
        if name == 'start_parcel_objs':
            # This converts raw labels into actual Parcel Objects
            names = self.get_start_and_end_parcels()['start']
            objs = [getattr(self.yba, n) for n in names]
            super().__setattr__(name, objs)
            return objs

        if name == 'end_parcel_objs':
            names = self.get_start_and_end_parcels()['end']
            objs = [getattr(self.yba, n) for n in names]
            super().__setattr__(name, objs)
            return objs

        if name == 'fiber_parcel_map':
            # Run the calculation once and cache it
            val = self.get_fiber_parcel_map(R=4)
            super().__setattr__('fiber_parcel_map', val)
            return val
            
        raise AttributeError(f"Tract '{self.name}' has no attribute '{name}'")
    
    def get_nearest_parcel(self):
        tract = self.poly 
        tract_points = self.coords.T if self.coords.shape[0] == 3 else self.coords
        brain_mesh = self.yba.whole_brain # Using the pv object
        labels = self.yba.parcel_labels
        tree = self.yba.brain_tree

        distances, indices = tree.query(tract_points)
        
        parcels_nearest = labels[indices]
        nearest_parcel_objs = [self.yba.__dict__[pcl] for pcl in parcels_nearest]
        super().__setattr__('nearest_parcels', nearest_parcel_objs)

        return self.__dict__['nearest_parcels']

    def get_fiber_parcel_map(self, R=4):
        """
        Returns a dictionary mapping {fiber_id: [ParcelObject1, ParcelObject2, ...]}
        for all parcels within radius R of each specific fiber.
        """
        tract = self.poly 
        brain_mesh = self.yba.whole_brain # Using the pv object
        labels = self.yba.parcel_labels
        
        # Map each point in the polydata to its parent cell
        point_to_fiber = np.empty(tract.n_points, dtype=int)
        for i in range(tract.n_cells):
            point_to_fiber[tract.get_cell(i).point_ids] = i

        #Query the KDTree for brain points near the tract
        #near_indices[i] contains list of tract-point indices near brain-point i
        near_indices = self.tract_tree.query_ball_point(brain_mesh.points, r=R)

        #Associate Brain Indices with Fiber IDs
        fiber_to_parcel_names = {}
        
        for brain_idx, tract_pt_indices in enumerate(near_indices):
            if tract_pt_indices:
                #Which fiber does this brain point belong to?
                unique_fibers = np.unique(point_to_fiber[tract_pt_indices])
                parcel_name = labels[brain_idx]
                
                for f_id in unique_fibers:
                    if f_id not in fiber_to_parcel_names:
                        fiber_to_parcel_names[f_id] = set()
                    fiber_to_parcel_names[f_id].add(parcel_name)
        
        #Convert names to actual Parcel Objects
        fiber_to_parcel_objs = {
            f_id.item(): [getattr(self.yba, name) for name in names]
            for f_id, names in fiber_to_parcel_names.items()
        }
        
        return fiber_to_parcel_objs

    def _calculate_endpoints(self, bracket):
        # tract_pts is N x 3
        tract_pts = self.coords.T if self.coords.shape[0] == 3 else self.coords
        offsets = self.offsets[0]
        
        # Identify the indices for the 'bracket' points at start and end
        # if bracket=3, we take points [0,1,2] and [end-3, end-2, end-1]
        s_indices = np.concatenate([offsets[:-1] + i for i in range(bracket)])
        e_indices = np.concatenate([offsets[1:] - (i+1) for i in range(bracket)])

        # Query the YBA brain tree for the closest parcel points
        s_dist, s_bin = self.yba.brain_tree.query(tract_pts[s_indices])
        e_dist, e_bin = self.yba.brain_tree.query(tract_pts[e_indices])

        return {
            's_dist': s_dist, 
            's_names': self.yba.parcel_labels[s_bin],
            'e_dist': e_dist, 
            'e_names': self.yba.parcel_labels[e_bin],
            'fiber_ids': np.arange(len(offsets) - 1),
            'bracket': bracket
        }

    def _filter_endpoints(self, endpoint_data, radius):
        #This function has an issue in that it only includes the fiber if all points in the bracket are within radius
        #This is silly because the idea of adjusting the bracket is that you are getting more parcels and capturing other potential endpoints
        valid = {'start': [], 'end': [], 'fiber_id': [], 'pairs': []}
        bracket = endpoint_data['bracket']

        for i in range(len(endpoint_data['fiber_ids']) - 1):
            #check if all sampled points for this fiber are within radius
            placement = i * bracket
            start_points = endpoint_data['s_dist'][placement: placement + bracket]
            end_points = endpoint_data['e_dist'][placement: placement + bracket]

            for j in range(bracket):
                if (start_points[j] < radius) and (end_points[j] < radius):
                    s_parcel = endpoint_data['s_names'][placement + j].item()
                    e_parcel = endpoint_data['e_names'][placement + j].item()

                    valid['pairs'].append((s_parcel, e_parcel))
                    valid['start'].append(s_parcel)
                    valid['end'].append(e_parcel)
                    valid['fiber_id'].append(endpoint_data['fiber_ids'][i])

        # for i, f_id in enumerate(endpoint_data['fiber_ids']):
        #     # Slice the bracket of points for this specific fiber
        #     start_slice = endpoint_data['s_dist'][i::len(endpoint_data['fiber_ids'])]
        #     end_slice = endpoint_data['e_dist'][i::len(endpoint_data['fiber_ids'])]
            
        #     # Condition: All points in the bracket must be within radius
        #     if np.all(start_slice < radius) and np.all(end_slice < radius):
        #         # Use the very first/last point as the definitive label
        #         s_name = endpoint_data['s_names'][i]
        #         e_name = endpoint_data['e_names'][i]
                
        #         valid['start'].append(s_name.item())
        #         valid['end'].append(e_name.item())
        #         valid['fiber_id'].append(f_id.item())
        #         valid['pairs'].append((s_name.item(), e_name.item()))
                
        return valid

    def get_start_and_end_parcels(self, bracket=3, radius=4):
        endpoints = self._calculate_endpoints(bracket)
        filtered = self._filter_endpoints(endpoints, radius)
        
        #create and/or update tract count connectivity in the global atlas object
        #ensure 'tract_count' exists as a Connectivity attribute first
        if not hasattr(self.yba, 'tract_count'):
            self.yba._create_brain_attribute('tract_count', BrainAttribute.Connectivity) #creates an empty tract_count attribute in YBA

        counts = Counter(filtered['pairs'])
        for (src, tgt), weight in counts.items():
            self.yba.tract_count.at[src, tgt] += weight
        
        #creates registry of intersecting tracts per parcel as a global atlas object, if it doesn't already exist
        for reg_name in ['_tract_starts', '_tract_ends']:
            if not hasattr(self.yba, reg_name):
                # Use GroupDict category because we are using Parcel Names as Keys
                self.yba._create_brain_attribute(reg_name, BrainAttribute.GroupDict)
                # Use super to avoid triggering Atlas __setattr__ logic during setup
                super(self.yba.__class__, self.yba).__setattr__(reg_name, {p: {t: [] for t in self.yba.tract_names} for p in self.yba.parcel_names})

        #pulls registry from YBA and starts filling it in
        starts = getattr(self.yba, '_tract_starts')
        ends = getattr(self.yba, '_tract_ends')

        for i, f_id in enumerate(filtered['fiber_id']):
            # Ensure we use .item() if these are numpy types to keep the dicts clean
            s_name, e_name = filtered['start'][i], filtered['end'][i]
            starts[s_name][self.name].append(f_id.item()) #.item() gets rid of the int64() tag, which isn't really needed here
            ends[e_name][self.name].append(f_id.item())

            #starts[s_name].append({'tract': self.name, 'fiber': f_id})
            #ends[e_name].append({'tract': self.name, 'fiber': f_id})

        # signal that adjustments were made for certain parcels in the global registry, so that if tract intersections are pulled for a single parcel as an attribute, the parcel knows to call the global atlas
        changed_parcels = set(filtered['start']) | set(filtered['end'])
        for p_name in changed_parcels:
            p_idx = self.yba.parcel_names.index(p_name)
            self.yba._bump_version('_tract_starts', parcel_idx=p_idx)
            self.yba._bump_version('_tract_ends', parcel_idx=p_idx)

        super().__setattr__(f'filtered_paths_{bracket}_{radius}', filtered)
        return filtered
    
    def get_path_lengths(self):
            if not hasattr(self, '_filtered_paths'):
                self.get_start_and_end_parcels()
                
            lengths = {}
            for f_id in self._filtered_paths['fiber_id']:
                cell = self.poly.get_cell(f_id)
                pts = cell.points
                # Sum of distances between consecutive points
                dist = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
                lengths[f_id] = dist
                
            super().__setattr__('path_lengths', lengths)
            return lengths

    def __repr__(self):
        return f"<Tract '{self.name}' | Fibers: {self.poly.n_cells if 'poly' in self.__dict__ else 'Not Loaded'}>"