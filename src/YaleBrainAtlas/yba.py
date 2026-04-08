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

INTRINSIC_TYPES = (numbers.Number, str)

from YaleBrainAtlas.attributes import BrainAttribute, MapProxy, ListProxy, TensorProxy, ArrayProxy
from YaleBrainAtlas.parcel import Parcel
from YaleBrainAtlas.tract import Tract

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

class YaleBrainAtlas:
    def __init__(self, name, root=root):
        #CORE IDENTITY -- name of brain
        super().__setattr__('name', name)

        #LOAD PHYSICAL DATA
        #pyvista object of whole brain (I'm considering this the surfaces mesh, it's a bit easier than having individual mesh files)
        wb = pv.read(os.path.join(root, 'data', 'YBA_surfaces_full.vtp'))
        super().__setattr__('whole_brain', wb) #load whole brain
        super().__setattr__('parcel_labels', wb['parcel_labels']) #parcel labels is used for more detailed surface labeling and color coding and things, so it's here -- but ideally I don't want to use this much
        super().__setattr__('brain_tree', cKDTree(wb.points)) #I've been doing a lot of 'closest point' searching stuff, which requires keeping the points in a cKDTree

        #NAMES AND INDICES
        #all the parcel names
        with open(os.path.join(root, 'data', "ParcelNames.txt"), "r") as file: 
            names = [line.rstrip('\n') for line in file]
        indices = list(range(len(names)))
        #parcel names
        super().__setattr__('parcel_names', names)

        #remove the corpus callosum from parcel names -- good to have as a reference
        names_noCC = np.array(names)[np.where(~pd.Series(names).str.contains('CC'))].tolist()
        super().__setattr__('parcel_names_noCC', names_noCC)

        #parcel index
        super().__setattr__('idx', indices) #initializes the YBA with 696 parcels

        #MAIN ATTRIBUTES INFRASTRUCTURE
        #attributes dictionary, which is a dictionary of attribute names and their types
        #attributes needs to be defined first (or almost first) because almost every other attribute in the YBA relies on it
        super().__setattr__('attributes', {
                'mesh': BrainAttribute.Mesh,
                'Number': BrainAttribute.Intrinsic
                })

        #version registry: tracks version of every global attribute (for stale cache fixes)
        #initialized explicity so _bump_version doesn't create one
        super().__setattr__('_attr_versions', {k: np.zeros(len(names), dtype=int) for k in ['mesh', 'Number']})

        #DATA STORAGE
        #parcel parameters dataframe with all the intrinsic parcel attributes
        super().__setattr__('parcel_parameters', pd.DataFrame(names, columns=['Number'], index=names))
        self.parcel_parameters['Number'] = list(range(696))

        #PARCEL OBJECTS
        #creates each parcel attribute with the corresponding parcel name
        for name in names:
            super().__setattr__(name, Parcel(name, self))

        #TRACT OBJECTS
        with open(os.path.join(root, 'data', "TractNames.txt"), "r") as file_t:
            tractNames = [line.rstrip('\n') for line in file_t]
        super().__setattr__('tract_names', tractNames)
        for name_t in tractNames:
            super().__setattr__(name_t, Tract(name_t, self, root))

        #TEMPLATES, ATTRIBUTE STORAGE, AND ATTRIBUTE TYPE HANDLING
        #attribute base module/blocks
        super().__setattr__('_all_parcel_dict', {pcl : None for pcl in names})
        super().__setattr__('_all_parcel_list', [None for pcl_id in indices])
        super().__setattr__('_all_parcel_zero_list', [0 for pcl_id in indices])
        super().__setattr__('_adjacency', pd.DataFrame(np.zeros([len(names), len(names)]), columns=names, index=names))
        super().__setattr__('_nparr', np.zeros([len(names)]))
        super().__setattr__('_torchTensor', torch.zeros([len(names)]))

        #attribute storage
        super().__setattr__('_attribute_type_map', {
            BrainAttribute.Other: self._all_parcel_dict.copy(),
            BrainAttribute.Connectivity: self._adjacency.copy(),
            BrainAttribute.GroupDict: self._all_parcel_dict.copy(),
            BrainAttribute.Group: self._all_parcel_list.copy(),
            BrainAttribute.MatrixNP: self._nparr.copy(),
            BrainAttribute.MatrixTensor: self._torchTensor.clone(),
            BrainAttribute.Mesh: trimesh.Trimesh(),
            BrainAttribute.NONE: None
        })

        super().__setattr__('_type_attribute_map', 
                          {type(brain_attribute_val): brain_attribute for brain_attribute, brain_attribute_val in self._attribute_type_map.items() if brain_attribute is not BrainAttribute.Other})

    
    def __str__(self):
        return (f"{self.name}, wb, parcel_labels, parcel_names, brain_tree, idx, attributes, parcel_parameters")
    
    def __repr__(self):
        # Filter out private attributes and format the rest
        attrs = ", ".join(
            f"{k}={v!r}" 
            for k, v in self.__dict__.items() 
            if not k.startswith("_")
        )
        return f"{self.__class__.__name__}({attrs})"
    
    def _infer_global_category(self, att_value):
        #A different set of conditionals than for parcels, since these are global, so they're meant to store ALL the parcel attributes
        att_type = None
        val_type = type(att_value)
        #The BrainIntrinsic type is a number, or a string. But not a list or iterable. If it's an iterable, and not of parcel length, then it should be other
        if isinstance(att_value, INTRINSIC_TYPES) or not isinstance(att_value, Iterable):
            att_type = BrainAttribute.BrainIntrinsic
        elif val_type in list(self._type_attribute_map):
            if val_type is pd.DataFrame:
                if att_value.shape == (len(self.parcel_names), len(self.parcel_names)):
                    #If the pandas dataframe is 696x696, for instance, we'll treat it as connectivity, since that is usually what it is
                    #This makes connectivity a narrow definition of inter-parcel connectivity
                    #There can be inter-region connectivity or inter-hemispheric connectivity, which would be stored as a BrainAttribute.Other type.
                    att_type = BrainAttribute.Connectivity
                elif (att_value.shape == (len(self.parcel_names), 1)) and (set(att_value.index) == set(self.parcel_names)):
                    att_type = BrainAttribute.Intrinsic

            elif val_type is dict:
                #if what's in the dictionary is a list or a set, and the keys match with the atlas parcel labels
                if all(isinstance(att_v_type, (list, set)) for att_v_type in att_value.values()) and (set(list(att_value)) == set(self.parcel_names)):
                    att_type = BrainAttribute.GroupDict

                #if we have a situation where the dictionary has keys corresponding to atlas parcel labels, and there's only one value per parcel
                elif all(len(att_v == 1) for att_v in att_value.values()) and (len(att_value) == len(self.parcel_names)) and (set(list(att_value)) == set(self.parcel_names)):
                    att_type = BrainAttribute.Intrinsic
                
                else:
                    att_type = BrainAttribute.Other
            
            elif val_type in (list, set):
                print(val_type)
                #if the length of the list or set is the same as that of the parcel_names, we assume each element corresponds to a parcel
                if len(att_value) == len(self.parcel_names):
                    att_type=BrainAttribute.Intrinsic

            elif val_type in (np.ndarray, torch.Tensor) and (att_value.shape[0] != len(self.parcel_names)):
                #if the np.ndarray or torch.Tensor is not parcels x d1 x d2 x d3 x ... then it's not a torch tensor or np.array Brain Attribute
                att_type = BrainAttribute.Other
            
            elif val_type in (np.ndarray, torch.Tensor) and (att_value.shape == (len(self.parcel_names),)) or (att_value.shape == (len(self.parcel_names), 1)):
                #we have a numpy nd array or torch tensor, but there's only one value per parcel
                #no point in storing that as a nd array
                #if you want to store that as something separate, you can take the transpose, and then it'll be (, 696) and that'll be a nparray
                att_type = BrainAttribute.Intrinsic

            else:
                att_type = self._type_attribute_map[val_type]

        elif val_type in (list, set, pd.Series):
            if len(att_value) == len(self.parcel_names):
                att_type=BrainAttribute.Intrinsic
            
        else:
            att_type = BrainAttribute.Other
         
        return att_type

    def __getattr__(self, att_name):
        #this will be called only if it can't find the regular attribute
        #one case where this might happen is if the attribute is in parcel_parameters, so we check that, and if it's not there, then return AttributeError
        if att_name in self.attributes:
            att_categ = self.attributes[att_name]
            if att_categ is BrainAttribute.Intrinsic:
                return self.parcel_parameters[att_name]
        
        #if it's a normal attribute (MatrixNP, dict, etc.), Python's default getattribute would find it before calling getattr
        
        raise AttributeError(f"Attribute {att_name} could not be found")
        
    
    def _create_brain_attribute(self, att_name, category):
        if category is BrainAttribute.Intrinsic: #Intrinsic is the exception, where it must be placed in the parcel_parameters dataframe
            self.parcel_parameters[att_name] = None #might have to directly handle the None cases when plotting and things, but for now I think it's okay
    
        elif category is BrainAttribute.Mesh:
            #create the multiblock for individual parcel meshes
            super().__setattr__(att_name, self._all_parcel_dict.copy())
            super().__setattr__(f"{att_name}_wb", pv.PolyData())
        #initializes global storage container for new attribute of the atlas
        elif category in list(self._attribute_type_map):
            #I thought about creating an igraph object for each connectivity graph, but I think this has complications in that I may inadvertently limit the kind of connectivity graphs I create
            #i.e. distinguishing between unweighted and weighted graphs, etc. etc.
            template = self._attribute_type_map[category]
            val_att = template.copy() if hasattr(template, 'copy') else template
            super().__setattr__(att_name, val_att)
        
        if att_name not in self._attr_versions:
            self._attr_versions[att_name] = np.zeros(len(self.parcel_names), dtype=int)

        self.attributes[att_name] = category     
        return

    def __setattr__(self, att_name, att_value): #not yet sure if this is necessary
        #if we already know what the attribute is, just update the attribute
        if att_name in self.attributes:
            category = self.attributes[att_name]
        else:
            category = self._infer_global_category(att_value)
            self.attributes[att_name] = category
        
        if category is BrainAttribute.Intrinsic:
            if isinstance(att_value, pd.Series):
                att_value = att_value.values.tolist()
            self.parcel_parameters[att_name] = att_value

        else:
            super().__setattr__(att_name, att_value)
            if category is BrainAttribute.Mesh and att_name != "mesh":
                # Assuming att_value is a dict of trimeshes/pyvista objects
                # We need to convert them and merge into one 'wb' object
                # Note: PyVista MultiBlocks have a .combine() method
                combined = att_value.combine(merge=True) if hasattr(att_value, 'combine') else att_value
                super().__setattr__(f"{att_name}_wb", combined)

                #make sure to create the whole brain version of this
        
        self.attributes[att_name] = category
        self._bump_version(att_name) #bump version on direct set
        
        return

    def _bump_version(self, att_name, parcel_idx=None):
        #triggering __getattr__ during init risks leading to recursions and crashes and things, so I'm calling versions explicity from the yba dictionary
        versions = self.__dict__.get('_attr_versions')

        if versions is None:
            #if we're in the middle of init and it doesn't exist yet, just skip
            return
        
        #signals to specific parcel that this specific attribute has changed
        if att_name not in versions:
            versions[att_name] = np.zeros(len(self.parcel_names), dtype=int)
        
        if parcel_idx is not None:
            #only if this parcel is marked as stale
            self._attr_versions[att_name][parcel_idx] += 1
        else:
            #mark whole brain as stale
            self._attr_versions[att_name] += 1
        
    
    def set_bulk_data(self, att_name, values):
        #vectorized setter, updates all parcels at once without looping through parcel objects
        category = self.attributes.get(att_name)

        #if the category is unknown, register it
        if category is None:
            category = self._infer_global_category(values)
            self._create_brain_attribute(att_name, category)
            
        if category is BrainAttribute.Intrinsic:
            #vectorized pandas update
            self.parcel_parameters[att_name] = values
        elif category in (BrainAttribute.MatrixNP, BrainAttribute.MatrixTensor):
            super().__setattr__(att_name, values)
        
        elif category in (BrainAttribute.Group, BrainAttribute.GroupDict, BrainAttribute.Other, BrainAttribute.Mesh):
            #I think mesh is stored as both a Group Mesh (dictionary) as well as the pyvista object, but usually at the atlas level we only care about the group mesh
            #the pyvista object is handled internally for, like, plotting and things
            super().__setattr__(att_name, values)
        
        elif category is BrainAttribute.Connectivity:
            # values should be a DataFrame (696x696) or a 2D array
            if isinstance(values, pd.DataFrame):
                super().__setattr__(att_name, values)
            else:
                # If it's a raw numpy array, wrap it in a DataFrame for consistency
                attribute_dataframe = pd.DataFrame(values, columns=self.parcel_names, index=self.parcel_names)
                super().__setattr__(att_name, attribute_dataframe)
        
        self._bump_version(att_name)
    
    def find_parcel_at_coord(self, coords):
        #coords is an array like (3,) or (N, 3)
        #returns the parcel object(s) closest to the input coordinates

        distance, index = self.brain_tree.query(coords)
        parcel_name = self.whole_brain.point_data['parcel_labels'][index]

        if isinstance(parcel_name, str): #this would be if we're dealing with only one parcel
            return {'parcel': getattr(self, parcel_name), 
                    'distance': distance}
        else: #in the case of multiple parcels. I might prefer to be a bit more explicit later on, but this is OK
            return {'parcels': [getattr(self, name) for name in parcel_name],
                    'distances': distance}