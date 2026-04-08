import numpy as np
import pandas as pd
from collections import deque
from collections.abc import Iterable
import trimesh
import torch
import numbers
import os
import sys

from YaleBrainAtlas.src.attributes import BrainAttribute, MapProxy, ListProxy, TensorProxy, ArrayProxy

INTRINSIC_TYPES = (numbers.Number, str)

class Parcel:
    def __init__(self, name, yba, **kwargs):
        #when the object is being initialized, we use super().__setattr__ to set the initial attributes without calling the defined setattr
        super().__setattr__('name', name)
        super().__setattr__('yba', yba)
        super().__setattr__('idx', self.yba.parcel_names.index(self.name))
        super().__setattr__('attributes', {})
        super().__setattr__('_cache_versions', {})
    
    def _infer_category(self, att_value):
        #infers what parcel attribute type/category is being stored in the parcel
        if att_value is None:
            category = BrainAttribute.NONE
        elif isinstance(att_value, np.ndarray):
            category = BrainAttribute.MatrixNP
        elif isinstance(att_value, torch.Tensor):
            category = BrainAttribute.MatrixTensor
        elif isinstance(att_value, (list, set)): #like a list of lists for EEG coordinates or something
            category = BrainAttribute.Group
        elif isinstance(att_value, trimesh.Trimesh):
            category = BrainAttribute.Mesh
        elif isinstance(att_value, INTRINSIC_TYPES) or not isinstance(att_value, Iterable):
            category = BrainAttribute.Intrinsic
        elif isinstance(att_value, dict) and (len(att_value) == len(self.yba.parcel_names)): #it's a adjacency map for that parcel
            category = BrainAttribute.Connectivity
        elif isinstance(att_value, dict):
            category = BrainAttribute.GroupDict #a dictionary where the keys are parcel names
        else:
            category = BrainAttribute.Other #pandas dataframe, polydatas, etc.
        return category
    
    def _wrap_in_proxy(self, data, attr_name, category):
        #Wraps raw atlas data in a proxy to ensure two-way syncing.
        if category is BrainAttribute.Connectivity or category is BrainAttribute.GroupDict: #GroupDict is always going to be a dict
            return MapProxy(data, self, attr_name)
        
        elif category is BrainAttribute.MatrixNP:
            return ArrayProxy(data, self, attr_name)
        
        elif category is BrainAttribute.MatrixTensor:
            return TensorProxy(data, self, attr_name)
        
        elif category is BrainAttribute.Group:
            return ListProxy(data, self, attr_name)
        
        # Intrinsic (strings/ints) or Mesh (objects) don't get proxies. Other PROBABLY doesn't get a proxy, but it's hard to say. 
        return data

    def _update_yba(self, att_name, att_value, category):
        #this is contingent on the attribute already existing in the YBA, so we don't need to check if the att_name exists in yba.attributes
        #quick check to make sure atlas categories match
        atlas_category = self.yba.attributes[att_name]
        if atlas_category is not category:
            raise AttributeError(f"Parcel attribute {att_name} category {category} does not match atlas {att_name} category {atlas_category}")
        
        #first check if the category is an intrinsic one, since this would not be accessible in the broader atlas dictionary
        if category is BrainAttribute.Intrinsic: #in the YBA, would be a pandas dataframe (parcel - row x parameters - column)
            #for the parcel, this would just be a number or a string
            atlas_attribute = self.yba.parcel_parameters
            atlas_attribute.at[self.name, att_name] = att_value
        
        else: #if the atlas attribute is not in the intrinsic parcel parameters dataframe, it'll probably be in the atlas object dictionary
            atlas_attribute = getattr(self.yba, att_name) #the atlas attribte name HAS to be the same as that of the parcel attribute name. 

        #could handle this in case-by-case basis per BrainAttribute. This would be more precise, but I think combining them is more elegant. 
        if category in (None, BrainAttribute.NONE, BrainAttribute.Group, BrainAttribute.GroupDict, BrainAttribute.Mesh, BrainAttribute.Other):
            #parcel attribute is None, list/set/dict, mesh, or anything else
            #parcel attribute is stored in the atlas as part of a dictionary, with parcel as key, and component being stored as value
            #exception for Group, which could be stored as a list of lists. In this case, we just replace the part of the nested list corresponding to parcel index
            if isinstance(atlas_attribute, (list, set)) and isinstance(att_value, (list, set)):
                if isinstance(att_value, (list, set)) is False:
                    raise AttributeError(f"Attribute {att_name} of type {category} is not a list, but is being added to a list of lists in YBA: {atlas_attribute}")
                atlas_attribute[self.idx] = att_value
            elif isinstance(atlas_attribute, dict):
                atlas_attribute[self.name] = att_value

        if category in (BrainAttribute.MatrixNP, BrainAttribute.MatrixTensor) :
            atlas_attribute[self.idx] = att_value
        
        if category is BrainAttribute.Connectivity:
            #the parcel verision of this is basically a dictionary, where the keys are the connected parcels and the values are the connectivity values
            #att_value would need to be a dict or series-like
            #conversion of the parcel dictionary version to the pandas dataframe version is done here

            for neighbor, weight in att_value.items():
                atlas_attribute.at[neighbor, self.name] = weight

        if category is BrainAttribute.Mesh:
            #this is the annoying one, because we have to convert it to a vtp whenever we update. So we pull from the MultiBlock object, and convert that to a .vtp as well
            #it's tedious and sacrifices storage space for "easier" conversions. the vtp should keep everything in the same coordinate system, which could be useful for something
            #the multiblock .vtm keeps every parcel mesh separate, which helps with parcel-level modifications, but bad for whole-brain work because the coordinate systems are different per parcel

            #the parcel input is a trimesh
            if att_name == "mesh":
                raise AttributeError(f"You cannot modify the parcel meshes of the atlas.")
            else:
                combined_meshes = getattr(self.yba, f"{att_name}_wb")
                atlas_attribute[self.name] = att_value
                combined_meshes = atlas_attribute.combine(merge=True) #combine the multiblock into single unstructured grid/polydata

        #setattr(self.yba, att_name, atlas_attribute) uncommenting this will clear the cache for all 696 parcels every time #this is sorta redundant because I'm already calling atlas_attribute and changing it. 
        if category is BrainAttribute.Mesh:
            setattr(self.yba, f"{att_name}_wb", combined_meshes)
        
        return
    
    def _sync_from_atlas(self, attr_name):
        #updates local stack with attribute from atlas
        category = self.yba.attributes.get(attr_name)
        atlas_value = getattr(self.yba, attr_name)

        def process_mesh(attr_name, attribute_init):
            if attr_name == "mesh":
                attribute_init = self.yba.whole_brain
                parcel_mask = attribute_init.point_data['parcel_labels'] == self.name
                parcel_vtp = attribute_init.extract_points(parcel_mask).extract_surface()
                parcel_faces = parcel_vtp.faces.reshape(-1, 4)[:, 1:]
                parcel_points = parcel_vtp.points
                attribute = trimesh.Trimesh(parcel_points, parcel_faces)
            else:
                attribute = attribute_init[self.name]
            return attribute

        #extract this parcel's data
        category_atlas_pull_registry = {
            BrainAttribute.Intrinsic: lambda atlas_value: self.yba.parcel_parameters.at[self.name, attr_name],
            BrainAttribute.Connectivity: lambda atlas_value: atlas_value[self.name].to_dict(),
            BrainAttribute.MatrixNP: lambda atlas_value: atlas_value[self.idx],
            BrainAttribute.MatrixTensor: lambda atlas_value: atlas_value[self.idx],
            BrainAttribute.Mesh: lambda atlas_value: process_mesh(attr_name, atlas_value),
            BrainAttribute.Group: lambda atlas_value: atlas_value[self.idx],
            BrainAttribute.GroupDict: lambda atlas_value: atlas_value[self.name],
            BrainAttribute.Other: lambda atlas_value: atlas_value[self.name] #I don't do this processing elsewhere, so maybe OTHER is it's own unique thing...I'm not exactly sure 
        }

        parcel_value = category_atlas_pull_registry[category](atlas_value)


        #update the stack and local version stamp
        self.__dict__[attr_name].append(self._wrap_in_proxy(parcel_value, attr_name, category))
        self._cache_versions[attr_name] = self.yba._attr_versions[attr_name][self.idx]

        return self.__dict__[attr_name][-1]

    def __setattr__(self, att_name, att_value):
        # exceptions for core management attributes, skips stacking
        if att_name in ('name', 'yba', 'idx', 'attributes'):
            super().__setattr__(att_name, att_value)
            return

        attributes_metadata = self.__dict__['attributes']
        # determine type of property -- the likely type of property (this part of the code is not perfect, but I think it should work; I think modifications will be tough, though)
        category = self._infer_category(att_value)

        #if the property already exists inside this parcel, check if the value being added matches the existing property types, and if it does, then add to stack
        if att_name in self.__dict__:
            preset_category = attributes_metadata.get(att_name)     
            if category != preset_category and (preset_category is not BrainAttribute.NONE) and (preset_category is not None):
                raise ValueError(f"Attribute ({category}) does not match typical type for this parcel attribute: {preset_category}")

        else:
            super().__setattr__(att_name, deque(maxlen=5)) #creates the stack if the property does not exist in the parcel
            attributes_metadata[att_name] = category

            #since the property doesn't exist in the parcel, we check if it doesn't exist in the yale brain atlas, and if it doesn't, then we create the global property
            yba_attribute = getattr(self.yba, att_name, None)
            if yba_attribute is None:
                self.__dict__['yba']._create_brain_attribute(att_name, category) #It basically creates the empty version of the attribute in the yba

        #add to the stack
        wrapped_att_value = self._wrap_in_proxy(att_value, att_name, category)
        self.__dict__[att_name].append(wrapped_att_value)
        #self.__dict__[att_name].append(att_value)

        #update the property in the yale brain atlas
        self._update_yba(att_name, att_value, category)

        #only tell the atlas to bump the index corresponding to this parcel
        self.yba._bump_version(att_name, parcel_idx=self.idx)

        #sync local version so we don't immediately re-fetch
        self._cache_versions[att_name] = self.yba._attr_versions[att_name][self.idx]
    
    def __getattribute__(self, attribute_name):
        #standard lookup. Generally modifying __getattribute__ is risky, but because I'm implementing a stack, it's a nice approach
        #the alternative would be to call parcel.attribute[-1] each time, which is kinda nice in its own way. Designer choice ig. 

        #internal bypass for initialized attributes, and start and end (parcel is "start of" or "end of" a fiber in a tract)
        if attribute_name.startswith('_') or attribute_name in ('name', 'yba', 'idx', 'attributes', 'starts_of', 'ends_of'):
            return super().__getattribute__(attribute_name)
        
        #grab the local value
        attribute_dict_val = super().__getattribute__('__dict__')
        attribute_value = attribute_dict_val.get(attribute_name) 

        if isinstance(attribute_value, deque):
            #check version for this parcel
            global_v = self.yba._attr_versions[attribute_name][self.idx]
            local_v = self._cache_versions.get(attribute_name, -1)

            if local_v < global_v:
                #sync without deleting the object
                #del self.__dict__[attribute_name]
                return self._sync_from_atlas(attribute_name) #getattr(self, attribute_name)
            
            return attribute_value[-1]

        return super().__getattribute__(attribute_name) #attribute_value will be None through attribute_dict_val.get, and the system will just return None!!! So we need to use super :)


    #properties like mesh, voxels, hull are not initialized, but can be implemented through lazy cache. They are implemented based on need.
    def __getattr__(self, attr):
        #if we are looking for attributes, and it hasn't been initialized
        #IPython searches for _repr_ stuff for a lot of internal attributes, so we are immediately rejecting those here

        #print(attr)
        if attr.startswith('_'):
            raise AttributeError(f"Internal attribute {attr} not found")
        
        if attr == 'attributes':
            raise AttributeError("Attributes dictionary not initalized yet")
        
        #Implements lazy caching -> checks the YBA to set the attribute value the first time, and then just pulls from the local attribute after that
        #CHECK TO SEE IF WE HAVE THE ATTRIBUTE IN THE YBA
        appearance_counter = 0
        attribute = None #creates the attribute to return

        # if attr in self.yba.attributes:
        #     print(f"{attr} in yba attributes")
        #     category = self.yba.attributes[attr]
        #     self.attributes[attr] = category

        #     if attr not in self.__dict__:
        #         super().__setattr__(attr, deque(maxlen=5))

        #     return self._sync_from_atlas(attr)
        
        # if attr == 'starts_of':
        #     return self.yba._tract_starts.get(self.name, [])
        
        # if attr == 'ends_of':
        #     return self.yba._tract_ends.get(self.name, [])
        
        # raise AttributeError(f"Attribute '{attr}' not found in this parcel or in atlas.") 

        if attr in self.yba.parcel_parameters.columns: # checks if the attribute is a general parcel parameter
            #if the attribute is just an atlas parameter and doesn't appear anywhere else, then add it as an attribute
            attribute = self.yba.parcel_parameters.at[self.name, attr] #[row, column]
            appearance_counter += 1
        else:
        #checks if the attribute is in the broader atlas dictionary
            attribute_init = getattr(self.yba, attr, None) #gets the attribute from the YBA
            if attribute_init is not None:
                # tries to figure out the (more complex) attribute type, so it can just extract the parcel information
                if self.yba.attributes[attr] is BrainAttribute.Connectivity: # num_parcels x num_parcels pandas dataframe -> select the parcel column
                    attribute = attribute_init[self.name].to_dict() #this creates a NEW OBJECT, which is a problem because if I modify the dictionary I'm not modifying the original thing
                    appearance_counter += 1
                elif self.yba.attributes[attr] in (BrainAttribute.MatrixNP, BrainAttribute.MatrixTensor): # e.g. shape: parcels x d1 x d2 x d3 x d4... -> select parcel (1 x d1 x d2 x ...)
                    attribute = attribute_init[self.idx]
                    appearance_counter += 1
                elif self.yba.attributes[attr] is BrainAttribute.Mesh: # e.g. whole brain mesh -> select parcel mesh from that
                    # special circumstance for the YBA brain, but if it's other meshes then we just have a dictionary.
                    if attr == "mesh":
                        attribute_init = self.yba.whole_brain
                        parcel_mask = attribute_init.point_data['parcel_labels'] == self.name
                        parcel_vtp = attribute_init.extract_points(parcel_mask).extract_surface()
                        parcel_faces = parcel_vtp.faces.reshape(-1, 4)[:, 1:] #to ensure that trimesh only gets the vertex indices
                        parcel_points = parcel_vtp.points
                        attribute = trimesh.Trimesh(parcel_points, parcel_faces)
                    else:
                        attribute = attribute_init[self.name] #in every other case, the mesh should be stored as multiblock dictionary
                        appearance_counter += 1
                elif self.yba.attributes[attr] in (BrainAttribute.Group, BrainAttribute.GroupDict):
                    if isinstance(attribute_init, dict): #BrainAttribue.GroupDict
                        attribute = attribute_init[self.name] #if it's an atlas dictionary, it needs to be that part of the dictionary with the key as the parcel name
                    if isinstance(attribute_init, (list, set)): #BrainAttribute.Group
                        attribute = attribute_init[self.idx] #if it's a list (e.g len 696) of lists with arbitrary length, get the list corresponding to the parcel id
                else:
                    attribute = attribute_init

        #IF WE HAVE THE ATTRIBUTE IN THE YBA, ADD IT TO THE PARCEL
        if attribute is not None:
            if appearance_counter > 1:
                raise AttributeError(f"Attribute {attr} has multiple appearances with same name in atlas, please define or specify type")
            else:
                #stamp the current version from the atlas
                self._cache_versions[attr] = self.yba._attr_versions[attr][self.idx]

                #super setattr function, which updates the stack for that object
                category = self.yba.attributes[attr]
                self.attributes[attr] = category #getattribute will work, since attributes is already created

                proxied_attribute = self._wrap_in_proxy(attribute, attr, category)
                super().__setattr__(attr, deque(maxlen=5)) #creates the stack for that object
                self.__dict__[attr].append(proxied_attribute) #adds the attribute to the stack
                return proxied_attribute #returns the actual value, not the whole stack. To call the stack, use __dict__[attribute name]
        
        if attr == 'starts_of':
            # Returns which tracts/fibers start at THIS parcel
            return self.yba._tract_starts.get(self.name, [])

        if attr == 'ends_of':
            # Returns which tracts/fibers end at THIS parcel
            return self.yba._tract_ends.get(self.name, [])
        
        #IF WE DON'T HAVE THE ATTRIBUTE IN THE YBA, WE RETURN THAT IT WAS NOT FOUND
        raise AttributeError(f"Attribute '{attr}' not found in this parcel or in atlas.")

    def __str__(self):
        return (
            f"<Parcel '{self.name}' | Atlas: {self.yba.name} | "
            f"Index: {self.idx} | {len(self.attributes)} Attributes>"
        )

    def __repr__(self):
        attr_list = ", ".join([f"{k}({v.name})" for k, v in self.attributes.items()])
        return (
            f"Parcel(name='{self.name}', idx={self.idx}, atlas='{self.yba.name}')\n"
            f"Attributes: [{attr_list if attr_list else 'None loaded'}]"
        )

    def __len__(self):
        # gives the number of attributes inside the parcel
        return len(self.attributes)

    def create_hull(self, alpha):
        # re-writes the convex hull that exists

        #takes the mesh (which is a surface mesh)
        parcel_mesh = self.mesh.copy() #this would be a trimesh object already
        #creates a hull around the mesh
    
        hull_parcel = parcel_mesh.delaunay_3d(alpha=alpha)
        surface = hull_parcel.extract_surface()
        surface = surface.clean().triangulate().clean()
        surface = surface.fill_holes(1)
        v_hull = surface.points
        f_hull = surface.faces.reshape(-1, 4)[:, 1:] #do I need to do faces.reshape if it's already a trimesh object?

        parcel_hull = trimesh.Trimesh(v_hull, f_hull)

        #creates the hull attribute for the parcel
        setattr(self, f"hull_{alpha}", parcel_hull)
        return(f"Hull with alpha={alpha} created and saved as 'self.hull_{alpha}'")
