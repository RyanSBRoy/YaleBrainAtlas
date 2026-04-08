from enum import Enum
import numpy as np
import pandas as pd
import torch
import numbers

#all the different attribute types for the YBA. The numbering is not really important
class BrainAttribute(Enum):
    Other = 8
    Intrinsic = 1
    BrainIntrinsic = 9
    Connectivity = 2
    MatrixNP = 6 #refers to multidimensional numpy or tensors
    MatrixTensor = 7
    Group = 4
    GroupDict = 5
    Mesh = 3
    NONE = 0

#Basically a wrapper for a dictionary, typically used for Connectivity, or a GroupDict or maybe Group
class MapProxy(dict):
    def __init__(self, data, parcel_obj, attr_name):
        super().__init__(data)
        super().__setattr__('_parcel', parcel_obj)
        super().__setattr__('_attr_name', attr_name)
    
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        setattr(self._parcel, self._attr_name, self)
    
    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        setattr(self._parcel, self._attr_name, self)

#wrapper for an array, see comment above (for MapProxy)
class ArrayProxy(np.ndarray):
    def __new__(cls, input_array, parcel_obj, attr_name):
        obj = np.asarray(input_array).view(cls)
        obj._parcel = parcel_obj
        obj._attr_name = attr_name
        return obj

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        setattr(self._parcel, self._attr_name, self)

#wrapper for a tensor see comment above (for MapProxy)
class TensorProxy:
    def __init__(self, data, parcel_obj, attr_name):
        self._data = data 
        self._parcel = parcel_obj
        self._attr_name = attr_name

    def __setitem__(self, key, value):
        self._data[key] = value
        setattr(self._parcel, self._attr_name, self._data)

    def __getattr__(self, name):
        # If user calls .mean(), .sum(), .device, etc., send it to the real tensor
        return getattr(self._data, name)
    
    def __repr__(self):
        return f"TensorProxy({self._data})"

#wrapper for a list see comment above (for MapProxy)
class ListProxy(list):
    def __init__(self, data, parcel_obj, attr_name):
        super().__init__(data)
        self._parcel = parcel_obj
        self._attr_name = attr_name

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        setattr(self._parcel, self._attr_name, self)

    def append(self, item):
        super().append(item)
        setattr(self._parcel, self._attr_name, self)

    def extend(self, other):
        super().extend(other)
        setattr(self._parcel, self._attr_name, self)