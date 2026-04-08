- [The Yale Brain Atlas](#the-yale-brain-atlas)
- [Analysis](#analysis)
  - [Example Walkthrough](#example-walkthrough)
    - [Import and Instantiation](#import-and-instantiation)
    - [Parcel Parameters](#parcel-parameters)
      - [Defining parcel parameters at the whole-brain level](#defining-parcel-parameters-at-the-whole-brain-level)
        - [Working with 690 parcels (no corpus callosum) in the YBA](#working-with-690-parcels-no-corpus-callosum-in-the-yba)
        - [Get all parcel parameters together](#get-all-parcel-parameters-together)
      - [Defining parcel parameters at the individual parcel-level](#defining-parcel-parameters-at-the-individual-parcel-level)
    - [Connectivities](#connectivities)
- [Visualization](#visualization)
  - [White Matter Tracts](#white-matter-tracts)


# The Yale Brain Atlas

This repository is an **unofficial, work-in-progress** code library for the [Yale Brain Atlas](https://yalebrainatlas.github.io/YaleBrainAtlas/atlas_viewer.html#model=images/Yale_Brain_Atlas.obj,images/Yale_Brain_Atlas.mtl) (YBA). It can be used for analysis and visualization of the brain, from the level of individual parcels to the whole cortex. 

For the official version of the Yale Brain Atlas, please refer to work done by the [Yale Clinical Neuroscience Neuroanalytics group](https://medicine.yale.edu/lab/ynn/), and the following [GitHub Page](https://github.com/YaleBrainAtlas/YaleBrainAtlas?tab=readme-ov-file).

# Analysis

The YBA can be fetched and modified at the individual parcel level, as well as the global brain (atlas) level.

## Example Walkthrough

### Import and Instantiation
First, import the Yale Brain Atlas

```python
from YaleBrainAtlas import *
# OR from YaleBrainAtlas import YaleBrainAtlas
```

Initialize a brain atlas for a sample subject

```python
Subject = YaleBrainAtlas('Subject')
```

### Parcel Parameters

#### Defining parcel parameters at the whole-brain level

There are 696 parcels in the Yale Brain Atlas.
Parcel parameters can be defined globally for the brain as a list, set, or Pandas Series of 696 values, or a dictionary or pandas dataframe of 696 keys/rows corresponding to the parcel name and one corresponding value per key/row.

```python
Subject.CT = np.random.randn(696)
Subject.PET = np.random.randn(696)

print(Subject.PET)
print(Subject.CT)
```
##### Working with 690 parcels (no corpus callosum) in the YBA
Sometimes we exclude the corpus callosum in the YBA. This leaves us with values in the form of a list corresponding to 690 parcels, instead of 696. To bring this into a subject's brain atlas object, we must first transform the 690-parcel list into a 696-parcel list.

```python
clusters = np.random.randn(690).tolist()
Subject.clusters = pd.Series(Subject.parcel_names).map(dict(zip(Subject.parcel_names_noCC.copy(), clusters)))

print(Subject.clusters)
```

Note that Subject has both parcel_names (Subject.parcel_names), of length 696, and parcel_names_noCC, of length 690, which contain parcel names with and without the Corpus Callosum respectively. 

##### Get all parcel parameters together
Parcel parameters can be numerical values or strings. The parcel parameters assigned to a YBA object can be obtained as a pandas dataframe through the 'parcel_parameters' attribute

```python
Subject.parcel_parameters
```

#### Defining parcel parameters at the individual parcel-level
Parcel objects in the YBA implement lazy caching, meaning that they first pull their value directly from the global Yale Brain Atlas object, and then store that value locally. 

```python
Subject.CT = np.random.randn(696)
print(Subject.CT)

print(Subject.L_TP1_A) #the CT value is not found as an attribute of the parcel, because we haven't called it yet

print(Subject.L_TP1_A.CT)

print(Subject.L_TP1_A) #the CT value will now be found as an attribute of the parcel...
print(Subject.L_TP1_B) #...but it won't be found for L_TP1_B because we haven't asked for it yet! 
```

You can change an existing attribute value for a parcel by modifying the parcel.
```python
print(Subject.CT.at['L_TP1_A'])
Subject.L_TP1_A.CT = 5
print(Subject.CT.at['L_TP1_A'])
```

Modifying the attribute value for a parcel value at the whole-brain level is a bit more difficult, in that you will need to explicitly update the version counter so that the parcel knows to pull the value from the global Yale Brain Atlas object. I am currently working on getting the the YBA to handle this internally. 

```python
print(Subject.L_TP1_A.CT)

#use 'at': using '.loc' or setting Subject.CT['L_TP1_A'] directly creates a copy of the dataframe outside of the YBA
Subject.CT.at['L_TP1_A'] = 6 

Subject._bump_version('CT', parcel_idx=Subject.L_TP1_A.idx) 
print(Subject.L_TP1_A.CT)
```

Whole-brain parcel parameters can be initialized at the level of parcels.

```python
Subject.L_TP1_A.MEG = 5
print(Subject.L_TP1_A.MEG)

print(Subject.MEG) #you should see that MEG is set to 5, and every other parcel has None/NA
print(Subject.parcel_parameters) #you should see that MEG is set to 5, and every other parcel has None/NA
```


### Connectivities

Connectivities can be entered at the whole-brain level as a 696x696 pandas dataframe
```python
Subject.FunctionalConnectivity = pd.DataFrame(np.ones([696, 696]), index=Subject.parcel_names, columns=Subject.parcel_names)
print(Subject.L_TP1_A.FunctionalConnectivity)
```

Connectivities can be modified at either the parcel level, or the whole brain level
```python
Subject.FunctionalConnectivity.at['L_TP1_A', 'L_TP1_A'] = 5
print(Subject.L_TP1_A.FunctionalConnectivity)

Subject.L_TP1_A.FunctionalConnectivity = 6
print(Subject.FunctionalConnectivity.at['L_TP1_A', 'L_TP1_A'])
```

# Visualization

```

```

## White Matter Tracts
Currently, the YBA is built for parcel visualization and analysis. The user must define their own set of streamlines. These streamlines must be in PyVista PolyLine format, or in .h5 format containing:

    1. 'points': a single array of all the (x, y, z) coordinates for all the streamlines of the tract, in MNI152 space. 
    2. 'offsets': a list of indices indicating the start index in 'points' for each fiber in the tract
    3. 'transform': a transformation matrix between MNI and subject space. 
