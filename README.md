- [The Yale Brain Atlas](#the-yale-brain-atlas)
- [Analysis](#analysis)
  - [Example Walkthrough](#example-walkthrough)
    - [Import and Instantiation](#import-and-instantiation)
    - [Parcel Parameters](#parcel-parameters)
      - [Defining parcel parameters at the whole-brain level](#defining-parcel-parameters-at-the-whole-brain-level)
        - [Working with 690 parcels (no corpus callosum) in the YBA](#working-with-690-parcels-no-corpus-callosum-in-the-yba)
        - [Get all parcel parameters together](#get-all-parcel-parameters-together)
  - [Full Code](#full-code)
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

```
from YaleBrainAtlas import *
# OR from YaleBrainAtlas import YaleBrainAtlas
```

Initialize a brain atlas for a sample subject

```
Subject = YaleBrainAtlas('Subject')
```

### Parcel Parameters

#### Defining parcel parameters at the whole-brain level

There are 696 parcels in the Yale Brain Atlas.
Parcel parameters can be defined globally for the brain as a list, set, or Pandas Series of 696 values, or a dictionary or pandas Dataframe of 696 keys/rows corresponding to the parcel name and one corresponding value per key/row.

```
Subject.CT = np.random.randn(696)
Subject.PET = np.random.randn(696)

print(Subject.PET)
print(Subject.CT)
```
##### Working with 690 parcels (no corpus callosum) in the YBA
Sometimes we exclude the corpus callosum in the YBA. This leaves us with values in the form of a list corresponding to 690 parcels, instead of 696. To bring this into a subject's brain atlas object, we must first transform the 690-parcel list into a 696-parcel list.

```
clusters = np.random.randn(690).tolist()
Subject.clusters = pd.Series(Subject.parcel_names).map(dict(zip(Subject.parcel_names_noCC.copy(), clusters)))

print(Subject.clusters)
```

Note that Subject has both parcel_names (Subject.parcel_names), of length 696, and parcel_names_noCC, of length 690, which contain parcel names with and without the Corpus Callosum respectively. 

##### Get all parcel parameters together
Parcel parameters can be numerical values or strings. The parcel parameters assigned to a YBA object can be obtained as a pandas dataframe through the 'parcel_parameters' attribute

```
Subject.parcel_parameters
```


Runa.clusters = pd.Series(Runa.parcel_names).map(dict(zip(Runa.parcel_names_noCC.copy(), clusters)))

## Full Code

```

```


# Visualization

```

```

## White Matter Tracts
Currently, the YBA is built for parcel visualization and analysis. The user must define their own set of streamlines. These streamlines must be in PyVista PolyLine format, or in .h5 format containing:

    1. 'points': a single array of all the (x, y, z) coordinates for all the streamlines of the tract, in MNI152 space. 
    2. 'offsets': a list of indices indicating the start index in 'points' for each fiber in the tract
    3. 'transform': a transformation matrix between MNI and subject space. 
