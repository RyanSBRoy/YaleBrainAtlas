import numpy as np
import pandas as pd
import pyvista as pv
import numbers
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import trimesh
import numbers
import os
import sys

from YaleBrainAtlas.src.attributes import BrainAttribute, MapProxy, ListProxy, TensorProxy, ArrayProxy
from YaleBrainAtlas.src.parcel import Parcel
from YaleBrainAtlas.src.yba import YaleBrainAtlas
from YaleBrainAtlas.src.tract import Tract

# with open("TractNames.txt", "r") as file_t:
#     tractNames = [line.rstrip('\n') for line in file_t]

class YBAVisualizer:
    def __init__(self, yba):
        self.yba = yba
        self.wb = yba.whole_brain
        #self.tracts = self._load_tracts('./tracts_')
        self.tracts_filtered = {}
        self.title = 'Main'
        self.fig = go.Figure().update_layout(
            template="plotly_white",
            scene=dict(xaxis_visible=False, 
                       yaxis_visible=False, 
                       zaxis_visible=False),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        self.figures = {self.title: self.fig}

        self.palette_range = [3, 0, 2, 1] + list(range(4, 20))
        self.yba_palette = sns.color_palette('deep', 20)

        #maps the parcel parameters for the YBA object to parcel labels
        self.pclPointMap = lambda param: pd.Series(self.yba.parcel_labels).map(self.yba.parcel_parameters[param].to_dict())

        #counts the number of parcels with a non-nan value for a given parameter set
        self.pclParamCount = lambda params: self.yba.parcel_parameters[params].count(axis=1)
    
    # def _load_tracts(self, path):
    #     tracts = {}
    #     if not os.path.exists(path):
    #         raise LookupError(f"Path {path} could not be found")
    #     else:
    #         if len(os.listdir(path)) == 0:
    #             raise ValueError("Path provided does not have any tracts")
    #         for tract_file in os.listdir(path):
    #             if tract_file.endswith('.vtp'):
    #                 tract_ = pv.read(os.path.join(path, tract_file))
    #                 tracts[tract_file[:-4]] = tract_
    #     return tracts
    
    def __str__(self):
        return f"yba: {self.yba.name} \n figures: {list(self.figures)}"
    
    def __repr__(self):
        if self.title is not None:
            return f"yba: {self.yba.name} \n figures: {list(self.figures)} \n Current Figure: {self.title}"
        else:
            return f"yba: {self.yba.name} \n figures: {list(self.figures)}"

    def new(self, name):
        self.fig = go.Figure().update_layout(
            template="plotly_white",
            scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        self.figures[name] = self.fig
        self.title = name       
        return
    
    def set(self, name):
        if name in self.figures:
            self.fig = self.figures[name]
            self.title = name
    
    def show(self, name):
        if name not in self.figures:
            raise KeyError(f"Error: Figure '{name}' does not exist.")

        fig = self.figures[name]
        fig.update_layout(title=name)
        return fig.show()

    def export_all(self, folder="Figures", fmt="html"):
        if not os.path.exists(folder):
            os.makedirs(folder)
        for name, fig in self.figures.items():
            filename = os.path.join(folder, f"{name}.{fmt}")
            if fmt == "html":
                fig.write_html(filename)
            else:
                fig.write_image(filename, engine="kaleido")
            print(f"Exported: {filename}")


    def add_custom_mesh_plotly(self, vertices, faces, color='green', opacity=0.8, label='Custom Mesh'):
        self.fig.add_trace(go.Mesh3d(
            x=vertices[:,0], y=vertices[:,1], z=vertices[:,2],
            i=faces[:,0], j=faces[:,1], k=faces[:,2],
            color=color, opacity=opacity, name=label
        ))

    def _rgb_to_hex(self, rgb_tuple):
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgb_tuple[0] * 255),
            int(rgb_tuple[1] * 255),
            int(rgb_tuple[2] * 255)
        )

    #This is for when we have multiple parameters provided as a list under intensities
    #These params need to be those that are defined already in the corresponding Yale Brain Atlas
    def _process_multiple_params(self, params: list):
        paramsPD = self.yba.parcel_parameters[params] #this is a 696 vector
        paramsPD = paramsPD.replace('None', np.nan).apply(pd.to_numeric)

        count = self.pclParamCount(params) #counts the number of parcels with non-nan values
        
        #parcel names, but if there's more than one value for a parcel, it returns the count of parameters
        #and if there's only one value for the parcel (one parameter), it returns the intensity of that parameter
        parcel_parameter_labels = pd.Series(np.where(count > 1, count, paramsPD.idxmax(axis=1)), index=paramsPD.index)
        mixed_params = parcel_parameter_labels.unique()

        #provide the parcel intensity, yes or no; for color mapping
        parcel_intensity_tf = pd.Series(np.where(count > 0, 1, 0), index=paramsPD.index)
        mixed_param_colormap = {
                mixed_params[i]: self._rgb_to_hex(self.yba_palette[self.palette_range[i]]) 
                for i in range(len(mixed_params) - 1, -1, -1)
            }
        
        #creates color mapping dictionary
        parcel_param_colormap = {
            idx: mixed_param_colormap[parcel_parameter_labels.loc[idx]] if count.loc[idx] > 0 
            else '#ffffff'
            for idx in parcel_parameter_labels.index
        }

        #creates hover labels
        parcel_hoverlabels_dict = {parcel: 
            f"<b>{parcel}</b><br>" + "<br>".join(f"{k}: {round(v, 5)}" for k, v in row.items() if pd.notna(v))
            for parcel, row in paramsPD.to_dict('index').items()
        }

        #maps the color map, hoverlabels, and intensities (binary) to parcel_labels
        parcel_labels_colormap = pd.Series(self.yba.parcel_labels).map(parcel_param_colormap)
        parcel_labels_hover = pd.Series(self.yba.parcel_labels).map(parcel_hoverlabels_dict)
        parcel_labels_intensities = pd.Series(self.yba.parcel_labels).map(parcel_intensity_tf.to_dict())

        return {
            'color': parcel_labels_colormap,
            'hovertext': parcel_labels_hover,
            'intensity': parcel_labels_intensities
        }

    
    def add_parcels(self, intensities, segment='whole', labels=None, colorscale='reds', opacity=0.3, title='Value', **kwargs):
        wb_mesh = self.wb
        p_labels = self.yba.parcel_labels   # vertex-length
        p_names = self.yba.parcel_names                            # length 696

        p_names_arr = np.array(p_names)
        p_names_noCC = p_names_arr[np.where(~pd.Series(p_names_arr).str.contains('CC'))].tolist()

        faces = wb_mesh.faces.reshape(-1, 4)[:, 1:]  
        continuous_colors = False
        # figure out what intensities gives us, and how to process that
        #If the intensities argument is a single string, or if it's a list with only one string
        if isinstance(intensities, str) or (isinstance(intensities, list) and len(intensities) == 1):
            parameter = intensities if isinstance(intensities, str) else intensities[0]

            if not isinstance(parameter, str):
                raise ValueError("The parameter value in the list provided must be a string")
            
            if parameter not in self.yba.parcel_parameters.columns:
                raise ValueError(f"{parameter} is not in parcel intrinisic attribute list for corresponding brain atlas")
            
            else:
                parcel_label_intensities = self.pclPointMap(parameter)

                if labels is None:
                    parcel_label_hover = [f"<b>{label}:</b><br> {round(val, 5)}" for label, val in zip(p_labels, parcel_label_intensities)]
                
                continuous_colors = True
        
        #If the intensities argument is a list with multiple strings
        elif isinstance(intensities, list) and (all(isinstance(item, str) for item in intensities)) and (len(intensities) > 1):
            if not all(intensity in self.yba.parcel_parameters.columns for intensity in intensities):
                raise ValueError(f"Intensities argument is a list of strings, but at least one of {intensities} is not in parcel intrinsic attribute list for corresponding brain atlas")
            else:
                parcel_label_parameters = self._process_multiple_params(intensities)
                parcel_label_intensities = parcel_label_parameters['intensity'].values
                if labels is None:
                    parcel_label_hover = parcel_label_parameters['hovertext'].values
                parcel_label_color = parcel_label_parameters['color'].values
                continuous_colors=False
        
        #If the intensities argument is a list with the actual numbers per parcel
        elif isinstance(intensities, list) and (all(isinstance(item, numbers.Number) for item in intensities)):
            if len(intensities) == len(p_labels):
                #then we have the intensity list corresponding to parcel points
                parcel_label_intensities = intensities
            
            elif len(intensities) == len(p_names):
                intensities_series = pd.Series(intensities, index=p_names)
                parcel_label_intensities = pd.Series(p_labels).map(intensities_series.to_dict()).values
            
            elif len(intensities) == len(p_names_noCC):
                intensities_series = pd.Series(intensities, index = p_names_noCC) ## p_names without corpus callosums
                parcel_label_intensities = pd.Series(p_labels).map(intensities_series.to_dict()).values
            
            else:
                raise ValueError(f"The length of intensities must be {len(p_labels)}, 696, or 690")
            
            if labels is None:

                parcel_label_hover = [f"<b>{label}:</b><br> {round(val, 5)}" for label, val in zip(p_labels, intensities)]
            continuous_colors = True
        
        #we've defined parcel_label_intensities, parcel_label_hover if labels is None, and continuous colors in one case
        if labels is not None:
            if isinstance(labels, list): 
                if len(labels) == len(p_labels):
                    parcel_label_hover = [f"<b>{label}:</b><br> {round(val, 5)}" for label, val in zip(p_labels, labels)]

                elif len(labels) == len(p_names):
                    labels_series = pd.Series(labels, index=p_names)
                    parcel_label_hover_suffix = pd.Series(p_labels).map(labels_series.to_dict()).values
                    parcel_label_hover = [f"<b>{label}:</b><br> {round(val, 5)}" for label, val in zip(p_labels, parcel_label_hover_suffix)]

                elif len(labels) == len(p_names_noCC):
                    labels_series = pd.Series(labels, index=p_names_noCC)
                    parcel_label_hover_suffix = pd.Series(p_labels).map(labels_series.to_dict()).values
                    parcel_label_hover = [f"<b>{label}:</b></br> {round(val, 5)}" for label, val in zip(p_labels, parcel_label_hover_suffix)]

                else:
                    raise ValueError(f"The length of labels must be either {len(p_labels)}, 696, or 690")
                                 
            else:
                raise TypeError("labels must be a list")
        
        #we clear up all of the cases for colors at the plotting phase

        
        # case: parcels = 'whole'
        # case: parcels = 'left' or 'right' hemisphere
        # filter out the parcels and only plot those

        hemisphere_mask = {
            'right_hemisphere': np.char.startswith(p_labels.astype(str), 'R_'),
            'left_hemisphere':  np.char.startswith(p_labels.astype(str), 'L_'),
        }
                
        if segment == 'whole':
            v_mask = np.ones(len(self.yba.parcel_labels), dtype=bool)
        elif isinstance(segment, (list, set)):
            v_mask = np.isin(self.yba.parcel_labels, list(segment))
        elif segment in hemisphere_mask:
            v_mask = hemisphere_mask[segment]
        else:
            raise ValueError(f"Unknown segment '{segment}'.")

        v_indices   = np.where(v_mask)[0]
        new_idx_map = np.full(len(p_labels), -1, dtype=int)
        new_idx_map[v_indices] = np.arange(len(v_indices))

        face_mask  = v_mask[faces[:, 0]] & v_mask[faces[:, 1]] & v_mask[faces[:, 2]]
        final_faces = new_idx_map[faces[face_mask]]

        final_x   = wb_mesh.points[v_indices, 0]
        final_y   = wb_mesh.points[v_indices, 1]
        final_z   = wb_mesh.points[v_indices, 2]

        final_intensities  = parcel_label_intensities[v_indices]
        final_hovertext = np.array(parcel_label_hover)[v_indices]
        if continuous_colors is False:
            self.fig.add_trace(go.Mesh3d(
                x=final_x, y=final_y, z=final_z,
                i=final_faces[:, 0], j=final_faces[:, 1], k=final_faces[:, 2],
                #intensity=final_intensities,
                vertexcolor=np.array(parcel_label_color)[v_indices],
                hovertext=final_hovertext,
                hoverinfo="text",
                opacity=opacity,
                colorbar=dict(title=title),
                name=str(segment),
                **kwargs
                #lighting=dict(ambient=0.1,
                #diffuse=1,
                #fresnel=3,  
                #specular=0.5, 
                #roughness=0.05),
                #lightposition=dict(x=100,
                                    #y=200,
                                    #z=1000),
            ))
        
        else:
            self.fig.add_trace(go.Mesh3d(
                x=final_x, y=final_y, z=final_z,
                i=final_faces[:, 0], j=final_faces[:, 1], k=final_faces[:, 2],
                intensity=final_intensities,
                colorscale=colorscale,
                hovertext=final_hovertext,
                hoverinfo="text",
                opacity=opacity,
                colorbar=dict(title=title),
                name=str(segment),
                **kwargs
                #lighting=dict(ambient=0.1,
                #diffuse=1,
                #fresnel=3,  
                #specular=0.5, 
                #roughness=0.05),
                #lightposition=dict(x=100,
                                    #y=200,
                                    #z=1000),
            ))

        return
    
    # def add_tracts(self, tractNames='whole', color='gold', opacity=0.5):
    #     first_tract = True
    #     for name, mesh in self.tracts.items():
    #         is_match = (tractNames == 'whole' or 
    #                    (tractNames == 'right' and (name.startswith('Left') or name.endswith('L'))) or 
    #                    (tractNames == 'left' and (name.startswith('Right') or name.endswith('R'))) or
    #                    (isinstance(tractNames, list) and name in tractNames))
            
    #         if is_match:
    #             points, lines = mesh.points, mesh.lines
    #             x, y, z = [], [], []
    #             i = 0
    #             while i < len(lines):
    #                 n_pts = lines[i]
    #                 seg = points[lines[i+1 : i+1+n_pts]]
    #                 x.extend(seg[:, 0].tolist() + [None])
    #                 y.extend(seg[:, 1].tolist() + [None])
    #                 z.extend(seg[:, 2].tolist() + [None])
    #                 i += (n_pts + 1)

    #             self.fig.add_trace(go.Scatter3d(
    #                 x=x, y=y, z=z, mode='lines', line=dict(color=color, width=3), opacity=opacity,
    #                 name="Tracts", legendgroup="Tracts", showlegend=first_tract, hoverinfo='name'
    #             ))
    #             first_tract = False
