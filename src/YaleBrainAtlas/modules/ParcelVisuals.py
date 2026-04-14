import numpy as np
import pandas as pd
import pyvista as pv
import seaborn as sns
import trimesh
import os
import sys

from YaleBrainAtlas.attributes import BrainAttribute, MapProxy, ListProxy, TensorProxy, ArrayProxy
from YaleBrainAtlas.parcel import Parcel
from YaleBrainAtlas.yba import YaleBrainAtlas

def get_parcel_meshes(subject: YaleBrainAtlas):
    
    subject_wb = subject.whole_brain
    pts = subject_wb.points
    faces = subject_wb.faces.reshape(-1, 4)[:, 1:]
    labels = subject_wb.point_data['parcel_labels']

    parcel_meshes_wb = {}

    for name in subject.parcel_names:
        point_mask = (labels == name)
        if not np.any(point_mask):
            continue
        
        subset_points = pts[point_mask]
        
        face_mask = point_mask[faces].all(axis=1)
        subset_faces_raw = faces[face_mask]
        
        old_indices = np.where(point_mask)[0]
        index_map = np.full(pts.shape[0], -1, dtype=int)
        index_map[old_indices] = np.arange(len(old_indices))
        
        subset_faces = index_map[subset_faces_raw]
        parcel_meshes_wb[name] = trimesh.Trimesh(subset_points, subset_faces, process=False)

    return parcel_meshes_wb

def linear_recursive_split(pnorm,
                    segment_vector,
                    minPt,
                    n_segments,
                    split_mesh,
                    meshes=None,
                    i=1):
  '''
  returns a list of the slices (n_segments) for a given parcel, along an axis (pnorm)
  '''

  if meshes is None:
    meshes = []

  if i == n_segments:
      meshes.append(split_mesh)
      return meshes

  else:
      #split mesh into top and bottom
      top_segment = trimesh.intersections.slice_mesh_plane(
          mesh = split_mesh,
          plane_normal = pnorm,
          plane_origin = minPt + i * segment_vector,
          cap=True
      )

      bottom_segment = trimesh.intersections.slice_mesh_plane(
          mesh = split_mesh,
          plane_normal = np.array(pnorm) * -1,
          plane_origin = minPt + i* segment_vector,
          cap=True
      )

      #store bottom segment, pass top segment
      meshes.append(bottom_segment)
      return linear_recursive_split(pnorm, segment_vector, minPt, n_segments, top_segment, meshes, i+1)


def linear_split_color_parcels(parcel, norm_axis, n_segments, colors=None):
    '''
    Returns a list of slices for the parcel, given the parcel mesh, the axis for splitting, and the number of segments. 
    Can also define a list of colors (this should be done)
    '''
    if colors is None:
        colors = [trimesh.visual.random_color() for i in range(n_segments)]

    parcel_centroid = parcel.centroid
    maxPt = parcel.vertices[np.argmax(np.dot(parcel.vertices - parcel_centroid, np.array(norm_axis)))]
    minPt = parcel.vertices[np.argmin(np.dot(parcel.vertices - parcel_centroid, np.array(norm_axis)))]

    segment_length = np.linalg.norm(maxPt - minPt)/n_segments
    segment_vector = (maxPt - minPt)/np.linalg.norm(maxPt - minPt) * segment_length

    meshes_list = linear_recursive_split(
        pnorm=norm_axis,
        segment_vector=segment_vector,
        minPt = minPt,
        n_segments = n_segments,
        split_mesh = parcel
    )

    for i, mesh_ in enumerate(meshes_list):
        mesh_.visual.face_colors = colors[i % len(colors)]

    return meshes_list