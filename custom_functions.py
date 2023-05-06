#imports
import pandas as pd
import numpy as np
from datetime import datetime
import kshingle as ks
import random
import math
import itertools
import prince
import warnings
from functools import wraps
from itertools import combinations, product
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.optimize import minimize, NonlinearConstraint
from rpack import pack
from grandalf.graphs import Vertex, Edge, Graph
from grandalf.layouts import SugiyamaLayout, DummyVertex
from netgraph_functions import get_geometric_layout

#function to get a matrix containing the jaccard similarity between every pair of titles in the set of datasets
def title_jaccard_similarity(df, shingle_length):
  #make a len(df) by len(df) matrix filled with zeros
  matrix = np.zeros((len(df), len(df)))

  for i in range(len(df)):
    for j in range(len(df)):
      if i == j:
        matrix[i][j] = 1
      elif i > j:
        continue
      else:
        title1 = df["title"][i]
        title2 = df["title"][j]
        if pd.isnull(title1) or pd.isnull(title2):
          matrix[i][j] = 0
          matrix[j][i] = 0
        else:
          shingles1 = ks.shingleset_range(title1, shingle_length, shingle_length)
          shingles2 = ks.shingleset_range(title2, shingle_length, shingle_length)
          intersection = len(set(shingles1).intersection(shingles2))
          union = len(set(list(shingles1) + list(shingles2)))
          similarity = intersection/union
          matrix[i][j] = similarity
          matrix[j][i] = similarity

  return matrix


#function to get a matrix containing the jaccard similarity between every pair of descriptions in the set of datasets
def description_jaccard_similarity(df, shingle_length):
  #make a len(df) by len(df) matrix filled with zeros
  matrix = np.zeros((len(df), len(df)))

  for i in range(len(df)):
    for j in range(len(df)):
      if i == j:
        matrix[i][j] = 1
      elif i > j:
        continue
      else:
        title1 = df["description"][i]
        title2 = df["description"][j]
        if pd.isnull(title1) or pd.isnull(title2):
          matrix[i][j] = 0
          matrix[j][i] = 0
        else:
          shingles1 = ks.shingleset_range(title1, shingle_length, shingle_length)
          shingles2 = ks.shingleset_range(title2, shingle_length, shingle_length)
          intersection = len(set(shingles1).intersection(shingles2))
          union = len(set(list(shingles1) + list(shingles2)))
          similarity = intersection/union
          matrix[i][j] = similarity
          matrix[j][i] = similarity

  return matrix


#function to get a matrix containing the jaccard similarity between every pair of titles and descriptions (treated as one string for each dataset) in the set of datasets
def title_and_description_jaccard_similarity(df, shingle_length):
  #make a len(df) by len(df) matrix filled with zeros
  matrix = np.zeros((len(df), len(df)))

  for i in range(len(df)):
    for j in range(len(df)):
      if i == j:
        matrix[i][j] = 1
      elif i > j:
        continue
      else:
        title1 = df["title"][i]
        title2 = df["title"][j]
        desc1 = df["description"][i]
        desc2 = df["description"][j]
        if (pd.isnull(title1) and pd.isnull(desc1)) or (pd.isnull(title2) and pd.isnull(desc2)):
          matrix[i][j] = 0
          matrix[j][i] = 0
        else:
          if pd.isnull(title1) and not pd.isnull(desc1):
            both1 = desc1
          elif pd.isnull(desc1) and not pd.isnull(title1):
            both1 = title1
          elif pd.isnull(title2) and not pd.isnull(desc2):
            both2 = desc2
          elif pd.isnull(desc2) and not pd.isnull(title2):
            both2 = title2
          else:
            both1 = title1 + " " + desc1
            both2 = title2 + " " + desc2
          shingles1 = ks.shingleset_range(both1, shingle_length, shingle_length)
          shingles2 = ks.shingleset_range(both2, shingle_length, shingle_length)
          intersection = len(set(shingles1).intersection(shingles2))
          union = len(set(list(shingles1) + list(shingles2)))
          similarity = intersection/union
          matrix[i][j] = similarity
          matrix[j][i] = similarity

  return matrix


#convert jaccard similarity matrix to jaccard distance matrix
def distance_from_similarity(similarity_matrix):
  jac_dist = []
  for i in range(len(similarity_matrix)):
    inner_matrix = []
    for j in range(len(similarity_matrix)):
      inner_matrix.append(1 - similarity_matrix[i][j])
    jac_dist.append(inner_matrix)
  return np.array(jac_dist)


#node ID is just the index of its row in the df
def get_edges(df):
  edges = []
  for i in range(len(df)):
    for j in range(len(df)):
      if j < i: #no duplicates (edges are bidirectional), no i = j
        edges.append((i, j))
      else:
        break
  return edges


def get_edge_lengths(edges, jac_dist_matrix):
  edge_lengths = {}
  for i in range(len(edges)):
    length = jac_dist_matrix[edges[i][0]][edges[i][1]]
    edge_lengths[edges[i]] = length
  return edge_lengths


def normalize(values, bounds):
    return [bounds['desired']['lower'] + (x - bounds['actual']['lower']) * (bounds['desired']['upper'] - bounds['desired']['lower']) / (bounds['actual']['upper'] - bounds['actual']['lower']) for x in values]


def get_node_positions(edges, edge_lengths):
  node_positions = get_geometric_layout(edges, edge_lengths, node_size=0., tol=1e-3, origin=(0,0), scale=(1, 1), pad_by=0.05)
  node_positions_arr = []
  for i in range(len(node_positions)):
    node_positions_arr.append(node_positions[i])
  node_positions_arr = np.array(node_positions_arr)
  return node_positions_arr


def get_df_cols(df):
  cat_col_names = []
  spatial_col_names = []
  temporal_col_names = []
  misc_col_names = []
  for i in range(len(df["cat_col_names"].to_list())):
    to_concat = []
    if pd.isnull(df["cat_col_names"].to_list()[i]): 
      to_concat = to_concat
      cat_col_names.append('')
    else:
      to_concat = to_concat + df["cat_col_names"].to_list()[i].split(", ")
      cat_col_names.append(' '.join(df["cat_col_names"].to_list()[i].split(", ")))
    if pd.isnull(df["spatial_col_names"].to_list()[i]): 
      to_concat = to_concat
      spatial_col_names.append('')
    else:
      to_concat = to_concat + df["spatial_col_names"].to_list()[i].split(", ")
      spatial_col_names.append(' '.join(df["spatial_col_names"].to_list()[i].split(", ")))
    if pd.isnull(df["temporal_col_names"].to_list()[i]): 
      to_concat = to_concat
      temporal_col_names.append('')
    else:
      to_concat = to_concat + df["temporal_col_names"].to_list()[i].split(", ")
      temporal_col_names.append(' '.join(df["temporal_col_names"].to_list()[i].split(", ")))

    misc_col_names.append(' '.join(list(set(df["all_col_names"].to_list()[i].split(", ")) - set(to_concat))))
  zipped = list(zip(cat_col_names, spatial_col_names, temporal_col_names, misc_col_names))
  df_cols = pd.DataFrame(zipped, columns=['cat_col_names', 'spatial_col_names', 'temporal_col_names', 'misc_col_names'])
  return df_cols