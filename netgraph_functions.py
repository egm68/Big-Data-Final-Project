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

#from https://github.com/paulbrodersen/netgraph/blob/e54e31441f8258411ff7e2fd6da0d3139487e6fe/netgraph/_node_layout.py#L1679 
def get_geometric_layout(edges, edge_length, node_size=0., tol=1e-3, origin=(0, 0), scale=(1, 1), pad_by=0.05):
    """Node layout for defined edge lengths but unknown node positions.
    Node positions are determined through non-linear optimisation: the
    total distance between nodes is maximised subject to the constraint
    imposed by the edge lengths, which are used as upper bounds.
    If provided, node sizes are used to set lower bounds to minimise collisions.
    ..note:: This implementation is slow.
    Parameters
    ----------
    edges : list
        The edges of the graph, with each edge being represented by a (source node ID, target node ID) tuple.
    edge_lengths : dict
        Mapping of edges to their lengths.
    node_size : scalar or dict, default 0.
        Size (radius) of nodes.
        Providing the correct node size minimises the overlap of nodes in the graph,
        which can otherwise occur if there are many nodes, or if the nodes differ considerably in size.
    tolerance : float, default 1e-3
        The tolerance of the cost function. Small values increase the accuracy, large values improve the computation time.
    origin : tuple, default (0, 0)
        The (float x, float y) coordinates corresponding to the lower left hand corner of the bounding box specifying the extent of the canvas.
    scale : tuple, default (1, 1)
        The (float x, float y) dimensions representing the width and height of the bounding box specifying the extent of the canvas.
    pad_by : float, default 0.05
        Padding around node positions to reduce clipping of the node artists with the frame,
        and to create space for routing curved edges including self-loops around nodes.
        This results in the following bounding box:
        xmin, xmax = origin[0] + pad_by * scale[0], origin[0] + scale[0] - pad_by * scale[0]
        ymin, ymax = origin[1] + pad_by * scale[1], origin[1] + scale[1] - pad_by * scale[1]
    Returns
    -------
    node_positions : dict
        Dictionary mapping each node ID to (float x, float y) tuple, the node position.
    """

    # TODO: assert triangle inequality is not violated.
    # HOLD: probably not necessary, as minimisation can still proceed when triangle inequality is violated.

    # assert that the edges fit within the canvas dimensions
    width, height = scale
    max_length = np.sqrt(width**2 + height**2)
    too_long = dict()
    for edge, length in edge_length.items():
        if length > max_length:
            too_long[edge] = length
    if too_long:
        msg = f"The following edges exceed the dimensions of the canvas (`scale={scale}`):"
        for edge, length in too_long.items():
            msg += f"\n\t{edge} : {length}"
        msg += "\nEither increase the `scale` parameter, or decrease the edge lengths."
        raise ValueError(msg)

    # ensure that graph is bi-directional
    edges = edges + [(target, source) for (source, target) in edges] # forces copy
    edges = list(set(edges))

    # upper bound: pairwise distance matrix with unknown distances set to the maximum possible distance given the canvas dimensions

    lengths = []
    for (source, target) in edges:
        if (source, target) in edge_length:
            lengths.append(edge_length[(source, target)])
        else:
            lengths.append(edge_length[(target, source)])

    sources, targets = zip(*edges)
    nodes = sources + targets
    unique_nodes = set(nodes)
    indices = range(len(unique_nodes))
    node_to_idx = dict(zip(unique_nodes, indices))
    source_indices = [node_to_idx[source] for source in sources]
    target_indices = [node_to_idx[target] for target in targets]

    total_nodes = len(unique_nodes)
    distance_matrix = np.full((total_nodes, total_nodes), max_length)
    distance_matrix[source_indices, target_indices] = lengths
    distance_matrix[np.diag_indices(total_nodes)] = 0
    upper_bounds = squareform(distance_matrix)

    # lower bound: sum of node sizes
    if isinstance(node_size, (int, float)):
        sizes = node_size * np.ones((total_nodes))
    elif isinstance(node_size, dict):
        sizes = np.array([node_size[node] if node in node_size else 0. for node in unique_nodes])

    sum_of_node_sizes = sizes[np.newaxis, :] + sizes[:, np.newaxis]
    sum_of_node_sizes -= np.diag(np.diag(sum_of_node_sizes)) # squareform requires zeros on diagonal
    lower_bounds = squareform(sum_of_node_sizes)
    invalid = lower_bounds > upper_bounds
    lower_bounds[invalid] = upper_bounds[invalid] - 1e-8

    # For an extended discussion of this cost function and alternatives see:
    # https://stackoverflow.com/q/75137677/2912349
    def cost_function(positions):
        return 1 / np.sum(np.log(pdist(positions.reshape((-1, 2))) + 1))
        #return 1. / np.sum((pdist(positions.reshape((-1, 2))))**0.2)

    def constraint_function(positions):
        positions = np.reshape(positions, (-1, 2))
        return pdist(positions)

    initial_positions = _initialise_geometric_node_layout(edges, edge_length)
    nonlinear_constraint = NonlinearConstraint(constraint_function, lb=lower_bounds, ub=upper_bounds, jac='2-point')
    result = minimize(
        cost_function,
        initial_positions.flatten(),
        method='SLSQP',
        jac='2-point',
        constraints=[nonlinear_constraint],
        options=dict(ftol=tol),
    )

    if not result.success:
        print("Warning: could not compute valid node positions for the given edge lengths.")
        print(f"scipy.optimize.minimize: {result.message}.")

    node_positions_as_array = result.x.reshape((-1, 2))
    node_positions_as_array = _fit_to_frame(node_positions_as_array, np.array(origin), np.array(scale), pad_by)
    node_positions = dict(zip(unique_nodes, node_positions_as_array))
    return node_positions


def _initialise_geometric_node_layout(edges, edge_length):
    """Initialises the node positions using the FR algorithm with weights.
    Shorter edges are given a larger weight such that the nodes experience a strong attractive force."""

    edge_weight = dict()
    for edge, length in edge_length.items():
        edge_weight[edge] = 1 / length
    node_positions = get_fruchterman_reingold_layout(edges, edge_weight=edge_weight)
    return np.array(list(node_positions.values()))


def _flatten(nested_list):
    """Flatten a nested list."""
    return [item for sublist in nested_list for item in sublist]


def _get_unique_nodes(edges):
    """
    Parameters
    ----------
    edges: list of tuple
        Edge list of the graph.

    Returns
    -------
    nodes: list
        List of unique nodes.

    Notes
    -----
    We cannot use numpy.unique, as it promotes nodes to numpy.float/numpy.int/numpy.str,
    and breaks for nodes that have a more complicated type such as a tuple.

    """
    return list(set(_flatten(edges)))


def get_fruchterman_reingold_layout(edges,
                                    edge_weights        = None,
                                    k                   = None,
                                    origin              = (0, 0),
                                    scale               = (1, 1),
                                    pad_by              = 0.05,
                                    initial_temperature = 1.,
                                    total_iterations    = 50,
                                    node_size           = 0,
                                    node_positions      = None,
                                    fixed_nodes         = None,
                                    *args, **kwargs):
    """'Spring' or Fruchterman-Reingold node layout.
    Uses the Fruchterman-Reingold algorithm [Fruchterman1991]_ to compute node positions.
    This algorithm simulates the graph as a physical system, in which nodes repell each other.
    For connected nodes, this repulsion is counteracted by an attractive force exerted by the edges, which are simulated as springs.
    The resulting layout is hence often referred to as a 'spring' layout.
    Parameters
    ----------
    edges : list
        The edges of the graph, with each edge being represented by a (source node ID, target node ID) tuple.
    edge_weights : dict
        Mapping of edges to edge weights.
    k : float or None, default None
        Expected mean edge length. If None, initialized to the sqrt(area / total nodes).
    origin : tuple, default (0, 0)
        The (float x, float y) coordinates corresponding to the lower left hand corner of the bounding box specifying the extent of the canvas.
    scale : tuple, default (1, 1)
        The (float x, float y) dimensions representing the width and height of the bounding box specifying the extent of the canvas.
    pad_by : float, default 0.05
        Padding around node positions to reduce clipping of the node artists with the frame,
        and to create space for routing curved edges including self-loops around nodes.
        This results in the following bounding box:
        :code:`xmin = origin[0] + pad_by * scale[0]`
        :code:`xmax = origin[0] + scale[0] - pad_by * scale[0]`
        :code:`ymin = origin[1] + pad_by * scale[1]`
        :code:`ymax = origin[1] + scale[1] - pad_by * scale[1]`
    total_iterations : int, default 50
        Number of iterations.
    initial_temperature: float, default 1.
        Temperature controls the maximum node displacement on each iteration.
        Temperature is decreased on each iteration to eventually force the algorithm into a particular solution.
        The size of the initial temperature determines how quickly that happens.
        Values should be much smaller than the values of `scale`.
    node_size : scalar or dict, default 0.
        Size (radius) of nodes.
        Providing the correct node size minimises the overlap of nodes in the graph,
        which can otherwise occur if there are many nodes, or if the nodes differ considerably in size.
    node_positions : dict or None, default None
        Mapping of nodes to their (initial) x,y positions. If None are given,
        nodes are initially placed randomly within the bounding box defined by `origin` and `scale`.
        If the graph has multiple components, explicit initial positions may result in a ValueError,
        if the initial positions fall outside of the area allocated to that specific component.
    fixed_nodes : list or None, default None
        Nodes to keep fixed at their initial positions.
    Returns
    -------
    node_positions : dict
        Dictionary mapping each node ID to (float x, float y) tuple, the node position.
    References
    ----------
    .. [Fruchterman1991] Fruchterman, TMJ and Reingold, EM (1991) ‘Graph drawing by force‐directed placement’,
       Software: Practice and Experience
    """

    assert len(edges) > 0, "The list of edges has to be non-empty."

    # This is just a wrapper around `_fruchterman_reingold`, which implements (the loop body of) the algorithm proper.
    # This wrapper handles the initialization of variables to their defaults (if not explicitely provided),
    # and checks inputs for self-consistency.

    origin = np.array(origin)
    scale = np.array(scale)
    assert len(origin) == len(scale), \
        "Arguments `origin` (d={}) and `scale` (d={}) need to have the same number of dimensions!".format(len(origin), len(scale))
    dimensionality = len(origin)

    if fixed_nodes is None:
        fixed_nodes = []

    connected_nodes = _get_unique_nodes(edges)

    if node_positions is None: # assign random starting positions to all nodes
        node_positions_as_array = np.random.rand(len(connected_nodes), dimensionality) * scale + origin
        unique_nodes = connected_nodes

    else:
        # 1) check input dimensionality
        dimensionality_node_positions = np.array(list(node_positions.values())).shape[1]
        assert dimensionality_node_positions == dimensionality, \
            "The dimensionality of values of `node_positions` (d={}) must match the dimensionality of `origin`/ `scale` (d={})!".format(dimensionality_node_positions, dimensionality)

        is_valid = _is_within_bbox(list(node_positions.values()), origin=origin, scale=scale)
        if not np.all(is_valid):
            error_message = "Some given node positions are not within the data range specified by `origin` and `scale`!"
            error_message += "\n\tOrigin : {}, {}".format(*origin)
            error_message += "\n\tScale  : {}, {}".format(*scale)
            error_message += "\nThe following nodes do not fall within this range:"
            for ii, (node, position) in enumerate(node_positions.items()):
                if not is_valid[ii]:
                    error_message += "\n\t{} : {}".format(node, position)
            error_message += "\nThis error can occur if the graph contains multiple components but some or all node positions are initialised explicitly (i.e. node_positions != None)."
            raise ValueError(error_message)

        # 2) handle discrepancies in nodes listed in node_positions and nodes extracted from edges
        if set(node_positions.keys()) == set(connected_nodes):
            # all starting positions are given;
            # no superfluous nodes in node_positions;
            # nothing left to do
            unique_nodes = connected_nodes
        else:
            # some node positions are provided, but not all
            for node in connected_nodes:
                if not (node in node_positions):
                    warnings.warn("Position of node {} not provided. Initializing to random position within frame.".format(node))
                    node_positions[node] = np.random.rand(2) * scale + origin

            unconnected_nodes = []
            for node in node_positions:
                if not (node in connected_nodes):
                    unconnected_nodes.append(node)
                    fixed_nodes.append(node)
                    # warnings.warn("Node {} appears to be unconnected. The current node position will be kept.".format(node))

            unique_nodes = connected_nodes + unconnected_nodes

        node_positions_as_array = np.array([node_positions[node] for node in unique_nodes])

    total_nodes = len(unique_nodes)

    if isinstance(node_size, (int, float)):
        node_size = node_size * np.ones((total_nodes))
    elif isinstance(node_size, dict):
        node_size = np.array([node_size[node] if node in node_size else 0. for node in unique_nodes])

    adjacency = _edge_list_to_adjacency_matrix(
        edges, edge_weights=edge_weights, unique_nodes=unique_nodes)

    # Forces in FR are symmetric.
    # Hence we need to ensure that the adjacency matrix is also symmetric.
    adjacency = adjacency + adjacency.transpose()

    if fixed_nodes:
        is_mobile = np.array([False if node in fixed_nodes else True for node in unique_nodes], dtype=bool)

        mobile_positions = node_positions_as_array[is_mobile]
        fixed_positions = node_positions_as_array[~is_mobile]

        mobile_node_sizes = node_size[is_mobile]
        fixed_node_sizes = node_size[~is_mobile]

        # reorder adjacency
        total_mobile = np.sum(is_mobile)
        reordered = np.zeros((adjacency.shape[0], total_mobile))
        reordered[:total_mobile, :total_mobile] = adjacency[is_mobile][:, is_mobile]
        reordered[total_mobile:, :total_mobile] = adjacency[~is_mobile][:, is_mobile]
        adjacency = reordered
    else:
        is_mobile = np.ones((total_nodes), dtype=bool)

        mobile_positions = node_positions_as_array
        fixed_positions = np.zeros((0, 2))

        mobile_node_sizes = node_size
        fixed_node_sizes = np.array([])

    if k is None:
        area = np.product(scale)
        k = np.sqrt(area / float(total_nodes))

    temperatures = _get_temperature_decay(initial_temperature, total_iterations)

    # --------------------------------------------------------------------------------
    # main loop

    for ii, temperature in enumerate(temperatures):
        candidate_positions = _fruchterman_reingold(mobile_positions, fixed_positions,
                                                    mobile_node_sizes, fixed_node_sizes,
                                                    adjacency, temperature, k)
        is_valid = _is_within_bbox(candidate_positions, origin=origin, scale=scale)
        mobile_positions[is_valid] = candidate_positions[is_valid]

    # --------------------------------------------------------------------------------
    # format output

    node_positions_as_array[is_mobile] = mobile_positions

    if np.all(is_mobile):
        node_positions_as_array = _fit_to_frame(node_positions_as_array, origin, scale, pad_by)

    node_positions = dict(zip(unique_nodes, node_positions_as_array))

    return node_positions


def _edge_list_to_adjacency_matrix(edges, edge_weights=None, unique_nodes=None):
    """Convert an edge list representation of a graph into the corresponding full rank adjacency matrix.
    Parameters
    ----------
    edges : list of tuple
        List of edges; each edge is identified by a (v1, v2) node tuple.
    edge_weights : list of int or float, optional
        List of corresponding edge weights.
    unique_nodes : list
        List of unique nodes. Required if any node is unconnected.
    Returns
    -------
    adjacency_matrix : numpy.array
        The full rank adjacency/weight matrix.
    """

    sources = [s for (s, _) in edges]
    targets = [t for (_, t) in edges]
    if edge_weights:
        weights = [edge_weights[edge] for edge in edges]
    else:
        weights = np.ones((len(edges)))

    if unique_nodes is None:
        # map nodes to consecutive integers
        nodes = sources + targets
        unique_nodes = set(nodes)

    indices = range(len(unique_nodes))
    node_to_idx = dict(zip(unique_nodes, indices))

    source_indices = [node_to_idx[source] for source in sources]
    target_indices = [node_to_idx[target] for target in targets]

    total_nodes = len(unique_nodes)
    adjacency_matrix = np.zeros((total_nodes, total_nodes))
    adjacency_matrix[source_indices, target_indices] = weights

    return adjacency_matrix


def _get_fr_repulsion(distance, direction, k):
    """Compute repulsive forces."""
    with np.errstate(divide='ignore', invalid='ignore'):
        magnitude = k**2 / distance
    vectors = direction * magnitude[..., None]
    # Note that we cannot apply the usual strategy of summing the array
    # along either axis and subtracting the trace,
    # as the diagonal of `direction` is np.nan, and any sum or difference of
    # NaNs is just another NaN.
    # Also we do not want to ignore NaNs by using np.nansum, as then we would
    # potentially mask the existence of off-diagonal zero distances.
    for ii in range(vectors.shape[-1]):
        np.fill_diagonal(vectors[:, :, ii], 0)
    return np.sum(vectors, axis=0)


def _get_fr_attraction(distance, direction, adjacency, k):
    """Compute attractive forces."""
    magnitude = 1./k * distance**2 * adjacency
    vectors = -direction * magnitude[..., None] # NB: the minus!
    for ii in range(vectors.shape[-1]):
        np.fill_diagonal(vectors[:, :, ii], 0)
    return np.sum(vectors, axis=0)


def _fruchterman_reingold(mobile_positions, fixed_positions,
                          mobile_node_radii, fixed_node_radii,
                          adjacency, temperature, k):
    """Inner loop of Fruchterman-Reingold layout algorithm."""

    combined_positions = np.concatenate([mobile_positions, fixed_positions], axis=0)
    combined_node_radii = np.concatenate([mobile_node_radii, fixed_node_radii])

    delta = mobile_positions[np.newaxis, :, :] - combined_positions[:, np.newaxis, :]
    distance = np.linalg.norm(delta, axis=-1)

    # alternatively: (hack adapted from igraph)
    if np.sum(distance==0) - np.trace(distance==0) > 0: # i.e. if off-diagonal entries in distance are zero
        warnings.warn("Some nodes have the same position; repulsion between the nodes is undefined.")
        rand_delta = np.random.rand(*delta.shape) * 1e-9
        is_zero = distance <= 0
        delta[is_zero] = rand_delta[is_zero]
        distance = np.linalg.norm(delta, axis=-1)

    # subtract node radii from distances to prevent nodes from overlapping
    distance -= mobile_node_radii[np.newaxis, :] + combined_node_radii[:, np.newaxis]

    # prevent distances from becoming less than zero due to overlap of nodes
    distance[distance <= 0.] = 1e-6 # 1e-13 is numerical accuracy, and we will be taking the square shortly

    with np.errstate(divide='ignore', invalid='ignore'):
        direction = delta / distance[..., None] # i.e. the unit vector

    # calculate forces
    repulsion    = _get_fr_repulsion(distance, direction, k)
    attraction   = _get_fr_attraction(distance, direction, adjacency, k)
    displacement = attraction + repulsion

    # limit maximum displacement using temperature
    displacement_length = np.linalg.norm(displacement, axis=-1)
    displacement = displacement / displacement_length[:, None] * np.clip(displacement_length, None, temperature)[:, None]

    mobile_positions = mobile_positions + displacement

    return mobile_positions


def _get_temperature_decay(initial_temperature, total_iterations, mode='quadratic', eps=1e-9):
    """Compute all temperature values for a given initial temperature and decay model."""
    x = np.linspace(0., 1., total_iterations)
    if mode == 'quadratic':
        y = (x - 1.)**2 + eps
    elif mode == 'linear':
        y = (1. - x) + eps
    else:
        raise ValueError("Argument `mode` one of: 'linear', 'quadratic'.")
    return initial_temperature * y


def _is_within_bbox(points, origin, scale):
    """Check if each of the given points is within the bounding box given by origin and scale."""
    minima = np.array(origin)
    maxima = minima + np.array(scale)
    return np.all(np.logical_and(points >= minima, points <= maxima), axis=1)


def _fit_to_frame(positions, origin, scale, pad_by):
    """Rotate, rescale and shift a set of positions such that they fit
    inside a frame while preserving the relative distances between
    them."""

    # find major axis
    delta = positions[np.newaxis, :] - positions[:, np.newaxis]
    distances = np.sum(delta**2, axis=-1)
    ii, jj = np.where(np.triu(distances)==np.max(distances))

    # use the first if there are several solutions
    ii = ii[0]
    jj = jj[0]

    # pivot around half-way point
    pivot = positions[ii] + 0.5 * delta[ii, jj]
    angle = _get_angle(*delta[ii, jj])

    if scale[0] < scale[1]: # portrait
        rotated_positions = _rotate((np.pi/2 - angle) % np.pi, positions, pivot)
    else: # landscape
        rotated_positions = _rotate(-angle % np.pi, positions, pivot)

    # shift to (0, 0)
    shifted_positions = rotated_positions - np.min(rotated_positions, axis=0)[np.newaxis, :]

    # rescale & center
    dx, dy = np.ptp(rotated_positions, axis=0)
    if dx/scale[0] < dy/scale[1]:
        rescaled_positions = shifted_positions * (1 - 2 * pad_by) * scale[1] / dy
        rescaled_positions[:, 0] += (scale[0] - np.ptp(rescaled_positions[:, 0])) / 2
        rescaled_positions[:, 1] += pad_by * scale[1]
    else:
        rescaled_positions = shifted_positions * (1 - 2 * pad_by) * scale[0] / dx
        rescaled_positions[:, 0] += pad_by * scale[0]
        rescaled_positions[:, 1] += (scale[1] - np.ptp(rescaled_positions[:, 1])) / 2

    # shift to origin
    reshifted_positions = rescaled_positions + np.array(origin)[np.newaxis, :]

    return reshifted_positions


def _get_angle(dx, dy, radians=False):
    """Angle of a vector in 2D."""
    angle = np.arctan2(dy, dx)
    if radians:
        angle *= 360 / (2.0 * np.pi)
    return angle


def _rotate(angle, points, origin=(0, 0)):
    # https://stackoverflow.com/a/58781388/2912349
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    origin = np.atleast_2d(origin)
    points = np.atleast_2d(points)
    return np.squeeze((R @ (points.T-origin.T) + origin.T).T)