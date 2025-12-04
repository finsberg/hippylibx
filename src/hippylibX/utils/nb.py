"""
Plotting utilities for notebooks
"""

# Copyright (c) 2016-2018, The University of Texas at Austin
# & University of California--Merced.
# Copyright (c) 2019-2020, The University of Texas at Austin
# University of California--Merced, Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.
import petsc4py.typing

import dolfinx as dlx
import matplotlib.colors as cls
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from matplotlib import animation


def _mesh2triang(mesh):
    xy = mesh.geometry.x
    cells = mesh.topology.connectivity(mesh.geometry.dim, 0).array.reshape((-1, 3))
    return tri.Triangulation(xy[:, 0], xy[:, 1], cells)


def _mplot_cellfunction(cellfn):
    C = cellfn.values
    tri = _mesh2triang(cellfn.mesh())
    return plt.tripcolor(tri, facecolors=C)


def _mplot_function(f, vmin, vmax, logscale):
    mesh = f.function_space.mesh

    mesh_dim = mesh.geometry.dim
    if mesh_dim != 2:
        raise AttributeError("Mesh must be 2D")

    # DG0 cellwise function
    if f.x.array.size == mesh.topology.index_map(mesh_dim).size_local:
        C = f.x.array.get_local()
        if logscale:
            return plt.tripcolor(_mesh2triang(mesh), C, vmin=vmin, vmax=vmax, norm=cls.LogNorm())
        else:
            return plt.tripcolor(_mesh2triang(mesh), C, vmin=vmin, vmax=vmax)
    # Scalar function, interpolated to vertices
    elif f.x.block_size == 1:
        V = dlx.fem.functionspace(mesh, ("Lagrange", 1))
        f_vertex = dlx.fem.Function(V)
        f_vertex.interpolate(f)
        C = f_vertex.x.array[:]
        if logscale:
            return plt.tripcolor(_mesh2triang(mesh), C, vmin=vmin, vmax=vmax, norm=cls.LogNorm())
        else:
            return plt.tripcolor(_mesh2triang(mesh), C, shading="gouraud", vmin=vmin, vmax=vmax)
    # Vector function, interpolated to vertices
    elif f.x.block_size == 2:
        V = dlx.fem.functionspace(mesh, ("Lagrange", 1, (mesh_dim,)))
        coords = V.tabulate_dof_coordinates()
        f_vertex = dlx.fem.Function(V)
        f_vertex.interpolate(f)
        w0 = f_vertex.x.array[:]
        num_vertices = mesh.topology.index_map(0).size_local
        if len(w0) != 2 * num_vertices:
            raise AttributeError("Vector field must be 2D")
        X = coords[:, 0]
        Y = coords[:, 1]
        U = w0[::2]
        V = w0[1::2]
        C = np.sqrt(U * U + V * V)
        return plt.quiver(X, Y, U, V, C, units="x", headaxislength=7, headwidth=7, headlength=7, scale=4, pivot="middle")


def plot(
    obj,
    colorbar=True,
    subplot_loc=None,
    mytitle=None,
    show_axis="off",
    vmin=None,
    vmax=None,
    logscale=False,
    aspect=None,
    cmap=None,
    fontsize=20,
):
    """
    Plot a generic dolfin object (if supported)
    """
    if subplot_loc is not None:
        plt.subplot(subplot_loc)
    #    plt.gca().set_aspect('equal')
    if isinstance(obj, dlx.fem.Function):
        pp = _mplot_function(obj, vmin, vmax, logscale)
    elif isinstance(obj, dlx.mesh.MeshTags):
        pp = _mplot_cellfunction(obj)
    elif isinstance(obj, dlx.mesh.Mesh):
        if obj.geometry().dim() != 2:
            raise AttributeError("Mesh must be 2D")
        pp = plt.triplot(_mesh2triang(obj), color="#808080")
        colorbar = False
    else:
        raise AttributeError("Failed to plot %s" % type(obj))

    plt.axis(show_axis)

    if colorbar:
        plt.colorbar(pp, fraction=0.1, pad=0.2)

    if aspect is not None:
        plt.gca().set_aspect(aspect)

    if mytitle is not None:
        plt.title(mytitle, fontsize=fontsize)

    if cmap:
        plt.set_cmap(cmap)
    else:
        plt.set_cmap("viridis")

    return pp


def multi1_plot(objs, titles, same_colorbar=True, show_axis="off", logscale=False, vmin=None, vmax=None, cmap=None, fontsize=20):
    """
    Plot a list of generic dolfin object in a single row
    """
    if vmin is None and vmax is None and same_colorbar:
        vmin = 1e30
        vmax = -1e30
        for f in objs:
            if isinstance(f, dlx.fem.Function):
                fmin = f.x.array.min()
                fmax = f.x.array.max()
                if fmin < vmin:
                    vmin = fmin
                if fmax > vmax:
                    vmax = fmax

    nobj = len(objs)
    if nobj == 1:
        plt.figure(figsize=(7.5, 5))
        subplot_loc = 110
    elif nobj == 2:
        plt.figure(figsize=(13, 5))
        subplot_loc = 120
    elif nobj == 3:
        plt.figure(figsize=(18, 4))
        subplot_loc = 130
    else:
        raise AttributeError("Too many figures")

    for i in range(nobj):
        try:
            cmapi = cmap[i]
        except (TypeError, IndexError):
            cmapi = cmap

        plot(
            objs[i],
            colorbar=True,
            subplot_loc=(subplot_loc + i + 1),
            mytitle=titles[i],
            show_axis="off",
            vmin=vmin,
            vmax=vmax,
            logscale=logscale,
            cmap=cmapi,
            fontsize=fontsize,
        )


def plot_pts(
    points,
    values,
    colorbar=True,
    subplot_loc=None,
    mytitle=None,
    show_axis="on",
    vmin=None,
    vmax=None,
    xlim=(0, 1),
    ylim=(0, 1),
    cmap=None,
):
    """
    Plot a cloud of points
    """
    if subplot_loc is not None:
        plt.subplot(subplot_loc)

    pp = plt.scatter(points[:, 0], points[:, 1], c=values.get_local(), marker=",", s=20, vmin=vmin, vmax=vmax)

    plt.axis(show_axis)

    if colorbar:
        plt.colorbar(pp, fraction=0.1, pad=0.2)
    else:
        plt.gca().set_aspect("equal")

    if mytitle is not None:
        plt.title(mytitle, fontsize=20)

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)

    if cmap:
        plt.set_cmap(cmap)
    else:
        plt.set_cmap("viridis")

    return pp


def show_solution(
    Vh,
    ic,
    state,
    same_colorbar=True,
    colorbar=True,
    mytitle=None,
    show_axis="off",
    logscale=False,
    times=[0, 0.4, 1.0, 2.0, 3.0, 4.0],
    cmap=None,
):
    """
    Plot a :code:TimeDependentVector at specified time steps
    """
    state.store(ic, 0)
    nrows = int(np.ceil(len(times) / 3.0))
    subplot_loc = nrows * 100 + 30
    plt.figure(figsize=(18, 4 * nrows))

    if mytitle is None:
        title_stamp = "Time {0}s"
    else:
        title_stamp = mytitle + " at time {0}s"

    vmin = None
    vmax = None

    if same_colorbar:
        vmin = 1e30
        vmax = -1e30
        for s in state.data:
            smax = s.max()
            smin = s.min()
            if smax > vmax:
                vmax = smax
            if smin < vmin:
                vmin = smin

    counter = 1
    myu = dlx.fem.Function(Vh)
    for i in times:
        try:
            state.retrieve(myu.vector(), i)
            plot(
                myu,
                subplot_loc=(subplot_loc + counter),
                mytitle=title_stamp.format(i),
                colorbar=colorbar,
                logscale=logscale,
                show_axis=show_axis,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
            counter = counter + 1
        except Exception:
            print("Invalid time: ", i)


def animate(Vh, state, same_colorbar=True, colorbar=True, subplot_loc=None, mytitle=None, show_axis="off", logscale=False):
    """
    Show animation for a :code:TimeDependentVector
    """

    fig = plt.figure()

    vmin = None
    vmax = None

    if same_colorbar:
        vmin = 1e30
        vmax = -1e30
        for s in state.data:
            smax = s.max()
            smin = s.min()
            if smax > vmax:
                vmax = smax
            if smin < vmin:
                vmin = smin

    def my_animate(i):
        time_stamp = "Time: {0:f} s"
        obj = dlx.fem.Function(Vh, state.data[i])
        t = mytitle + time_stamp.format(state.times[i])
        plt.clf()
        return plot(obj, colorbar=True, subplot_loc=None, mytitle=t, show_axis="off", vmin=vmin, vmax=vmax, logscale=False)

    return animation.FuncAnimation(fig, my_animate, np.arange(0, state.nsteps), blit=True)


def coarsen_v(fun, nx=16, ny=16):
    # mesh = dl.UnitSquareMesh(nx,ny)
    mesh = dlx.mesh.Mesh("ad_20.xml")
    dim = mesh.topology.dim
    V_H = dlx.fem.functionspace(mesh, ("CG", 1, (dim,)))
    dlx.parameters["allow_extrapolation"] = True
    fun_H = dlx.fem.interpolate(fun, V_H)
    dlx.parameters["allow_extrapolation"] = False
    return fun_H


def plot_eigenvalues(d, mytitle=None, subplot_loc=None):
    """
    Plot eigenvalues
    """
    k = d.shape[0]
    if subplot_loc is not None:
        plt.subplot(subplot_loc)
    plt.plot(range(0, k), d, "b*", range(0, k), np.ones(k), "-r")
    plt.yscale("log")
    if mytitle is not None:
        plt.title(mytitle)


def plot_eigenvectors(Vh, U, mytitle, which=[0, 1, 2, 5, 10, 15], cmap=None):
    """
    Plot specified vectors in a :code:MultiVector
    """
    nrows = int(np.ceil(len(which) / 3.0))
    subplot_loc = nrows * 100 + 30
    plt.figure(figsize=(18, 4 * nrows))

    title_stamp = mytitle + " {0}"
    u = dlx.fem.Function(Vh)

    counter = 1
    for i in which:
        assert i < U.nvec
        if (U[i])[0] >= 0:
            s = 1.0 / U[i].norm(petsc4py.typing.NormType.NORM_INFINITY)
        else:
            s = -1.0 / U[i].norm(petsc4py.typing.NormType.NORM_INFINITY)
        u.x.array[:] = 0.0
        u.x.petsc_vec.axpy(s, U[i])
        plot(u, subplot_loc=(subplot_loc + counter), mytitle=title_stamp.format(i), vmin=-1, vmax=1, cmap=cmap)
        counter = counter + 1
