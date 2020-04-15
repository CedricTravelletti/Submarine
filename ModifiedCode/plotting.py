""" Module grouping the plotting functionalities.

"""
from matplotlib import lines                    # Plotting lines in graph
import matplotlib.pyplot as plt                 # Plotting


def plot_graph(
        iax, iG, G_node_positions,
        x_resolution, y_resolution,
        color='k', nodenames=True):
    """ Short summary.

    What it does.

    Parameters
    ----------
    iax
    iG
    G_node_positions
    x_resolution
    y_resolution
    color
    nodenames

    Returns
    -------

    """
    for v_i in iG:
        v_i_pos = G_node_positions[v_i]
        iax.scatter([v_i_pos[0]], [v_i_pos[1]], c=color)
        for v_j in iG[v_i]:
            v_j_pos = G_node_positions[v_j]
            plt.annotate('', v_j_pos, xytext=v_i_pos,
                         arrowprops=dict(linewidth=0, width=1,
                                         headwidth=4, headlength=10, fc="k"))
        if nodenames:
            if v_i == 0 or v_i == 2:
                iax.annotate(str(v_i),
                            xy=(v_i_pos[0], v_i_pos[1]),
                            xytext=(v_i_pos[0] - x_resolution * 0.05, v_i_pos[1] + y_resolution * 0.01),
                            verticalalignment='bottom',
                            fontsize=12,
                            color='k')
            else:
                iax.annotate(str(v_i),
                            xy=(v_i_pos[0], v_i_pos[1]),
                            xytext=(v_i_pos[0] + x_resolution * 0.01, v_i_pos[1] + y_resolution * 0.01),
                            verticalalignment='bottom',
                            fontsize=12,
                            color='r')


def plot_edge_points(iax, G_edge_points, c='b'):
    """ Short summary.

    What it does.

    Parameters
    ----------
    iax
    G_edge_points
    c

    Returns
    -------

    """
    for v in G_edge_points:
        for point in list(G_edge_points[v]):
            iax.scatter([point[0]], [point[1]], c=c)

def plot_path(iax, G_node_positions, Pat, linestyle='-'):
    """ Short summary.

    What it does.

    Parameters
    ----------
    iax
    G_node_positions
    Par
    linestyle

    Returns
    -------

    """
    v_prev = Pat[0]
    for v in Pat[1:]:
        start_p = G_node_positions[v_prev]
        end_p = G_node_positions[v]
        line_artist = lines.Line2D(
            [start_p[0], end_p[0]], [start_p[1], end_p[1]],
            color="red", linewidth=5.0,
            linestyle=linestyle, antialiased=True)
        iax.add_artist(line_artist)
        v_prev = v
