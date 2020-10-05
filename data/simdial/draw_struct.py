"""Thanks to https://gist.github.com/Zsailer/818b971cd469f6a055a2844199581795
"""


def draw_networkx_nodes_ellipses(G,
                                 pos,
                                 nodelist=None,
                                 node_height=1,
                                 node_width=2,
                                 node_angle=0,
                                 node_color='r',
                                 edge_color='b',
                                 node_shape='o',
                                 alpha=1.0,
                                 cmap=None,
                                 vmin=None,
                                 vmax=None,
                                 ax=None,
                                 linewidths=None,
                                 label=None,
                                 **kwds):
    import collections
    import networkx as nx
    from collections.abc import Iterable
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import numpy as np
    except ImportError:
        raise ImportError("Matplotlib required for draw()")
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise

    if ax is None:
        ax = plt.gca()

    if nodelist is None:
        nodelist = list(G)

    if not nodelist or len(nodelist) == 0:  # empty nodelist, no drawing
        return None

    try:
        xy = np.asarray([pos[v] for v in nodelist])
    except KeyError as e:
        raise nx.NetworkXError('Node %s has no position.' % e)
    except ValueError:
        raise nx.NetworkXError('Bad value in node positions.')

    if isinstance(alpha, collections.Iterable):
        node_color = nx.apply_alpha(node_color, alpha, nodelist, cmap, vmin,
                                    vmax)
        alpha = None

    if cmap is not None:
        cm = mpl.cm.get_cmap(cmap)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        cm = norm = None

    node_collection = mpl.collections.EllipseCollection(
        widths=node_width,
        heights=node_height,
        angles=0,
        offsets=np.array(xy),
        cmap=cm,
        norm=norm,
        transOffset=ax.transData,
        linewidths=linewidths)

    node_collection.set_alpha(alpha)
    node_collection.set_label(label)
    node_collection.set_facecolor(node_color)
    node_collection.set_edgecolor(edge_color)
    node_collection.set_zorder(2)
    ax.add_collection(node_collection)
    ax.autoscale_view()

    return node_collection