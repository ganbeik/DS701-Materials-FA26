from IPython.display import display_html
from IPython.display import display
import matplotlib.pyplot as plt
from IPython.display import HTML
import binascii
import base64

def hide_code_in_slideshow():
    import os
    uid = binascii.hexlify(os.urandom(8)).decode('UTF-8')
    html = """<div id="%s"></div>
    <script type="text/javascript">
        $(function(){
            var p = $("#%s");
            if (p.length==0) return;

            while (!p.hasClass("cell")) {
                p=p.parent();

                if (p.prop("tagName") =="body") return;
            }
            var cell = p;
            cell.find(".input").addClass("hide-in-slideshow")
        });
    </script>""" % (uid, uid)
    display_html(html, raw=True)
    return html

##########################
# python notebook does not support matplotlib animations
# so these functions create an .mp4 video and then display
# it using inline HTML
##########################

# Source: http://nbviewer.ipython.org/url/jakevdp.github.io/downloads/notebooks/AnimationEmbedding.ipynb


# updated to python3, now uses package base64 to do the encoding, reading comments from this post:
# http://jakevdp.github.io/blog/2013/05/12/embedding-matplotlib-animations/
from tempfile import NamedTemporaryFile

VIDEO_TAG = """<video controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""

def anim_to_html(anim):
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=20, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = base64.b64encode(video).decode('utf-8')
    
    return VIDEO_TAG.format(anim._encoded_video)

def display_animation(anim):
    # plt.close(anim._fig)
    return HTML(anim_to_html(anim))

def display_saved_anim(fname):
    with open(fname,'rb') as f:
        video = f.read()
    return HTML(VIDEO_TAG.format(base64.b64encode(video).decode('utf-8')))


# ---------------------------------------------------------------------------
# Colour-blind-safe palette for classification plots (lectures 09, 13, ...).
#
# Okabe & Ito (2008) colours: distinguishable under the common forms of
# red-green colour blindness, unlike matplotlib's `prism` / pure red-green
# ListedColormaps these replace.  Use `CLASS_CMAP_BOLD` for points and
# `CLASS_CMAP_LIGHT` for decision regions so the two layers read as one.
# ---------------------------------------------------------------------------
from matplotlib.colors import ListedColormap, to_rgba

OKABE_ITO = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#0072B2",  # blue
    "#F0E442",  # yellow
    "#000000",  # black
]


def class_cmaps(n_classes=3, light_alpha=0.3):
    """Return (bold, light) ListedColormaps for `n_classes` classes.

    `bold` colours the training points; `light` is the same hues at
    `light_alpha` opacity for `contourf`/`pcolormesh` decision regions.
    """
    colours = OKABE_ITO[:n_classes]
    bold = ListedColormap(colours, name="class_bold")
    light = ListedColormap(
        [to_rgba(c, alpha=light_alpha) for c in colours], name="class_light"
    )
    return bold, light


CLASS_CMAP_BOLD, CLASS_CMAP_LIGHT = class_cmaps(3)


# ---------------------------------------------------------------------------
# Network helpers (lecture 21).
# ---------------------------------------------------------------------------

# Twelve distinguishable colours for categorical node labels (e.g. the twelve
# football conferences): Okabe-Ito first, then four of Paul Tol's muted hues.
CATEGORICAL_12 = OKABE_ITO + ["#999999", "#882255", "#44AA99", "#999933"]


def draw_centrality(G, cent, title, seed=9, cmap="plasma"):
    """Colour the nodes of `G` by the dict `cent`, circular and spring layouts side by side.

    One shared colourbar is drawn under both panels so the two views can be
    compared node for node.
    """
    import networkx as nx

    values = [cent[v] for v in G]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    layouts = [("circular", nx.circular_layout(G)),
               ("spring", nx.spring_layout(G, seed=seed))]
    for ax, (name, pos) in zip(axes, layouts):
        nodes = nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color=values,
                                       cmap=cmap, node_size=150)
        nx.draw_networkx_edges(G, pos=pos, ax=ax, alpha=0.4)
        ax.set_title(f"{title} ({name} layout)")
        ax.axis("off")
    cbar = fig.colorbar(nodes, ax=list(axes), orientation="horizontal",
                        fraction=0.05, pad=0.05)
    cbar.set_label(title)
    return fig
