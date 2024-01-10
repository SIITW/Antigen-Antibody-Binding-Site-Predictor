from torch.utils.data import Dataset
import numpy as np
import random
import os

import torch

from plotly import graph_objs as go
from plotly.subplots import make_subplots
import plotly
#
# from collections import defaultdict

# import torchvision.transforms as transforms
from scipy import ndimage

def vis_patch(self, ppi, html_path=None, attn=None):
    feature_pairs = {
            'shape_index': (0, 5),
            'ddc': (1, 6),
            'electrostatics': (2, 7),
            'charge': (3, 8),
            'hydrophobicity': (4, 9),
            'RASA': (11, 12),
            'patch_dist': (10,),
       }
    grid_dir = self.grid_dir
    resnames_path = grid_dir + ppi + '_resnames.npy'
    patch_path = grid_dir + ppi + '.npy'
    patch_np = np.load(patch_path, allow_pickle=True)

    patch_resnames = np.load(resnames_path, allow_pickle=True)
    # patch_resnames = patch_resnames[:,:,0]
    n_feat = int(patch_np.shape[-1] / 2)
    key_names = list(feature_pairs.keys())
    fig = make_subplots(2, n_feat,
                            subplot_titles=key_names[:n_feat])

    patch_dist = patch_np[:, :, feature_pairs['patch_dist']].reshape((patch_np.shape[0], patch_np.shape[1]))
    patch_dist = np.round(patch_dist, 2)
    for col_i in range(n_feat):
        for row_i, pair_i in enumerate(feature_pairs[key_names[col_i]]):
            patch_i = patch_np[:, :, pair_i]
            if attn is not None:
                #mask = (attn+0.01)*(attn==0) + attn
                mask = (attn>0) * attn
                patch_i = patch_i * mask

            customdata = np.stack([patch_resnames[:, :, row_i], patch_dist], axis=-1)

            fig.add_trace(go.Heatmap(
                    z=patch_i,
                    customdata=customdata,
                    hovertemplate='<b>Value:%{z:.3f}</b><br>Amino Acid:%{customdata[0]}; dist:%{customdata[1]}',
                    name='',
                    colorscale='RdBu',
                    zmid=0,
                    showscale=False,
                    showlegend=False
                )
                    ,
                    row_i + 1, col_i + 1)
        fig.update_layout(
            title_text='The interactive patch pair for {}. Hover to see the value and corresponding amino acid name.'.format(
                ppi))
        if html_path is not None:
            plotly.offline.plot(fig, filename=html_path)
        else:
            fig.show()