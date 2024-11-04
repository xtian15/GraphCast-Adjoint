import numpy as np
#import jax.numpy as np
import netCDF4 as nc
import chex
import jax
import xarray
import dataclasses
from os.path import exists
from datetime import datetime, timedelta
from copy import deepcopy
from geopy.distance import distance as geoDistance
from graphcast import checkpoint
from graphcast import graphcast
from graphcast import data_utils
from graphcast.normalization import normalize, unnormalize
from graphcast import xarray_tree
from graphcast import xarray_jax
from graphcast import model_utils


import haiku as hk

# ----- FWD -----
def swish(x):
    res=x/(1.+np.exp(-x))
    return res

def ELU(x, alpha=1.0):
    res=x.copy()
    res[x<=0]=alpha*(np.exp(x[x<=0])-1.)
    return res

def linear(x):
    return x

def denseLayer(x, w, b, acti):
    res = np.matmul(x, w) + b
    return acti(res)

def denseLayers(x, ws, bs, actis):
    ix = x.copy()
    for i in range(len(ws)):
        ix = denseLayer(ix, ws[i], bs[i], actis[i])
    return ix

def var(x, axis=1):
    # res= ((x.T-x.mean(axis=axis))**2).T.mean(axis=axis)  # 1 liner
    x_mean=x.mean(axis=axis)
    x_devi=( (x.T-x_mean)**2 ).T
    res = x_devi.mean(axis=axis)
    return res

def layerNorm(x, w, b, eps=1.E-5):
    # assuming x.shape=(nCells, nHidden), e.g. nHidden=128
    
    res= ( (x.T-x.mean(axis=1))/np.sqrt(x.var(axis=1)+eps) ).T
    res = res*w + b
    return res

def layerNorm_fwd(x, w, b, eps=1.E-5):

    x_mean=x.mean(axis=1)
    res = ( (x.T-x_mean)/np.sqrt(var(x, axis=1)+eps) ).T
    res = res*w + b
    return res

def node_forward(x, edge_index, edge_attr, ws, bs, actis, w_ly=None, b_ly=None, receive=True):

    if receive:
        ### manual message passing
        row, col = edge_index
        nCells, nEdges, nHid=x.shape[0], edge_attr.shape[0], edge_attr.shape[1]
        out=np.zeros([nCells, nHid], float)
        for iEdge in range(nEdges):
            out[col[iEdge]]+=edge_attr[iEdge]
        out=np.concatenate((x, out), axis=-1)
    else:
        out=x.copy()

    out=denseLayers(out, ws, bs, actis)

    if (w_ly is not None) and (b_ly is not None):
        out=layerNorm(out, w_ly, b_ly)
    
    return out

def edge_forward(
        x_src, x_tgt, edge_index, edge_attr, ws, bs, actis, w_ly=None, b_ly=None
):
    src_idx, tgt_idx=edge_index
    source, target=x_src[src_idx,:], x_tgt[tgt_idx,:]

    out=np.concatenate((edge_attr, source, target), axis=-1)
    out=denseLayers(out, ws, bs, actis)
    if (w_ly is not None) and (b_ly is not None):
        out=layerNorm(out, w_ly, b_ly)
    #out+=edge_attr
    return out

def graph_mp(x, edge_index, edge_attr,
             ws_node, bs_node, ws_edge, bs_edge, actis,
             w_ly_node=None, b_ly_node=None, w_ly_edge=None, b_ly_edge=None,
             with_sender=None):
    """
    x: node features
    edge_index: [source, target]
    edge_attr: edge features
    """

    x_old=x.copy()
    edge_attr_old=edge_attr.copy()
    nMP=len(ws_node)  # number of message passing rounds

    if with_sender is None:
        for iMP in range(nMP):
            edge_attr_new=edge_forward(
	        x_old, x_old.copy(), edge_index, edge_attr_old, ws_edge[iMP], bs_edge[iMP], actis,
	        w_ly=w_ly_edge[iMP], b_ly=b_ly_edge[iMP],
	    )
            x_new=node_forward(
	        x_old, edge_index, edge_attr_new, ws_node[iMP], bs_node[iMP], actis,
	        w_ly=w_ly_node[iMP], b_ly=b_ly_node[iMP], receive=True,
	    )
            x_old=x_new+x_old
            edge_attr_old=edge_attr_new+edge_attr_old

        return x_old, edge_attr_old

    else:
        x_sender, ws_sender, bs_sender, w_ly_sender, b_ly_sender=with_sender
        for iMP in range(nMP):            
            edge_attr_new=edge_forward(
	    	x_sender, x_old, edge_index, edge_attr_old, ws_edge[iMP], bs_edge[iMP], actis,
	    	w_ly=w_ly_edge[iMP], b_ly=b_ly_edge[iMP],
	    )
            x_sender_new=node_forward(
	    	x_sender, edge_index, edge_attr_new, ws_sender[iMP], bs_sender[iMP], actis,
	    	w_ly=w_ly_sender[iMP], b_ly=b_ly_sender[iMP], receive=False,
	    )
            x_new = node_forward(
	    	x_old, edge_index, edge_attr_new, ws_node[iMP], bs_node[iMP], actis,
	    	w_ly=w_ly_node[iMP], b_ly=b_ly_node[iMP], receive=True,
	    )
            x_old=x_new+x_old
            edge_attr_old=edge_attr_new+edge_attr_old
            x_sender=x_sender_new+x_sender

        return x_old, edge_attr_old, x_sender

# ----- TLM -----
def swish_tlm(dx, x):
    res_tl=dx/(1.+np.exp(-x)) - x/(1.+np.exp(-x))**2 * (-np.exp(-x)*dx)
    res   =x/(1.+np.exp(-x))
    return res_tl, res

def ELU_tlm(dx, x, alpha=1.0):
    res_tl=dx.copy()
    res   =x.copy()

    res_tl[x<=0]=alpha*np.exp(x[x<=0])*dx[x<=0]
    res[x<=0]   =alpha*(np.exp(x[x<=0])-1.)

    return res_tl, res

def linear_tlm(dx, x):
    return dx, x

def denseLayer_tlm(dx, x, w, b, acti_tlm):
    res_tl=np.matmul(dx, w)
    res=np.matmul(x, w)+b
    return acti_tlm(res_tl, res)

def denseLayers_tlm(dx, x, ws, bs, acti_tlms):
    ix_tl, ix = dx.copy(), x.copy()
    for i in range(len(ws)):
        ix_tl, ix=denseLayer_tlm(ix_tl, ix, ws[i], bs[i], acti_tlms[i])
    return ix_tl, ix

def mean_tlm(dx, x, axis=1):
    return dx.mean(axis=axis), x.mean(axis=axis)

def var_tlm(dx, x, axis=1):
    x_mean_tl=mean_tlm(dx, x, axis=axis)[0]
    x_mean   =x.mean(axis=axis)  # forward calculation needed

    x_devi_tl=( 2.*(x.T-x_mean)*(dx.T-x_mean_tl) ).T
    x_devi   =( (x.T-x_mean)**2 ).T

    res_tl=mean_tlm(x_devi_tl, x_devi, axis=axis)[0]
    res = x_devi.mean(axis=axis)
    
    return res_tl, res

def layerNorm_tlm(dx, x, w, b, eps=1.E-5):

    x_mean_tl=mean_tlm(dx, x, axis=1)[0]
    x_mean   =x.mean(axis=1)

    x_var_tl =var_tlm(dx, x, axis=1)[0]
    x_var    =var(x, axis=1)
    
    res_tl=( (dx.T-x_mean_tl)/np.sqrt(x_var+eps) ).T \
        -0.5*( (x.T-x_mean)*x_var_tl/((x_var+eps)**(3./2.)) ).T
    res= ( (x.T-x.mean(axis=1))/np.sqrt(x.var(axis=1)+eps) ).T

    res_tl=res_tl*w
    res = res*w + b

    return res_tl, res

def node_forward_tlm(x_tl, x, edge_index, edge_attr_tl, edge_attr,
                     ws, bs, acti_tlms, w_ly=None, b_ly=None,
                     receive=True):
    if receive:
        row, col = edge_index
        nCells, nEdges, nHid=x.shape[0], edge_attr.shape[0], edge_attr.shape[1]

        out_tl=np.zeros([nCells, nHid], np.float64)
        out   =np.zeros([nCells, nHid], np.float64)
        for iEdge in range(nEdges):
            out_tl[col[iEdge]]+=edge_attr_tl[iEdge]
            out[col[iEdge]]+=edge_attr[iEdge]
        out_tl=np.concatenate((x_tl, out_tl), axis=-1)
        out   =np.concatenate((x, out), axis=-1)  # forward calculation needed
    else:
        out_tl=x_tl.copy()
        out=x.copy()

    out_tl, out=denseLayers_tlm(out_tl, out, ws, bs, acti_tlms)

    if (w_ly is not None) and (b_ly is not None):
        out_tl, out = layerNorm_tlm(out_tl, out, w_ly, b_ly)

    return out_tl, out

def edge_forward_tlm(
        dx_src, x_src, dx_tgt, x_tgt, edge_index, edge_attr_tl, edge_attr,
        ws, bs, acti_tlms, w_ly=None, b_ly=None
):

    src_idx, tgt_idx=edge_index
    source_tl, target_tl=dx_src[src_idx,:], dx_tgt[tgt_idx,:]
    source, target=x_src[src_idx,:], x_tgt[tgt_idx,:]

    out_tl=np.concatenate((edge_attr_tl, source_tl, target_tl), axis=-1)
    out=np.concatenate((edge_attr, source, target), axis=-1)
    out_tl, out=denseLayers_tlm(out_tl, out, ws, bs, acti_tlms)
    if (w_ly is not None) and (b_ly is not None):
        out_tl, out=layerNorm_tlm(out_tl, out, w_ly, b_ly)

    return out_tl, out

def graph_mp_tlm(x_tl, x, edge_index, edge_attr_tl, edge_attr,
                 ws_node, bs_node, ws_edge, bs_edge, acti_tlms,
                 w_ly_node=None, b_ly_node=None, w_ly_edge=None, b_ly_edge=None,
                 with_sender=None):

    nMP=len(ws_node)
    
    x_old_tl=x_tl.copy()
    x_old=x.copy()

    edge_attr_old_tl=edge_attr_tl.copy()
    edge_attr_old=edge_attr.copy()

    if with_sender is None:
        for iMP in range(nMP):
            edge_attr_new_tl, edge_attr_new=edge_forward_tlm(
                x_old_tl, x_old, x_old_tl.copy(), x_old.copy(),
                edge_index, edge_attr_old_tl, edge_attr_old,
                ws_edge[iMP], bs_edge[iMP], acti_tlms,
                w_ly=w_ly_edge[iMP], b_ly=b_ly_edge[iMP]
            )

            x_new_tl, x_new=node_forward_tlm(
                x_old_tl, x_old, edge_index, edge_attr_new_tl, edge_attr_new,
                ws_node[iMP], bs_node[iMP], acti_tlms,
                w_ly=w_ly_node[iMP], b_ly=b_ly_node[iMP],
                receive=True
            )
            
            x_old_tl=x_new_tl+x_old_tl
            x_old=x_new+x_old
            
            edge_attr_old_tl=edge_attr_new_tl+edge_attr_old_tl
            edge_attr_old=edge_attr_new+edge_attr_old

        return x_old_tl, x_old, edge_attr_old_tl, edge_attr_old
    
    else:
        x_sender_tl, x_sender, ws_sender, bs_sender, w_ly_sender, b_ly_sender=\
            with_sender
        for iMP in range(nMP):
            edge_attr_new_tl, edge_attr_new=edge_forward_tlm(
                x_sender_tl, x_sender, x_old_tl, x_old, edge_index,
                edge_attr_old_tl, edge_attr_old, ws_edge[iMP], bs_edge[iMP],
                actis_tlm, w_ly=w_ly_edge[iMP], b_ly=b_ly_edge[iMP],
            )
            x_sender_new_tl, x_sender_new=node_forward_tlm(
                x_sender_tl, x_sender, edge_index,
                edge_attr_new_tl, edge_attr_new,
                ws_sender[iMP], bs_sender[iMP], actis_tlm,
                w_ly=w_ly_sender[iMP], b_ly=b_ly_sender[iMP], receive=False,
            )
            x_new_tl, x_new = node_forward_tlm(
                x_old_tl, x_old, edge_index, edge_attr_new_tl, edge_attr_new,
                ws_node[iMP], bs_node[iMP], actis_tlm,
                w_ly=w_ly_node[iMP], b_ly=b_ly_node[iMP], receive=True,
            )
            x_old_tl=x_new_tl+x_old_tl
            x_old=x_new+x_old
            
            edge_attr_old_tl=edge_attr_new_tl+edge_attr_old_tl
            edge_attr_old=edge_attr_new+edge_attr_old
            
            x_sender_tl=x_sender_new_tl+x_sender_tl
            x_sender=x_sender_new+x_sender

        return x_old_tl, x_old, edge_attr_old_tl, edge_attr_old, x_sender_tl, x_sender
             

# ----- ADJ -----
def swish_adj(x, dy):
    dx = x/(1.+np.exp(-x))**2 * np.exp(-x) * dy
    dx = dx + 1./(1.+np.exp(-x)) * dy
    dy = 0.
    return dx

def ELU_adj(x, dy, alpha=1.0):
    dx=dy.copy()
    dx[x<=0]=alpha*np.exp(x[x<=0])*dy[x<=0]

    return dx

def linear_adj(x, dy):
    return dy

def denseLayer_adj(x, dy, w, b, acti_adj):
    # ----- forward calculation -----
    res=np.matmul(x, w)+b
    # ----- end forward calculation -----

    res_ad=acti_adj(res, dy)
    return np.matmul(res_ad, w.T)

def denseLayers_adj(x, dy, ws, bs, actis, acti_adjs):
    # ----- forward calculation -----
    ix=x.copy()
    x_fwd=[]
    for i in range(len(ws)):
        x_fwd.append(ix.copy())
        ix = denseLayer(ix, ws[i], bs[i], actis[i])
    # ----- end forward calculation -----

    iy = dy.copy()
    for i in range(len(ws)-1, -1, -1):
        iy = denseLayer_adj(x_fwd[i], iy, ws[i], bs[i], acti_adjs[i])
    return iy

def mean_adj(x, dy, axis=1):
    n=x.shape[axis]
    dx=np.zeros(x.shape)
    dy=dy/n
    dx=(dx.T+dy).T
    dy=0.
    return dx

def var_adj(x, dy, axis=1):

    # ----- forward calculation -----
    x_mean=x.mean()
    # ----- end forward calculation -----
    x_devi_ad=mean_adj(x, dy, axis=axis)
    x_mean_ad=(-2.*(x.T-x_mean).T * x_devi_ad).sum(axis=axis)
    dx=2.*(x.T-x_mean).T * x_devi_ad

    dx = dx + mean_adj(x, x_mean_ad)
    
    return dx

def layerNorm_adj(x, dy, w, b, eps=1.E-5):
    # ----- forward calculation -----
    x_mean=x.mean(axis=1)
    x_var    =var(x, axis=1)
    # ----- end forward calculation -----
    
    dy=dy*w
    
    x_var_ad = (-0.5*( (x.T-x_mean)/((x_var+eps)**(3./2.)) ).T*dy).sum(axis=1)
    x_mean_ad = (-1.*dy.T/np.sqrt(x_var+eps)).T.sum(axis=1)
    x_ad = (dy.T/np.sqrt(x_var+eps)).T
    dy=0.

    x_ad = x_ad + var_adj(x, x_var_ad)
    x_ad = x_ad + mean_adj(x, x_mean_ad)

    return x_ad

def node_forward_adj(
        out_ad, x, x_ad, edge_index, edge_attr, edge_attr_ad,
        ws, bs, actis, acti_adjs, w_ly=None, b_ly=None,
        receive=True
):
    row, col = edge_index
    nCells, nEdges, nHid=x.shape[0], edge_attr.shape[0], edge_attr.shape[1]

    if (w_ly is not None) and (b_ly is not None):

        # ----- forward calculation -----
        if receive:
            ### manual message passing
            row, col = edge_index
            nCells, nEdges, nHid=x.shape[0], edge_attr.shape[0], edge_attr.shape[1]
            out=np.zeros([nCells, nHid], float)
            for iEdge in range(nEdges):
                out[col[iEdge]]+=edge_attr[iEdge]
            out=np.concatenate((x, out), axis=-1)
        else:
            out=x.copy()

        out=denseLayers(out, ws, bs, actis)        
        # ----- end forward calculation -----
        
        out_ad = layerNorm_adj(out, out_ad, w_ly, b_ly)

    # ----- forward calculation -----
    if receive:
        ### manual message passing
        row, col = edge_index
        nCells, nEdges, nHid=x.shape[0], edge_attr.shape[0], edge_attr.shape[1]
        out=np.zeros([nCells, nHid], float)
        for iEdge in range(nEdges):
            out[col[iEdge]]+=edge_attr[iEdge]
        out=np.concatenate((x, out), axis=-1)
    else:
        out=x.copy()
    # ----- end forward calculation -----

    out_ad = denseLayers_adj(out, out_ad, ws, bs, actis, acti_adjs)

    if receive:
        x_ad = x_ad + out_ad[:,:nHid]
        out_ad = out_ad[:,nHid:]
        
        for iEdge in range(nEdges):
            edge_attr_ad[iEdge]+=out_ad[col[iEdge]]
        out_ad[()]=0.
    else:
        x_ad+=out_ad
        out_ad=0.

    return x_ad, edge_attr_ad

def edge_forward_adj(
        edge_new_ad, x_src, x_src_ad, x_tgt, x_tgt_ad,
        edge_index, edge_attr, edge_attr_ad, ws, bs,
        actis, acti_adjs, w_ly=None, b_ly=None
):
    
    nCells, nEdges, nHid=x_src.shape[0], edge_attr.shape[0], edge_attr.shape[1]
    
    src_idx, tgt_idx=edge_index
    out_ad = edge_new_ad
    
    if (w_ly is not None) and (b_ly is not None):
        # ----- forward calculation -----
        source, target=x_src[src_idx,:], x_tgt[tgt_idx,:]
        out=np.concatenate((edge_attr, source, target), axis=-1)
        out=denseLayers(out, ws, bs, actis)
        # ----- end forward calculation -----
        out_ad = layerNorm_adj(out, out_ad, w_ly, b_ly)

    # ----- forward calculation -----
    source, target=x_src[src_idx,:], x_tgt[tgt_idx,:]
    out=np.concatenate((edge_attr, source, target), axis=-1)
    # ----- end forward calculation -----
    out_ad=denseLayers_adj(out, out_ad, ws, bs, actis, acti_adjs)

    edge_attr_ad = edge_attr_ad+out_ad[:,0:nHid]
    source_ad, target_ad = out_ad[:,nHid:nHid*2], out_ad[:,nHid*2:nHid*3]

    for iEdge in range(nEdges):
        x_src_ad[src_idx[iEdge],:]+=source_ad[iEdge,:]
        x_tgt_ad[tgt_idx[iEdge],:]+=target_ad[iEdge,:]

    source_ad[()]=0.
    target_ad[()]=0.

    return x_src_ad, x_tgt_ad, edge_attr_ad

def graph_mp_adj(x, x_ad, edge_index, edge_attr, edge_attr_ad,
                 ws_node, bs_node, ws_edge, bs_edge, actis, acti_adjs,
                 w_ly_node=None, b_ly_node=None, w_ly_edge=None, b_ly_edge=None,
                 with_sender=None):
    nMP=len(ws_node)

    if with_sender is None:
        # ----- forward calculation -----
        x_old_fwd, e_old_fwd, e_new_fwd=[], [], []
        x_old=x.copy()
        edge_attr_old=edge_attr.copy()
        for iMP in range(nMP):
            x_old_fwd.append(x_old.copy())
            e_old_fwd.append(edge_attr_old.copy())
            
            edge_attr_new=edge_forward(
                x_old, x_old.copy(), edge_index, edge_attr_old, ws_edge[iMP], bs_edge[iMP], actis,
                w_ly=w_ly_edge[iMP], b_ly=b_ly_edge[iMP],
            )
            e_new_fwd.append(edge_attr_new.copy())
            
            x_new=node_forward(
                x_old, edge_index, edge_attr_new, ws_node[iMP], bs_node[iMP], actis,
                w_ly=w_ly_node[iMP], b_ly=b_ly_node[iMP], receive=True
            )
            x_old=x_new+x_old
            edge_attr_old=edge_attr_new+edge_attr_old
        # ----- end forward calculation -----

        x_old_ad=x_ad.copy()
        edge_attr_old_ad=edge_attr_ad.copy()
        for iMP in range(nMP-1,-1,-1):
            x_new_ad = x_old_ad.copy()
            edge_attr_new_ad = edge_attr_old_ad.copy()
            
            x_old_ad, edge_attr_new_ad = node_forward_adj(
                x_new_ad, x_old_fwd[iMP], x_old_ad, edge_index,
                e_new_fwd[iMP], edge_attr_new_ad,
                ws_node[iMP], bs_node[iMP], actis, acti_adjs,
                w_ly=w_ly_node[iMP], b_ly=b_ly_node[iMP], receive=True
            )

            x_src_ad = x_old_fwd[iMP]*0.
            x_tgt_ad = x_old_fwd[iMP]*0.
            x_src_ad, x_tgt_ad, edge_attr_old_ad = edge_forward_adj(
                edge_attr_new_ad, x_old_fwd[iMP], x_src_ad,
                x_old_fwd[iMP].copy(), x_tgt_ad, edge_index,
                e_old_fwd[iMP], edge_attr_old_ad,
                ws_edge[iMP], bs_edge[iMP], actis, acti_adjs,
                w_ly=w_ly_edge[iMP], b_ly=b_ly_edge[iMP]
            )
            x_old_ad = x_old_ad + x_src_ad + x_tgt_ad

        edge_attr_ad = edge_attr_old_ad.copy()
        edge_attr_old_ad[()]=0.
        
        x_ad = x_old_ad.copy()
        x_old_ad[()]=0.

        return x_ad, edge_attr_ad
    else:
        # with_sender
        x_sender_ad, x_sender, ws_sender, bs_sender, w_ly_sender, b_ly_sender=with_sender
        # ----- forward calculation -----
        x_old=x.copy()
        edge_attr_old=edge_attr.copy()
        x_sender_fwd, x_old_fwd, e_old_fwd, e_new_fwd=[], [], [], []
        for iMP in range(nMP):
            x_sender_fwd.append(x_sender.copy())
            x_old_fwd.append(x_old.copy())
            e_old_fwd.append(edge_attr_old.copy())
            
            edge_attr_new=edge_forward(
                x_sender, x_old, edge_index, edge_attr_old, ws_edge[iMP], bs_edge[iMP], actis,
                w_ly=w_ly_edge[iMP], b_ly=b_ly_edge[iMP],
            )
            
            e_new_fwd.append(edge_attr_new.copy())
            x_sender_new=node_forward(
                x_sender, edge_index, edge_attr_new, ws_sender[iMP], bs_sender[iMP], actis,
                w_ly=w_ly_sender[iMP], b_ly=b_ly_sender[iMP], receive=False,
            )
            x_new = node_forward(
                x_old, edge_index, edge_attr_new, ws_node[iMP], bs_node[iMP], actis,
                w_ly=w_ly_node[iMP], b_ly=b_ly_node[iMP], receive=True,
            )
            x_old=x_new+x_old
            edge_attr_old=edge_attr_new+edge_attr_old
            x_sender=x_sender_new+x_sender
        # ----- end forward calculation -----
        x_old_ad=x_ad.copy()
        edge_attr_old_ad=edge_attr_ad.copy()
        for iMP in range(nMP-1,-1,-1):
            x_sender_new_ad=x_sender_ad.copy()
            edge_attr_new_ad=edge_attr_old_ad.copy()
            x_new_ad=x_old_ad.copy()

            x_old_ad, edge_attr_new_ad=node_forward_adj(
                x_new_ad, x_old_fwd[iMP], x_old_ad, edge_index,
                e_new_fwd[iMP], edge_attr_new_ad,
                ws_node[iMP], bs_node[iMP], actis, acti_adjs,
                w_ly=w_ly_node[iMP], b_ly=b_ly_node[iMP], receive=True,
            )

            x_sender_ad, edge_attr_new_ad=node_forward_adj(
                x_sender_new_ad, x_sender_fwd[iMP], x_sender_ad, edge_index,
                e_new_fwd[iMP], edge_attr_new_ad,
                ws_sender[iMP], bs_sender[iMP], actis, acti_adjs,
                w_ly=w_ly_sender[iMP], b_ly=b_ly_sender[iMP], receive=False,
            )

            x_sender_ad, x_old_ad, edge_attr_old_ad=edge_forward_adj(
                edge_attr_new_ad, x_sender_fwd[iMP], x_sender_ad,
                x_old_fwd[iMP], x_old_ad, edge_index,
                e_old_fwd[iMP], edge_attr_old_ad,
                ws_edge[iMP], bs_edge[iMP], actis, acti_adjs,
                w_ly=w_ly_edge[iMP], b_ly=b_ly_edge[iMP],
            )
        return x_sender_ad, x_old_ad, edge_attr_old_ad


# ----- Tools -----
def cell2EdgeAttrs(latCell, lonCell, cellsOnCell):
    Rearth=6371
    nCells=latCell.shape[0]
    maxEdges=cellsOnCell.shape[1]
    edge_attrs=[]
    for iCell in range(nCells):
        idx=0
        while idx<maxEdges and cellsOnCell[iCell,idx]!=0:
            # cell distances: preparing for adaptive mesh
            srcCoord=(latCell[iCell], lonCell[iCell])
            tgtCoord=(latCell[cellsOnCell[iCell,idx]-1],
                      lonCell[cellsOnCell[iCell,idx]-1])
            dist=geoDistance(srcCoord, tgtCoord).km/Rearth  # in radians
            edge_attrs.append([np.sin(dist), np.cos(dist)])
            
            idx+=1
    return np.array(edge_attrs)

# ----- FWD in development -----

actis=[swish, linear]
actis_tlm=[swish_tlm, linear_tlm]
actis_adj=[swish_adj, linear_adj]
def gc_grid2mesh(params, grid_nodes, mesh_nodes, edges, edge_indices):
    #grid2mesh:
    #  INPUT : grid_nodes, mesh_nodes, edges
    #  OUTPUT: latent_grid_nodes, latent_mesh_nodes

    # ----- Encoder -----
    grid_nodes_encoded=denseLayers(
        grid_nodes,
        [
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_grid_nodes_mlp/~/linear_0']['w'],
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_grid_nodes_mlp/~/linear_1']['w'],
        ],
        [
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_grid_nodes_mlp/~/linear_0']['b'],
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_grid_nodes_mlp/~/linear_1']['b'],
        ], actis
    )
    grid_nodes_encoded=layerNorm(
        grid_nodes_encoded,
        params['grid2mesh_gnn/~_networks_builder/encoder_nodes_grid_nodes_layer_norm']['scale'],
        params['grid2mesh_gnn/~_networks_builder/encoder_nodes_grid_nodes_layer_norm']['offset'],
    )

    mesh_nodes_encoded=denseLayers(
        mesh_nodes,
        [
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_mesh_nodes_mlp/~/linear_0']['w'],
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_mesh_nodes_mlp/~/linear_1']['w'],
        ],
        [
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_mesh_nodes_mlp/~/linear_0']['b'],
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_mesh_nodes_mlp/~/linear_1']['b'],
        ], actis
    )
    mesh_nodes_encoded=layerNorm(
        mesh_nodes_encoded,
        params['grid2mesh_gnn/~_networks_builder/encoder_nodes_mesh_nodes_layer_norm']['scale'],
        params['grid2mesh_gnn/~_networks_builder/encoder_nodes_mesh_nodes_layer_norm']['offset'],
    )

    edges_encoded=denseLayers(
        edges,
        [
            params['grid2mesh_gnn/~_networks_builder/encoder_edges_grid2mesh_mlp/~/linear_0']['w'],
            params['grid2mesh_gnn/~_networks_builder/encoder_edges_grid2mesh_mlp/~/linear_1']['w'],
        ],
        [
            params['grid2mesh_gnn/~_networks_builder/encoder_edges_grid2mesh_mlp/~/linear_0']['b'],
            params['grid2mesh_gnn/~_networks_builder/encoder_edges_grid2mesh_mlp/~/linear_1']['b'],
        ],
        actis
    )
    edges_encoded=layerNorm(
        edges_encoded,
        params['grid2mesh_gnn/~_networks_builder/encoder_edges_grid2mesh_layer_norm']['scale'],
        params['grid2mesh_gnn/~_networks_builder/encoder_edges_grid2mesh_layer_norm']['offset'],
    )

    # ----- Processor -----
    sender_pkg=[
        grid_nodes_encoded,
        [[
            params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_0']['w'],
            params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_1']['w'],
        ]],
        [[
            params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_0']['b'],
            params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_1']['b'],
        ]],
        [params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_grid_nodes_layer_norm']['scale']],
        [params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_grid_nodes_layer_norm']['offset']],
    ]
    
    mesh_nodes_processed, edge_processed, grid_nodes_processed=graph_mp(
        mesh_nodes_encoded, edge_indices, edges_encoded,
        [[
            params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_0']['w'],
            params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_1']['w'],
        ]],
        [[
            params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_0']['b'],
            params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_1']['b'],
        ]],
        [[
            params['grid2mesh_gnn/~_networks_builder/processor_edges_0_grid2mesh_mlp/~/linear_0']['w'],
            params['grid2mesh_gnn/~_networks_builder/processor_edges_0_grid2mesh_mlp/~/linear_1']['w'],
        ]],
        [[
            params['grid2mesh_gnn/~_networks_builder/processor_edges_0_grid2mesh_mlp/~/linear_0']['b'],
            params['grid2mesh_gnn/~_networks_builder/processor_edges_0_grid2mesh_mlp/~/linear_1']['b'],
        ]],
        actis,
        w_ly_node=[params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_layer_norm']['scale']],
        b_ly_node=[params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_layer_norm']['offset']],
        w_ly_edge=[params['grid2mesh_gnn/~_networks_builder/processor_edges_0_grid2mesh_layer_norm']['scale']],
        b_ly_edge=[params['grid2mesh_gnn/~_networks_builder/processor_edges_0_grid2mesh_layer_norm']['offset']],
        with_sender=sender_pkg,
    )

    return grid_nodes_processed, mesh_nodes_processed

def gc_grid2mesh_tlm(
        params, grid_nodes_tl, grid_nodes,
        mesh_nodes_tl, mesh_nodes, edges_tl, edges, edge_indices
):
    #grid2mesh:
    #  INPUT : grid_nodes, mesh_nodes, edges
    #  OUTPUT: latent_grid_nodes, latent_mesh_nodes

    # ----- Encoder -----
    grid_nodes_encoded_tl, grid_nodes_encoded=denseLayers_tlm(
        grid_nodes_tl, grid_nodes,
        [
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_grid_nodes_mlp/~/linear_0']['w'],
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_grid_nodes_mlp/~/linear_1']['w'],
        ],
        [
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_grid_nodes_mlp/~/linear_0']['b'],
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_grid_nodes_mlp/~/linear_1']['b'],
        ], actis_tlm
    )
    grid_nodes_encoded_tl, grid_nodes_encoded=layerNorm_tlm(
        grid_nodes_encoded_tl, grid_nodes_encoded,
        params['grid2mesh_gnn/~_networks_builder/encoder_nodes_grid_nodes_layer_norm']['scale'],
        params['grid2mesh_gnn/~_networks_builder/encoder_nodes_grid_nodes_layer_norm']['offset'],
    )

    mesh_nodes_encoded_tl, mesh_nodes_encoded=denseLayers_tlm(
        mesh_nodes_tl, mesh_nodes,
        [
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_mesh_nodes_mlp/~/linear_0']['w'],
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_mesh_nodes_mlp/~/linear_1']['w'],
        ],
        [
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_mesh_nodes_mlp/~/linear_0']['b'],
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_mesh_nodes_mlp/~/linear_1']['b'],
        ], actis_tlm
    )
    mesh_nodes_encoded_tl, mesh_nodes_encoded=layerNorm_tlm(
        mesh_nodes_encoded_tl, mesh_nodes_encoded,
        params['grid2mesh_gnn/~_networks_builder/encoder_nodes_mesh_nodes_layer_norm']['scale'],
        params['grid2mesh_gnn/~_networks_builder/encoder_nodes_mesh_nodes_layer_norm']['offset'],
    )

    edges_encoded_tl, edges_encoded=denseLayers_tlm(
        edges_tl, edges,
        [
            params['grid2mesh_gnn/~_networks_builder/encoder_edges_grid2mesh_mlp/~/linear_0']['w'],
            params['grid2mesh_gnn/~_networks_builder/encoder_edges_grid2mesh_mlp/~/linear_1']['w'],
        ],
        [
            params['grid2mesh_gnn/~_networks_builder/encoder_edges_grid2mesh_mlp/~/linear_0']['b'],
            params['grid2mesh_gnn/~_networks_builder/encoder_edges_grid2mesh_mlp/~/linear_1']['b'],
        ],
        actis_tlm
    )
    edges_encoded_tl, edges_encoded=layerNorm_tlm(
        edges_encoded_tl, edges_encoded,
        params['grid2mesh_gnn/~_networks_builder/encoder_edges_grid2mesh_layer_norm']['scale'],
        params['grid2mesh_gnn/~_networks_builder/encoder_edges_grid2mesh_layer_norm']['offset'],
    )

    # ----- Processor -----
    sender_pkg=[
        grid_nodes_encoded_tl, grid_nodes_encoded,
        [[
            params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_0']['w'],
            params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_1']['w'],
        ]],
        [[
            params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_0']['b'],
            params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_1']['b'],
        ]],
        [params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_grid_nodes_layer_norm']['scale']],
        [params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_grid_nodes_layer_norm']['offset']],
    ]

    mesh_nodes_processed_tl, mesh_nodes_processed, \
        edge_processed_tl, edge_processed, \
        grid_nodes_processed_tl, grid_nodes_processed=graph_mp_tlm(
            mesh_nodes_encoded_tl, mesh_nodes_encoded,
            edge_indices, edges_encoded_tl, edges_encoded,
            [[
                params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_0']['w'],
                params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_1']['w'],
            ]],
            [[
                params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_0']['b'],
                params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_1']['b'],
            ]],
            [[
                params['grid2mesh_gnn/~_networks_builder/processor_edges_0_grid2mesh_mlp/~/linear_0']['w'],
                params['grid2mesh_gnn/~_networks_builder/processor_edges_0_grid2mesh_mlp/~/linear_1']['w'],
            ]],
            [[
                params['grid2mesh_gnn/~_networks_builder/processor_edges_0_grid2mesh_mlp/~/linear_0']['b'],
                params['grid2mesh_gnn/~_networks_builder/processor_edges_0_grid2mesh_mlp/~/linear_1']['b'],
            ]],
            actis_tlm,
            w_ly_node=[params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_layer_norm']['scale']],
            b_ly_node=[params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_layer_norm']['offset']],
            w_ly_edge=[params['grid2mesh_gnn/~_networks_builder/processor_edges_0_grid2mesh_layer_norm']['scale']],
            b_ly_edge=[params['grid2mesh_gnn/~_networks_builder/processor_edges_0_grid2mesh_layer_norm']['offset']],
            with_sender=sender_pkg,
        )

    return grid_nodes_processed_tl, grid_nodes_processed, \
        mesh_nodes_processed_tl, mesh_nodes_processed

def gc_grid2mesh_adj(
        params, grid_nodes, grid_nodes_ad, mesh_nodes, mesh_nodes_ad,
        edges, edge_indices,
):
    # ----- forward calculation -----
    grid_nodes_encoded=denseLayers(
        grid_nodes,
        [
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_grid_nodes_mlp/~/linear_0']['w'],
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_grid_nodes_mlp/~/linear_1']['w'],
        ],
        [
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_grid_nodes_mlp/~/linear_0']['b'],
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_grid_nodes_mlp/~/linear_1']['b'],
        ], actis
    )
    grid_nodes_encoded_woLN=grid_nodes_encoded.copy()
    grid_nodes_encoded=layerNorm(
        grid_nodes_encoded,
        params['grid2mesh_gnn/~_networks_builder/encoder_nodes_grid_nodes_layer_norm']['scale'],
        params['grid2mesh_gnn/~_networks_builder/encoder_nodes_grid_nodes_layer_norm']['offset'],
    )

    mesh_nodes_encoded=denseLayers(
        mesh_nodes,
        [
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_mesh_nodes_mlp/~/linear_0']['w'],
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_mesh_nodes_mlp/~/linear_1']['w'],
        ],
        [
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_mesh_nodes_mlp/~/linear_0']['b'],
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_mesh_nodes_mlp/~/linear_1']['b'],
        ], actis
    )
    mesh_nodes_encoded_woLN=mesh_nodes_encoded.copy()
    mesh_nodes_encoded=layerNorm(
        mesh_nodes_encoded,
        params['grid2mesh_gnn/~_networks_builder/encoder_nodes_mesh_nodes_layer_norm']['scale'],
        params['grid2mesh_gnn/~_networks_builder/encoder_nodes_mesh_nodes_layer_norm']['offset'],
    )

    edges_encoded=denseLayers(
        edges,
        [
            params['grid2mesh_gnn/~_networks_builder/encoder_edges_grid2mesh_mlp/~/linear_0']['w'],
            params['grid2mesh_gnn/~_networks_builder/encoder_edges_grid2mesh_mlp/~/linear_1']['w'],
        ],
        [
            params['grid2mesh_gnn/~_networks_builder/encoder_edges_grid2mesh_mlp/~/linear_0']['b'],
            params['grid2mesh_gnn/~_networks_builder/encoder_edges_grid2mesh_mlp/~/linear_1']['b'],
        ],
        actis
    )
    edges_encoded_woLN=edges_encoded.copy()
    edges_encoded=layerNorm(
        edges_encoded,
        params['grid2mesh_gnn/~_networks_builder/encoder_edges_grid2mesh_layer_norm']['scale'],
        params['grid2mesh_gnn/~_networks_builder/encoder_edges_grid2mesh_layer_norm']['offset'],
    )
    
    # ----- end forward calculation -----

    grid_nodes_processed_ad=grid_nodes_ad.copy()
    mesh_nodes_processed_ad=mesh_nodes_ad.copy()
    sender_pkg=[
        grid_nodes_processed_ad, grid_nodes_encoded,
        [[
            params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_0']['w'],
            params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_1']['w'],
        ]],
        [[
            params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_0']['b'],
            params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_1']['b'],
        ]],
        [params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_grid_nodes_layer_norm']['scale']],
        [params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_grid_nodes_layer_norm']['offset']],
    ]

    edges_ad=np.zeros(edges_encoded.shape)    
    grid_nodes_encoded_ad, mesh_nodes_encoded_ad, edges_encoded_ad=graph_mp_adj(
        mesh_nodes_encoded, mesh_nodes_processed_ad, edge_indices, edges_encoded, edges_ad,
        [[
            params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_0']['w'],
            params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_1']['w'],
        ]],
        [[
            params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_0']['b'],
            params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_1']['b'],
        ]],
        [[
            params['grid2mesh_gnn/~_networks_builder/processor_edges_0_grid2mesh_mlp/~/linear_0']['w'],
            params['grid2mesh_gnn/~_networks_builder/processor_edges_0_grid2mesh_mlp/~/linear_1']['w'],
        ]],
        [[
            params['grid2mesh_gnn/~_networks_builder/processor_edges_0_grid2mesh_mlp/~/linear_0']['b'],
            params['grid2mesh_gnn/~_networks_builder/processor_edges_0_grid2mesh_mlp/~/linear_1']['b'],
        ]],
        actis, actis_adj,
        w_ly_node=[params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_layer_norm']['scale']],
        b_ly_node=[params['grid2mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_layer_norm']['offset']],
        w_ly_edge=[params['grid2mesh_gnn/~_networks_builder/processor_edges_0_grid2mesh_layer_norm']['scale']],
        b_ly_edge=[params['grid2mesh_gnn/~_networks_builder/processor_edges_0_grid2mesh_layer_norm']['offset']],
        with_sender=sender_pkg,
    )

    edges_encoded_ad=layerNorm_adj(
        edges_encoded_woLN, edges_encoded_ad,
        params['grid2mesh_gnn/~_networks_builder/encoder_edges_grid2mesh_layer_norm']['scale'],
        params['grid2mesh_gnn/~_networks_builder/encoder_edges_grid2mesh_layer_norm']['offset'],
    )
    edges_ad=denseLayers_adj(
        edges, edges_encoded_ad,
                [
            params['grid2mesh_gnn/~_networks_builder/encoder_edges_grid2mesh_mlp/~/linear_0']['w'],
            params['grid2mesh_gnn/~_networks_builder/encoder_edges_grid2mesh_mlp/~/linear_1']['w'],
        ],
        [
            params['grid2mesh_gnn/~_networks_builder/encoder_edges_grid2mesh_mlp/~/linear_0']['b'],
            params['grid2mesh_gnn/~_networks_builder/encoder_edges_grid2mesh_mlp/~/linear_1']['b'],
        ],
        actis, actis_adj
    )

    mesh_nodes_encoded_ad=layerNorm_adj(
        mesh_nodes_encoded_woLN, mesh_nodes_encoded_ad,
        params['grid2mesh_gnn/~_networks_builder/encoder_nodes_mesh_nodes_layer_norm']['scale'],
        params['grid2mesh_gnn/~_networks_builder/encoder_nodes_mesh_nodes_layer_norm']['offset'],
    )
    mesh_nodes_ad=denseLayers_adj(
        mesh_nodes, mesh_nodes_encoded_ad,
        [
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_mesh_nodes_mlp/~/linear_0']['w'],
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_mesh_nodes_mlp/~/linear_1']['w'],
        ],
        [
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_mesh_nodes_mlp/~/linear_0']['b'],
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_mesh_nodes_mlp/~/linear_1']['b'],
        ],
        actis, actis_adj
    )

    grid_nodes_encoded_ad=layerNorm_adj(
        grid_nodes_encoded_woLN, grid_nodes_encoded_ad,
        params['grid2mesh_gnn/~_networks_builder/encoder_nodes_grid_nodes_layer_norm']['scale'],
        params['grid2mesh_gnn/~_networks_builder/encoder_nodes_grid_nodes_layer_norm']['offset'],
    )
    grid_nodes_ad=denseLayers_adj(
        grid_nodes, grid_nodes_encoded_ad,
        [
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_grid_nodes_mlp/~/linear_0']['w'],
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_grid_nodes_mlp/~/linear_1']['w'],
        ],
        [
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_grid_nodes_mlp/~/linear_0']['b'],
            params['grid2mesh_gnn/~_networks_builder/encoder_nodes_grid_nodes_mlp/~/linear_1']['b'],
        ], actis, actis_adj
    )
    
    return grid_nodes_ad, mesh_nodes_ad, edges_ad
    
def gc_meshGNN(params, mesh_nodes, edges, edge_indices, nMP=16):
    # mesh_gnn
    #   INPUT : mesh_nodes, edges, edge_indices
    #   OUTPUT: processed_mesh_nodes

    # ----- Encoder -----
    edges_encoded=denseLayers(
        edges,
        [
            params['mesh_gnn/~_networks_builder/encoder_edges_mesh_mlp/~/linear_0']['w'],
            params['mesh_gnn/~_networks_builder/encoder_edges_mesh_mlp/~/linear_1']['w'],
        ],
        [
            params['mesh_gnn/~_networks_builder/encoder_edges_mesh_mlp/~/linear_0']['b'],
            params['mesh_gnn/~_networks_builder/encoder_edges_mesh_mlp/~/linear_1']['b'],
        ],
        actis
    )
    edges_encoded=layerNorm(
        edges_encoded,
        params['mesh_gnn/~_networks_builder/encoder_edges_mesh_layer_norm']['scale'],
        params['mesh_gnn/~_networks_builder/encoder_edges_mesh_layer_norm']['offset'],
    )

    # ----- Processor -----
    ws_node, bs_node, ws_edge, bs_edge=[], [], [], []
    w_ly_node, b_ly_node, w_ly_edge, b_ly_edge=[], [], [], []
    for iMP in range(nMP):
        ws_node.append([
            params['mesh_gnn/~_networks_builder/processor_nodes_{:d}_mesh_nodes_mlp/~/linear_0'.format(iMP)]['w'],
            params['mesh_gnn/~_networks_builder/processor_nodes_{:d}_mesh_nodes_mlp/~/linear_1'.format(iMP)]['w'],
        ])
        bs_node.append([
            params['mesh_gnn/~_networks_builder/processor_nodes_{:d}_mesh_nodes_mlp/~/linear_0'.format(iMP)]['b'],
            params['mesh_gnn/~_networks_builder/processor_nodes_{:d}_mesh_nodes_mlp/~/linear_1'.format(iMP)]['b'],
        ])
        ws_edge.append([
            params['mesh_gnn/~_networks_builder/processor_edges_{:d}_mesh_mlp/~/linear_0'.format(iMP)]['w'],
            params['mesh_gnn/~_networks_builder/processor_edges_{:d}_mesh_mlp/~/linear_1'.format(iMP)]['w'],
        ])
        bs_edge.append([
            params['mesh_gnn/~_networks_builder/processor_edges_{:d}_mesh_mlp/~/linear_0'.format(iMP)]['b'],
            params['mesh_gnn/~_networks_builder/processor_edges_{:d}_mesh_mlp/~/linear_1'.format(iMP)]['b'],
        ])
        w_ly_node.append(params['mesh_gnn/~_networks_builder/processor_nodes_{:d}_mesh_nodes_layer_norm'.format(iMP)]['scale'])
        b_ly_node.append(params['mesh_gnn/~_networks_builder/processor_nodes_{:d}_mesh_nodes_layer_norm'.format(iMP)]['offset'])
        w_ly_edge.append(params['mesh_gnn/~_networks_builder/processor_edges_{:d}_mesh_layer_norm'.format(iMP)]['scale'])
        b_ly_edge.append(params['mesh_gnn/~_networks_builder/processor_edges_{:d}_mesh_layer_norm'.format(iMP)]['offset'])

    
    mesh_nodes_processed, edges_processed = graph_mp(
        mesh_nodes, edge_indices, edges_encoded, ws_node, bs_node, ws_edge, bs_edge, actis,
        w_ly_node=w_ly_node, b_ly_node=b_ly_node, w_ly_edge=w_ly_edge, b_ly_edge=b_ly_edge,
        with_sender=None
    )

    return mesh_nodes_processed

def gc_meshGNN_tlm(
        params, mesh_nodes_tl, mesh_nodes, edges_tl, edges, edge_indices, nMP=16
):
    # mesh_gnn
    #   INPUT : mesh_nodes, edges, edge_indices
    #   OUTPUT: processed_mesh_nodes

    # ----- Encoder -----
    edges_encoded_tl, edges_encoded=denseLayers_tlm(
        edges_tl, edges,
        [
            params['mesh_gnn/~_networks_builder/encoder_edges_mesh_mlp/~/linear_0']['w'],
            params['mesh_gnn/~_networks_builder/encoder_edges_mesh_mlp/~/linear_1']['w'],
        ],
        [
            params['mesh_gnn/~_networks_builder/encoder_edges_mesh_mlp/~/linear_0']['b'],
            params['mesh_gnn/~_networks_builder/encoder_edges_mesh_mlp/~/linear_1']['b'],
        ],
        actis_tlm
    )
    edges_encoded_tl, edges_encoded=layerNorm_tlm(
        edges_encoded_tl, edges_encoded,
        params['mesh_gnn/~_networks_builder/encoder_edges_mesh_layer_norm']['scale'],
        params['mesh_gnn/~_networks_builder/encoder_edges_mesh_layer_norm']['offset'],
    )

    # ----- Processor -----
    ws_node, bs_node, ws_edge, bs_edge=[], [], [], []
    w_ly_node, b_ly_node, w_ly_edge, b_ly_edge=[], [], [], []
    for iMP in range(nMP):
        ws_node.append([
            params['mesh_gnn/~_networks_builder/processor_nodes_{:d}_mesh_nodes_mlp/~/linear_0'.format(iMP)]['w'],
            params['mesh_gnn/~_networks_builder/processor_nodes_{:d}_mesh_nodes_mlp/~/linear_1'.format(iMP)]['w'],
        ])
        bs_node.append([
            params['mesh_gnn/~_networks_builder/processor_nodes_{:d}_mesh_nodes_mlp/~/linear_0'.format(iMP)]['b'],
            params['mesh_gnn/~_networks_builder/processor_nodes_{:d}_mesh_nodes_mlp/~/linear_1'.format(iMP)]['b'],
        ])
        ws_edge.append([
            params['mesh_gnn/~_networks_builder/processor_edges_{:d}_mesh_mlp/~/linear_0'.format(iMP)]['w'],
            params['mesh_gnn/~_networks_builder/processor_edges_{:d}_mesh_mlp/~/linear_1'.format(iMP)]['w'],
        ])
        bs_edge.append([
            params['mesh_gnn/~_networks_builder/processor_edges_{:d}_mesh_mlp/~/linear_0'.format(iMP)]['b'],
            params['mesh_gnn/~_networks_builder/processor_edges_{:d}_mesh_mlp/~/linear_1'.format(iMP)]['b'],
        ])
        w_ly_node.append(params['mesh_gnn/~_networks_builder/processor_nodes_{:d}_mesh_nodes_layer_norm'.format(iMP)]['scale'])
        b_ly_node.append(params['mesh_gnn/~_networks_builder/processor_nodes_{:d}_mesh_nodes_layer_norm'.format(iMP)]['offset'])
        w_ly_edge.append(params['mesh_gnn/~_networks_builder/processor_edges_{:d}_mesh_layer_norm'.format(iMP)]['scale'])
        b_ly_edge.append(params['mesh_gnn/~_networks_builder/processor_edges_{:d}_mesh_layer_norm'.format(iMP)]['offset'])

    mesh_nodes_processed_tl, mesh_nodes_processed, \
        edges_processed_tl, edges_processed = graph_mp_tlm(
            mesh_nodes_tl, mesh_nodes, edge_indices,
            edges_encoded_tl, edges_encoded,
            ws_node, bs_node, ws_edge, bs_edge, actis_tlm,
            w_ly_node=w_ly_node, b_ly_node=b_ly_node,
            w_ly_edge=w_ly_edge, b_ly_edge=b_ly_edge,
            with_sender=None
        )

    return mesh_nodes_processed_tl, mesh_nodes_processed

def gc_meshGNN_adj(params, mesh_nodes, mesh_nodes_ad, edges, edge_indices, nMP=16):

    # ----- forward calculation -----
    edges_encoded=denseLayers(
        edges,
        [
            params['mesh_gnn/~_networks_builder/encoder_edges_mesh_mlp/~/linear_0']['w'],
            params['mesh_gnn/~_networks_builder/encoder_edges_mesh_mlp/~/linear_1']['w'],
        ],
        [
            params['mesh_gnn/~_networks_builder/encoder_edges_mesh_mlp/~/linear_0']['b'],
            params['mesh_gnn/~_networks_builder/encoder_edges_mesh_mlp/~/linear_1']['b'],
        ],
        actis
    )
    edges_encoded=layerNorm(
        edges_encoded,
        params['mesh_gnn/~_networks_builder/encoder_edges_mesh_layer_norm']['scale'],
        params['mesh_gnn/~_networks_builder/encoder_edges_mesh_layer_norm']['offset'],
    )
    # ----- end forward calculation -----
    
    ws_node, bs_node, ws_edge, bs_edge=[], [], [], []
    w_ly_node, b_ly_node, w_ly_edge, b_ly_edge=[], [], [], []
    for iMP in range(nMP):
        ws_node.append([
            params['mesh_gnn/~_networks_builder/processor_nodes_{:d}_mesh_nodes_mlp/~/linear_0'.format(iMP)]['w'],
            params['mesh_gnn/~_networks_builder/processor_nodes_{:d}_mesh_nodes_mlp/~/linear_1'.format(iMP)]['w'],
        ])
        bs_node.append([
            params['mesh_gnn/~_networks_builder/processor_nodes_{:d}_mesh_nodes_mlp/~/linear_0'.format(iMP)]['b'],
            params['mesh_gnn/~_networks_builder/processor_nodes_{:d}_mesh_nodes_mlp/~/linear_1'.format(iMP)]['b'],
        ])
        ws_edge.append([
            params['mesh_gnn/~_networks_builder/processor_edges_{:d}_mesh_mlp/~/linear_0'.format(iMP)]['w'],
            params['mesh_gnn/~_networks_builder/processor_edges_{:d}_mesh_mlp/~/linear_1'.format(iMP)]['w'],
        ])
        bs_edge.append([
            params['mesh_gnn/~_networks_builder/processor_edges_{:d}_mesh_mlp/~/linear_0'.format(iMP)]['b'],
            params['mesh_gnn/~_networks_builder/processor_edges_{:d}_mesh_mlp/~/linear_1'.format(iMP)]['b'],
        ])
        w_ly_node.append(params['mesh_gnn/~_networks_builder/processor_nodes_{:d}_mesh_nodes_layer_norm'.format(iMP)]['scale'])
        b_ly_node.append(params['mesh_gnn/~_networks_builder/processor_nodes_{:d}_mesh_nodes_layer_norm'.format(iMP)]['offset'])
        w_ly_edge.append(params['mesh_gnn/~_networks_builder/processor_edges_{:d}_mesh_layer_norm'.format(iMP)]['scale'])
        b_ly_edge.append(params['mesh_gnn/~_networks_builder/processor_edges_{:d}_mesh_layer_norm'.format(iMP)]['offset'])

    edges_encoded_ad=np.zeros(edges_encoded.shape)
    mesh_nodes_ad, edges_ad = graph_mp_adj(
        mesh_nodes, mesh_nodes_ad, edge_indices, edges_encoded, edges_encoded_ad,
        ws_node, bs_node, ws_edge, bs_edge, actis, actis_adj,
        w_ly_node=w_ly_node, b_ly_node=b_ly_node, w_ly_edge=w_ly_edge, b_ly_edge=b_ly_edge,
        with_sender=None
    )

    return mesh_nodes_ad
    

def gc_mesh2grid(params, grid_nodes, mesh_nodes, edges, edge_indices):
    #mesh2grid:
    #  INPUT : grid_nodes, mesh_nodes, edges, edge_indices
    #  output: output_grid_nodes
    
    # ----- Encoder -----
    edges_encoded=denseLayers(
        edges,
        [
            params['mesh2grid_gnn/~_networks_builder/encoder_edges_mesh2grid_mlp/~/linear_0']['w'],
            params['mesh2grid_gnn/~_networks_builder/encoder_edges_mesh2grid_mlp/~/linear_1']['w'],
        ],
        [
            params['mesh2grid_gnn/~_networks_builder/encoder_edges_mesh2grid_mlp/~/linear_0']['b'],
            params['mesh2grid_gnn/~_networks_builder/encoder_edges_mesh2grid_mlp/~/linear_1']['b'],
        ],
        actis
    )
    edges_encoded=layerNorm(
        edges_encoded,
        params['mesh2grid_gnn/~_networks_builder/encoder_edges_mesh2grid_layer_norm']['scale'],
        params['mesh2grid_gnn/~_networks_builder/encoder_edges_mesh2grid_layer_norm']['offset']
    )

    # ----- Processor -----
    sender_pkg=[
        mesh_nodes,
        [[
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_0']['w'],
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_1']['w'],
        ]],
        [[
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_0']['b'],
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_1']['b'],
        ]],
        [params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_layer_norm']['scale']],
        [params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_layer_norm']['offset']],
    ]

    grid_nodes_processed, edges_processed, mesh_nodes_processed=graph_mp(
        grid_nodes, edge_indices, edges_encoded,
        [[
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_0']['w'],
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_1']['w'],
        ]],
        [[
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_0']['b'],
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_1']['b'],
        ]],
        [[
            params['mesh2grid_gnn/~_networks_builder/processor_edges_0_mesh2grid_mlp/~/linear_0']['w'],
            params['mesh2grid_gnn/~_networks_builder/processor_edges_0_mesh2grid_mlp/~/linear_1']['w'],
        ]],
        [[
            params['mesh2grid_gnn/~_networks_builder/processor_edges_0_mesh2grid_mlp/~/linear_0']['b'],
            params['mesh2grid_gnn/~_networks_builder/processor_edges_0_mesh2grid_mlp/~/linear_1']['b'],
        ]],
        actis,
        w_ly_node=[params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_grid_nodes_layer_norm']['scale']],
        b_ly_node=[params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_grid_nodes_layer_norm']['offset']],
        w_ly_edge=[params['mesh2grid_gnn/~_networks_builder/processor_edges_0_mesh2grid_layer_norm']['scale']],
        b_ly_edge=[params['mesh2grid_gnn/~_networks_builder/processor_edges_0_mesh2grid_layer_norm']['offset']],
        with_sender=sender_pkg,
    )
    
    # ----- Decoder -----
    grid_nodes_output=denseLayers(
        grid_nodes_processed,
        [
            params['mesh2grid_gnn/~_networks_builder/decoder_nodes_grid_nodes_mlp/~/linear_0']['w'],
            params['mesh2grid_gnn/~_networks_builder/decoder_nodes_grid_nodes_mlp/~/linear_1']['w'],
        ],
        [
            params['mesh2grid_gnn/~_networks_builder/decoder_nodes_grid_nodes_mlp/~/linear_0']['b'],
            params['mesh2grid_gnn/~_networks_builder/decoder_nodes_grid_nodes_mlp/~/linear_1']['b'],
        ],
        actis
    )
    
    return grid_nodes_output

def gc_mesh2grid_tlm(
        params, grid_nodes_tl, grid_nodes, mesh_nodes_tl, mesh_nodes,
        edges_tl, edges, edge_indices
):
    #mesh2grid:
    #  INPUT : grid_nodes, mesh_nodes, edges, edge_indices
    #  output: output_grid_nodes
    
    # ----- Encoder -----
    edges_encoded_tl, edges_encoded=denseLayers_tlm(
        edges_tl, edges,
        [
            params['mesh2grid_gnn/~_networks_builder/encoder_edges_mesh2grid_mlp/~/linear_0']['w'],
            params['mesh2grid_gnn/~_networks_builder/encoder_edges_mesh2grid_mlp/~/linear_1']['w'],
        ],
        [
            params['mesh2grid_gnn/~_networks_builder/encoder_edges_mesh2grid_mlp/~/linear_0']['b'],
            params['mesh2grid_gnn/~_networks_builder/encoder_edges_mesh2grid_mlp/~/linear_1']['b'],
        ],
        actis_tlm
    )
    edges_encoded_tl, edges_encoded=layerNorm_tlm(
        edges_encoded_tl, edges_encoded,
        params['mesh2grid_gnn/~_networks_builder/encoder_edges_mesh2grid_layer_norm']['scale'],
        params['mesh2grid_gnn/~_networks_builder/encoder_edges_mesh2grid_layer_norm']['offset']
    )

    # ----- Processor -----
    sender_pkg=[
        mesh_nodes_tl, mesh_nodes,
        [[
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_0']['w'],
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_1']['w'],
        ]],
        [[
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_0']['b'],
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_1']['b'],
        ]],
        [params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_layer_norm']['scale']],
        [params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_layer_norm']['offset']],
    ]

    grid_nodes_processed_tl, grid_nodes_processed, \
        edges_processed_tl, edges_processed, \
        mesh_nodes_processed_tl, mesh_nodes_processed=graph_mp_tlm(
            grid_nodes_tl, grid_nodes,
            edge_indices, edges_encoded_tl, edges_encoded,
            [[
                params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_0']['w'],
                params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_1']['w'],
            ]],
            [[
                params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_0']['b'],
                params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_1']['b'],
            ]],
            [[
                params['mesh2grid_gnn/~_networks_builder/processor_edges_0_mesh2grid_mlp/~/linear_0']['w'],
                params['mesh2grid_gnn/~_networks_builder/processor_edges_0_mesh2grid_mlp/~/linear_1']['w'],
            ]],
            [[
                params['mesh2grid_gnn/~_networks_builder/processor_edges_0_mesh2grid_mlp/~/linear_0']['b'],
                params['mesh2grid_gnn/~_networks_builder/processor_edges_0_mesh2grid_mlp/~/linear_1']['b'],
            ]],
            actis_tlm,
            w_ly_node=[params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_grid_nodes_layer_norm']['scale']],
            b_ly_node=[params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_grid_nodes_layer_norm']['offset']],
            w_ly_edge=[params['mesh2grid_gnn/~_networks_builder/processor_edges_0_mesh2grid_layer_norm']['scale']],
            b_ly_edge=[params['mesh2grid_gnn/~_networks_builder/processor_edges_0_mesh2grid_layer_norm']['offset']],
            with_sender=sender_pkg,
        )
    
    # ----- Decoder -----
    grid_nodes_output_tl, grid_nodes_output=denseLayers_tlm(
        grid_nodes_processed_tl, grid_nodes_processed,
        [
            params['mesh2grid_gnn/~_networks_builder/decoder_nodes_grid_nodes_mlp/~/linear_0']['w'],
            params['mesh2grid_gnn/~_networks_builder/decoder_nodes_grid_nodes_mlp/~/linear_1']['w'],
        ],
        [
            params['mesh2grid_gnn/~_networks_builder/decoder_nodes_grid_nodes_mlp/~/linear_0']['b'],
            params['mesh2grid_gnn/~_networks_builder/decoder_nodes_grid_nodes_mlp/~/linear_1']['b'],
        ],
        actis_tlm
    )
    
    return grid_nodes_output_tl, grid_nodes_output

def gc_mesh2grid_adj(
        params, grid_nodes, grid_nodes_output_ad, mesh_nodes,
        edges, edge_indices
):
    
    # ----- forward calculation -----
    # ----- Encoder -----
    edges_encoded=denseLayers(
        edges,
        [
            params['mesh2grid_gnn/~_networks_builder/encoder_edges_mesh2grid_mlp/~/linear_0']['w'],
            params['mesh2grid_gnn/~_networks_builder/encoder_edges_mesh2grid_mlp/~/linear_1']['w'],
        ],
        [
            params['mesh2grid_gnn/~_networks_builder/encoder_edges_mesh2grid_mlp/~/linear_0']['b'],
            params['mesh2grid_gnn/~_networks_builder/encoder_edges_mesh2grid_mlp/~/linear_1']['b'],
        ],
        actis
    )
    edges_encoded=layerNorm(
        edges_encoded,
        params['mesh2grid_gnn/~_networks_builder/encoder_edges_mesh2grid_layer_norm']['scale'],
        params['mesh2grid_gnn/~_networks_builder/encoder_edges_mesh2grid_layer_norm']['offset']
    )

    sender_pkg=[
        mesh_nodes,
        [[
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_0']['w'],
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_1']['w'],
        ]],
        [[
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_0']['b'],
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_1']['b'],
        ]],
        [params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_layer_norm']['scale']],
        [params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_layer_norm']['offset']],
    ]

    grid_nodes_processed, edges_processed, mesh_nodes_processed=graph_mp(
        grid_nodes, edge_indices, edges_encoded,
        [[
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_0']['w'],
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_1']['w'],
        ]],
        [[
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_0']['b'],
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_1']['b'],
        ]],
        [[
            params['mesh2grid_gnn/~_networks_builder/processor_edges_0_mesh2grid_mlp/~/linear_0']['w'],
            params['mesh2grid_gnn/~_networks_builder/processor_edges_0_mesh2grid_mlp/~/linear_1']['w'],
        ]],
        [[
            params['mesh2grid_gnn/~_networks_builder/processor_edges_0_mesh2grid_mlp/~/linear_0']['b'],
            params['mesh2grid_gnn/~_networks_builder/processor_edges_0_mesh2grid_mlp/~/linear_1']['b'],
        ]],
        actis,
        w_ly_node=[params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_grid_nodes_layer_norm']['scale']],
        b_ly_node=[params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_grid_nodes_layer_norm']['offset']],
        w_ly_edge=[params['mesh2grid_gnn/~_networks_builder/processor_edges_0_mesh2grid_layer_norm']['scale']],
        b_ly_edge=[params['mesh2grid_gnn/~_networks_builder/processor_edges_0_mesh2grid_layer_norm']['offset']],
        with_sender=sender_pkg,
    )
    # ----- end forward calculation -----

    grid_nodes_processed_ad=denseLayers_adj(
        grid_nodes_processed, grid_nodes_output_ad,
        [
            params['mesh2grid_gnn/~_networks_builder/decoder_nodes_grid_nodes_mlp/~/linear_0']['w'],
            params['mesh2grid_gnn/~_networks_builder/decoder_nodes_grid_nodes_mlp/~/linear_1']['w'],
        ],
        [
            params['mesh2grid_gnn/~_networks_builder/decoder_nodes_grid_nodes_mlp/~/linear_0']['b'],
            params['mesh2grid_gnn/~_networks_builder/decoder_nodes_grid_nodes_mlp/~/linear_1']['b'],
        ],
        actis, actis_adj,
    )

    mesh_nodes_processed_ad=np.zeros(mesh_nodes_processed.shape)
    edges_processed_ad=np.zeros(edges_processed.shape)
    sender_pkg_ad=[
        mesh_nodes_processed_ad, mesh_nodes,
        [[
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_0']['w'],
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_1']['w'],
        ]],
        [[
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_0']['b'],
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_1']['b'],
        ]],
        [params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_layer_norm']['scale']],
        [params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_layer_norm']['offset']],
    ]
    mesh_nodes_ad, grid_nodes_ad, edges_encoded_ad=graph_mp_adj(
        grid_nodes, grid_nodes_processed_ad, edge_indices, edges_encoded, edges_processed_ad,
        [[
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_0']['w'],
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_1']['w'],
        ]],
        [[
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_0']['b'],
            params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_grid_nodes_mlp/~/linear_1']['b'],
        ]],
        [[
            params['mesh2grid_gnn/~_networks_builder/processor_edges_0_mesh2grid_mlp/~/linear_0']['w'],
            params['mesh2grid_gnn/~_networks_builder/processor_edges_0_mesh2grid_mlp/~/linear_1']['w'],
        ]],
        [[
            params['mesh2grid_gnn/~_networks_builder/processor_edges_0_mesh2grid_mlp/~/linear_0']['b'],
            params['mesh2grid_gnn/~_networks_builder/processor_edges_0_mesh2grid_mlp/~/linear_1']['b'],
        ]],
        actis, actis_adj,
        w_ly_node=[params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_grid_nodes_layer_norm']['scale']],
        b_ly_node=[params['mesh2grid_gnn/~_networks_builder/processor_nodes_0_grid_nodes_layer_norm']['offset']],
        w_ly_edge=[params['mesh2grid_gnn/~_networks_builder/processor_edges_0_mesh2grid_layer_norm']['scale']],
        b_ly_edge=[params['mesh2grid_gnn/~_networks_builder/processor_edges_0_mesh2grid_layer_norm']['offset']],
        with_sender=sender_pkg_ad,
    )

    return grid_nodes_ad, mesh_nodes_ad
    

def gc_grid2grid(params, grid_nodes, mesh_nodes, edges, edge_indices):
    # INPUT:
    #   edges={
    #          'grid2mesh': np.array,
    #          'meshGNN':   np.array,
    #          'mesh2grid': np.array,
    #         }
    #   edges_indices={
    #          'grid2mesh': np.array,
    #          'meshGNN':   np.array,
    #          'mesh2grid': np.array,
    #         }

    latent_grid_nodes, latent_mesh_nodes=gc_grid2mesh(
        params, grid_nodes, mesh_nodes,
        edges['grid2mesh'], edge_indices['grid2mesh']
    )
    
    updated_latent_mesh_nodes=gc_meshGNN(
        params, latent_mesh_nodes, edges['meshGNN'], edge_indices['meshGNN']
    )

    output_grid_nodes=gc_mesh2grid(
        params, latent_grid_nodes, updated_latent_mesh_nodes,
        edges['mesh2grid'], edge_indices['mesh2grid']
    )

    return output_grid_nodes

def gc_grid2grid_tlm(
        params, grid_nodes_tl, grid_nodes, mesh_nodes_tl, mesh_nodes,
        edges_tl, edges, edge_indices
):
    # INPUT:
    #   edges={
    #          'grid2mesh': np.array,
    #          'meshGNN':   np.array,
    #          'mesh2grid': np.array,
    #         }
    #   edges_indices={
    #          'grid2mesh': np.array,
    #          'meshGNN':   np.array,
    #          'mesh2grid': np.array,
    #         }

    latent_grid_nodes_tl, latent_grid_nodes, \
        latent_mesh_nodes_tl, latent_mesh_nodes=gc_grid2mesh_tlm(
            params, grid_nodes_tl, grid_nodes, mesh_nodes_tl, mesh_nodes,
            edges_tl['grid2mesh'], edges['grid2mesh'], edge_indices['grid2mesh']
        )

    updated_latent_mesh_nodes_tl, updated_latent_mesh_nodes=gc_meshGNN_tlm(
        params, latent_mesh_nodes_tl, latent_mesh_nodes,
        edges_tl['meshGNN'], edges['meshGNN'], edge_indices['meshGNN']
    )

    output_grid_nodes_tl, output_grid_nodes=gc_mesh2grid_tlm(
        params, latent_grid_nodes_tl, latent_grid_nodes,
        updated_latent_mesh_nodes_tl, updated_latent_mesh_nodes,
        edges_tl['mesh2grid'], edges['mesh2grid'], edge_indices['mesh2grid']
    )

    return output_grid_nodes_tl, output_grid_nodes

def gc_grid2grid_adj(
        params, grid_nodes, output_grid_nodes_ad, mesh_nodes,
        edges, edge_indices,
):

    # ----- forward calculation -----
    latent_grid_nodes, latent_mesh_nodes=gc_grid2mesh(
        params, grid_nodes, mesh_nodes,
        edges['grid2mesh'], edge_indices['grid2mesh']
    )
    
    updated_latent_mesh_nodes=gc_meshGNN(
        params, latent_mesh_nodes, edges['meshGNN'], edge_indices['meshGNN']
    )
    # ----- end forward calculation -----
    
    latent_grid_nodes_ad, updated_latent_mesh_nodes_ad=gc_mesh2grid_adj(
        params, latent_grid_nodes, output_grid_nodes_ad, updated_latent_mesh_nodes,
        edges['mesh2grid'], edge_indices['mesh2grid']
    )

    latent_mesh_nodes_ad=gc_meshGNN_adj(
        params, latent_mesh_nodes, updated_latent_mesh_nodes_ad,
        edges['meshGNN'], edge_indices['meshGNN']
    )

    grid_nodes_ad, mesh_nodes_ad, _=gc_grid2mesh_adj(
        params, grid_nodes, latent_grid_nodes_ad, mesh_nodes, latent_mesh_nodes_ad,
        edges['grid2mesh'], edge_indices['grid2mesh']
    )

    return grid_nodes_ad, mesh_nodes_ad
    

def loadInputs(
        task_config,
        fName='notebook/source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc'
):

    with open(fName, 'rb') as f:
        example_batch = xarray.load_dataset(f).compute()

    eval_steps=1
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        example_batch, target_lead_times=slice("6h", f"{eval_steps*6}h"),
        **dataclasses.asdict(task_config))

    return inputs, targets, forcings

def unnormalize_prediction_and_add_input(
        inputs, norm_prediction,
        scales=None, locations=None, residual_scales=None, residual_locations=None
):
    if norm_prediction.sizes.get('time') != 1:
        raise ValueError(
            'normalization.InputsAndResiduals only supports predicting a '
            'single timestep.')
    if norm_prediction.name in inputs:
        # Residuals are assumed to be predicted as normalized (unit variance),
        # but the scale and location they need mapping to is that of the residuals
        # not of the values themselves.
        prediction = unnormalize(
            norm_prediction, residual_scales, residual_locations)
        # A prediction for which we have a corresponding input -- we are
        # predicting the residual:
        last_input = inputs[norm_prediction.name].isel(time=-1)
        prediction += last_input
        return prediction
    else:
        # A predicted variable which is not an input variable. We are predicting
        # it directly, so unnormalize it directly to the target scale/location:
        return unnormalize(norm_prediction, scales, locations)

def unnormalize_prediction_and_add_input_adj(
        inputs_ad, inputs, prediction_ad, 
        scales=None, locations=None, residual_scales=None, residual_locations=None
):
    # will only consider the "if norm_prediction.name in inputs:" part
    last_input_ad=prediction_ad.copy()
    last_input_pointer=inputs_ad[prediction_ad.name].isel(time=-1)
    last_input_pointer+=np.squeeze(last_input_ad)

    norm_prediction_ad=unnormalize(
        prediction_ad, residual_scales, residual_locations)
    return norm_prediction_ad, inputs_ad

def inputs_to_grid_node_features(
        inputs: xarray.Dataset,
        forcings: xarray.Dataset,
) -> chex.Array:
    """xarrays -> [num_grid_nodes, batch, num_channels]."""

    # xarray `Dataset` (batch, time, lat, lon, level, multiple vars)
    # to xarray `DataArray` (batch, lat, lon, channels)
    stacked_inputs = model_utils.dataset_to_stacked(inputs)
    stacked_forcings = model_utils.dataset_to_stacked(forcings)
    stacked_inputs = xarray.concat(
        [stacked_inputs, stacked_forcings], dim="channels")

    # xarray `DataArray` (batch, lat, lon, channels)
    # to single numpy array with shape [lat_lon_node, batch, channels]
    grid_xarray_lat_lon_leading = model_utils.lat_lon_to_leading_axes(
        stacked_inputs)

    return xarray_jax.unwrap(grid_xarray_lat_lon_leading.data).reshape(
        (-1,) + grid_xarray_lat_lon_leading.data.shape[2:])

def inputs_to_grid_node_features_adj(
        grid_node_features_ad, inputs_ad, forcings,
):
    # returning: norm_inputs_ad

    stacked_forcings = model_utils.dataset_to_stacked(forcings).values
    nLat, nLon=stacked_forcings.shape[1:3]
    nChan_forcings=stacked_forcings.shape[-1]
    nLev=inputs_ad.level.shape[0]

    grid_lat_lon_leading_ad=grid_node_features_ad.reshape(
        nLat, nLon, grid_node_features_ad.shape[-1])
    # removing forcings
    grid_lat_lon_leading_ad=grid_lat_lon_leading_ad[:,:,:-nChan_forcings]

    norm_inputs_ad=inputs_ad*0.
    idx=0
    for name in sorted(norm_inputs_ad.data_vars.keys()):
        if len(norm_inputs_ad[name].shape)==4:
            norm_inputs_ad[name][0,0,:,:]+=grid_lat_lon_leading_ad[:,:,idx]
            norm_inputs_ad[name][0,1,:,:]+=grid_lat_lon_leading_ad[:,:,idx+1]

            idx+=2
        elif len(norm_inputs_ad[name].shape)==5:
            norm_inputs_ad[name][0,0,:,:,:]+=np.swapaxes(grid_lat_lon_leading_ad[:,:,idx:idx+nLev].T,1,2)
            norm_inputs_ad[name][0,1,:,:,:]+=np.swapaxes(grid_lat_lon_leading_ad[:,:,idx+nLev:idx+nLev*2].T,1,2)
            idx+=nLev*2
        else:
            if 'time' in norm_inputs_ad[name].dims:
                idx+=2
            else:
                idx+=1

    return norm_inputs_ad

def grid_node_outputs_to_prediction(
        grid_node_outputs: chex.Array,
        targets_template: xarray.Dataset,
) -> xarray.Dataset:
    """[num_grid_nodes, batch, num_outputs] -> xarray."""
    
    # numpy array with shape [lat_lon_node, batch, channels]
    # to xarray `DataArray` (batch, lat, lon, channels)
    assert targets_template.lat is not None and targets_template.lon is not None
    grid_shape = (targets_template.lat.shape[0], targets_template.lon.shape[0])
    grid_outputs_lat_lon_leading = grid_node_outputs.reshape(
        grid_shape + grid_node_outputs.shape[1:])

    dims = ("lat", "lon", "batch", "channels")
    grid_xarray_lat_lon_leading = xarray_jax.DataArray(
        data=grid_outputs_lat_lon_leading,
        dims=dims)
    grid_xarray = model_utils.restore_leading_axes(grid_xarray_lat_lon_leading)

    # xarray `DataArray` (batch, lat, lon, channels)
    # to xarray `Dataset` (batch, one time step, lat, lon, level, multiple vars)
    return model_utils.stacked_to_dataset(
        grid_xarray.variable, targets_template)

def grid_node_outputs_to_prediction_adj(
        predictions_ad, template_dataset,
        preserved_dims: tuple[str, ...] = ("batch", "lat", "lon"),
):

    stacked_pred_ad=model_utils.dataset_to_stacked(predictions_ad).values
    output_grid_nodes_ad=stacked_pred_ad.reshape(-1,stacked_pred_ad.shape[-1])

    return output_grid_nodes_ad
    
class gcForecast(object):

    def __init__(
            self,
            params,
            diffs_stddev_by_level,
            mean_by_level,
            stddev_by_level,
            staticFile='notebook/Static_Res_1.00deg.npz',
    ):
        self.params=params
        holder=np.load(staticFile)
        self.edges={
            'grid2mesh': holder['edges_grid2mesh'],
            'meshGNN'  : holder['edges_mesh_gnn'],
            'mesh2grid': holder['edges_mesh2grid'],
        }
        self.edge_indices={
            'grid2mesh': holder['edge_indices_grid2mesh'],
            'meshGNN'  : holder['edge_indices_mesh_gnn'],
            'mesh2grid': holder['edge_indices_mesh2grid'],
        }
        self.grid_nodes_fixed=holder['grid_nodes_fixed']
        self.mesh_nodes_fixed=holder['mesh_nodes_fixed']

        self.diffs_stddev_by_level=diffs_stddev_by_level
        self.mean_by_level=mean_by_level
        self.stddev_by_level=stddev_by_level

    def forecast(self, inputs, forcings, targets_template):

        norm_inputs=normalize(inputs, self.stddev_by_level, self.mean_by_level)
        norm_forcings=normalize(forcings, self.stddev_by_level, self.mean_by_level)

        grid_node_features=inputs_to_grid_node_features(
            norm_inputs, norm_forcings)
        grid_node_features=np.squeeze(grid_node_features)  # removing dimension [batch]

        grid_nodes=np.concatenate((grid_node_features, self.grid_nodes_fixed), axis=-1)
        mesh_nodes=np.concatenate((
            np.zeros([self.mesh_nodes_fixed.shape[0], grid_node_features.shape[-1]]),
            self.mesh_nodes_fixed), axis=-1)

        output_grid_nodes=gc_grid2grid(
            self.params, grid_nodes, mesh_nodes, self.edges, self.edge_indices
        )

        norm_predictions=grid_node_outputs_to_prediction(
            output_grid_nodes[:,np.newaxis,:], targets_template)

        predictions=xarray_tree.map_structure(
            lambda pred: unnormalize_prediction_and_add_input(
                inputs, pred, scales=self.stddev_by_level,
                locations=self.mean_by_level,
                residual_scales=self.diffs_stddev_by_level),
            norm_predictions)

        return predictions

    def forecast_tlm(self, inputs_tl, inputs, forcings, targets_template):
        
        norm_inputs_tl=normalize(inputs_tl, self.stddev_by_level, self.mean_by_level*0.)
        norm_inputs=normalize(inputs, self.stddev_by_level, self.mean_by_level)
        
        norm_forcings=normalize(forcings, self.stddev_by_level, self.mean_by_level)

        grid_node_features_tl=inputs_to_grid_node_features(
            norm_inputs_tl, norm_forcings*0.)
        grid_node_features=inputs_to_grid_node_features(
            norm_inputs, norm_forcings)

        grid_node_features_tl=np.squeeze(grid_node_features_tl)
        grid_node_features=np.squeeze(grid_node_features)

        grid_nodes_tl=np.concatenate((grid_node_features_tl, self.grid_nodes_fixed*0.), axis=-1)
        grid_nodes=np.concatenate((grid_node_features, self.grid_nodes_fixed), axis=-1)

        mesh_nodes_tl=np.concatenate((
            np.zeros([self.mesh_nodes_fixed.shape[0], grid_node_features.shape[-1]]),
            self.mesh_nodes_fixed*0.), axis=-1)
        mesh_nodes=np.concatenate((
            np.zeros([self.mesh_nodes_fixed.shape[0], grid_node_features.shape[-1]]),
            self.mesh_nodes_fixed), axis=-1)

        edges_tl=deepcopy(self.edges)
        for key in edges_tl: edges_tl[key]*=0.
        output_grid_nodes_tl, output_grid_nodes=gc_grid2grid_tlm(
            self.params,
            grid_nodes_tl, grid_nodes,
            mesh_nodes_tl, mesh_nodes,
            edges_tl, self.edges, self.edge_indices
        )

        norm_predictions_tl=grid_node_outputs_to_prediction(
            output_grid_nodes_tl[:,np.newaxis,:], targets_template)
        norm_predictions=grid_node_outputs_to_prediction(
            output_grid_nodes[:,np.newaxis,:], targets_template)

        predictions_tl=xarray_tree.map_structure(
            lambda pred: unnormalize_prediction_and_add_input(
                inputs_tl, pred, scales=self.stddev_by_level,
                locations=self.mean_by_level*0.,
                residual_scales=self.diffs_stddev_by_level),
            norm_predictions_tl)
        predictions=xarray_tree.map_structure(
            lambda pred: unnormalize_prediction_and_add_input(
                inputs, pred, scales=self.stddev_by_level,
                locations=self.mean_by_level,
                residual_scales=self.diffs_stddev_by_level),
            norm_predictions)

        return predictions_tl, predictions

    def forecast_adj(self, predictions_ad, inputs, forcings, targets_template):

        # normal adjoint flow

        # -- forward calculation --
        norm_inputs=normalize(inputs, self.stddev_by_level, self.mean_by_level)
        norm_forcings=normalize(forcings, self.stddev_by_level, self.mean_by_level)
        
        grid_node_features=inputs_to_grid_node_features(
            norm_inputs, norm_forcings)
        grid_node_features=np.squeeze(grid_node_features)  # removing dimension [batch]
        
        grid_nodes=np.concatenate((grid_node_features, self.grid_nodes_fixed), axis=-1)
        mesh_nodes=np.concatenate((
            np.zeros([self.mesh_nodes_fixed.shape[0], grid_node_features.shape[-1]]),
            self.mesh_nodes_fixed), axis=-1)
        # -- end forward calculation --
        
        norm_predictions_ad=predictions_ad*0.
        inputs_ad=inputs*0.

        for name in predictions_ad:
            norm_predictions_ad[name], inputs_ad=\
                unnormalize_prediction_and_add_input_adj(
                    inputs_ad, inputs, predictions_ad[name],
                    scales=self.stddev_by_level,
                    locations=self.mean_by_level*0.,
                    residual_scales=self.diffs_stddev_by_level)

        output_grid_nodes_ad=grid_node_outputs_to_prediction_adj(
            norm_predictions_ad, targets_template)

        grid_nodes_ad, mesh_nodes_ad=gc_grid2grid_adj(
            self.params, grid_nodes, output_grid_nodes_ad, mesh_nodes,
            self.edges, self.edge_indices
        )

        grid_node_features_ad=grid_nodes_ad[:,:-3]

        norm_inputs_ad=inputs_to_grid_node_features_adj(
            grid_node_features_ad, inputs_ad*0., forcings*0.
        )

        inputs_ad=inputs_ad+normalize(norm_inputs_ad, self.stddev_by_level, self.mean_by_level*0.)

        return inputs_ad

    def tlad_stitched(self, inputs_tl, inputs, forcings, targets_template):

        # TLM
        norm_inputs_tl=normalize(inputs_tl, self.stddev_by_level, self.mean_by_level*0.)
        norm_inputs=normalize(inputs, self.stddev_by_level, self.mean_by_level)

        norm_forcings=normalize(forcings, self.stddev_by_level, self.mean_by_level)

        grid_node_features_tl=inputs_to_grid_node_features(
            norm_inputs_tl, norm_forcings*0.)
        grid_node_features=inputs_to_grid_node_features(
            norm_inputs, norm_forcings)

        grid_node_features_tl=np.squeeze(grid_node_features_tl)
        grid_node_features=np.squeeze(grid_node_features)

        grid_nodes_tl=np.concatenate((grid_node_features_tl, self.grid_nodes_fixed*0.), axis=-1)
        grid_nodes=np.concatenate((grid_node_features, self.grid_nodes_fixed), axis=-1)

        mesh_nodes_tl=np.concatenate((
            np.zeros([self.mesh_nodes_fixed.shape[0], grid_node_features.shape[-1]]),
            self.mesh_nodes_fixed*0.), axis=-1)
        mesh_nodes=np.concatenate((
            np.zeros([self.mesh_nodes_fixed.shape[0], grid_node_features.shape[-1]]),
            self.mesh_nodes_fixed), axis=-1)

        edges_tl=deepcopy(self.edges)
        for key in edges_tl: edges_tl[key]*=0.
        output_grid_nodes_tl, output_grid_nodes=gc_grid2grid_tlm(
            self.params,
            grid_nodes_tl, grid_nodes,
            mesh_nodes_tl, mesh_nodes,
            edges_tl, self.edges, self.edge_indices
        )

        norm_predictions_tl=grid_node_outputs_to_prediction(
            output_grid_nodes_tl[:,np.newaxis,:], targets_template)

        predictions_tl=xarray_tree.map_structure(
            lambda pred: unnormalize_prediction_and_add_input(
                inputs_tl, pred, scales=self.stddev_by_level,
                locations=self.mean_by_level*0.,
                residual_scales=self.diffs_stddev_by_level),
            norm_predictions_tl)

        # predictions_tl.to_netcdf(path='predictions_tl.nc', mode='w')

        predictions_tl=xarray.open_dataset('predictions_tl.nc')

        LHS=0.
        for key in predictions_tl:
            LHS+=(predictions_tl[key].values**2).sum()


        print('LHS=', LHS)

        predictions_ad=predictions_tl.copy()  # getting from TLM

        # ADJ
        norm_predictions_ad=predictions_ad*0.
        inputs_ad=inputs_tl*0.

        for name in predictions_ad:
            norm_predictions_ad[name], inputs_ad=\
                unnormalize_prediction_and_add_input_adj(
                    inputs_ad, inputs, predictions_ad[name],
                    scales=self.stddev_by_level,
                    locations=self.mean_by_level*0.,
                    residual_scales=self.diffs_stddev_by_level)

        output_grid_nodes_ad=grid_node_outputs_to_prediction_adj(
            norm_predictions_ad, targets_template)

        grid_nodes_ad, mesh_nodes_ad=gc_grid2grid_adj(
            self.params, grid_nodes, output_grid_nodes_ad, mesh_nodes,
            self.edges, self.edge_indices
        )

        grid_node_features_ad=grid_nodes_ad[:,:-3]

        norm_inputs_ad=inputs_to_grid_node_features_adj(
            grid_node_features_ad, inputs_ad*0., forcings*0.
        )

        inputs_ad=inputs_ad+normalize(norm_inputs_ad, self.stddev_by_level, self.mean_by_level*0.)
        

def check_tlm(fwd, inputs, forcings, targets):
    idx2=(0,1,90,180)
    var2d=[
        '2m_temperature', 'mean_sea_level_pressure', '10m_v_component_of_wind', 
        '10m_u_component_of_wind', 'total_precipitation_6hr',
    ]
    idx3=(0,1,5,90,180)
    var3d=[
        'temperature', 'geopotential', 'u_component_of_wind', 
        'v_component_of_wind', 'vertical_velocity', 'specific_humidity',
    ]
    inputs_rand=inputs*0.
    for i in var2d+var3d:
        inputs_rand[i][()]=np.random.rand(*inputs[i].shape)


    pred_fwd=fwd.forecast(inputs, forcings, targets)

    for i in range(1, 10):

        t1=datetime.today()
        scale=10.**(-i)
        inputs_tl=inputs*inputs_rand*scale

        pred_tl, _=fwd.forecast_tlm(inputs_tl, inputs, forcings, targets)
        pred_tl.to_netcdf(path='Pred_tl_{:02d}.nc'.format(i), mode='w')

        pred_plus=fwd.forecast(inputs+inputs_tl, forcings, targets)
        pred_plus.to_netcdf(path='Pred_plus_{:02d}.nc'.format(i), mode='w')

        t2=datetime.today()
        print('finished scaling {:d} in '.format(i), t2-t1)

def check_adj(fwd, inputs, forcings, targets):
    np.random.seed(10)
    inputs_rand=inputs*0.
    for i in var2d+var3d:
        inputs_rand[i][()]=np.random.rand(*inputs[i].shape)

    inputs_tl=inputs*0.
    for i in var2d+var3d:
        inputs_tl[i][()]=inputs[i]*inputs_rand[i]*1.E-3

    #fwd.tlad_stitched(inputs_tl, inputs, forcings, targets)
    #exit('done with stitched')

    inputs_tl.to_netcdf(path='inputs_tl.nc', mode='w')
    t1=datetime.today()
    pred_tl, _=fwd.forecast_tlm(inputs_tl, inputs, forcings, targets)
    t2=datetime.today()

    pred_tl.to_netcdf(path='pred_tl.nc', mode='w')
    print('TLM: ', t2-t1)

    predictions_ad=pred_tl.copy()
    
    inputs_ad=fwd.forecast_adj(predictions_ad, inputs, forcings, targets)

    t3=datetime.today()
    inputs_ad.to_netcdf(path='inputs_ad.nc', mode='w')
    
    print('ADJ: ', t3-t2)
    
if __name__=='__main__':

    ### reading weights and biases
    with open('./notebook/GraphCast_small.npz', 'rb') as f:
    #with open('notebook/GraphCast_params_0.25deg.npz', 'rb') as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    params = ckpt.params
    model_config = ckpt.model_config
    task_config = ckpt.task_config

    # float32 to float64
    for key in params.keys():
        if 'norm' in key:
            params[key]['offset']=params[key]['offset'].astype(float)
            params[key]['scale']=params[key]['scale'].astype(float)
        else:
            params[key]['b']=params[key]['b'].astype(float)
            params[key]['w']=params[key]['w'].astype(float)
    #for key in params.keys():
    #    print(key)
    #exit()

    inputs, targets, forcings=loadInputs(
        task_config,
        fName='notebook/source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc'
    )

    for i in inputs:
        inputs[i]=inputs[i].astype(float)
    for i in forcings:
        forcings[i]=forcings[i].astype(float)
    for i in targets:
        targets[i]=targets[i].astype(float)

    ### normalization and denormalization
    with open("notebook/stats/diffs_stddev_by_level.nc", "rb") as f:
        diffs_stddev_by_level = xarray.load_dataset(f).compute()
    with open("notebook/stats/mean_by_level.nc", "rb") as f:
        mean_by_level = xarray.load_dataset(f).compute()
    with open("notebook/stats/stddev_by_level.nc", "rb") as f:
        stddev_by_level = xarray.load_dataset(f).compute()

    fwd=gcForecast(
        params, diffs_stddev_by_level, mean_by_level, stddev_by_level,
        staticFile='notebook/Static_Res_1.00deg.npz',
        #staticFile='notebook/Static_Res_0.25deg.npz',
    )

    predictions_ad=targets*0.
    holder=predictions_ad['temperature'].sel(level=500, lat=30, lon=270)
    holder[()]=1.

    inputs_ad=fwd.forecast_adj(predictions_ad, inputs, forcings, targets)
    inputs_ad.to_netcdf(path='InputsAD-T500-1Point.nc', mode='w')
    # check_tlm(fwd, inputs, forcings, targets)
    
    # check_adj(fwd, inputs, forcings, targets)

