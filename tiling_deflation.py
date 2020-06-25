# coding: utf-8
### the deflation (up-down) method to generate the 2D penrose_P3 tiling

### author: Guodong Yu
###         Wuhan University, Wuhan, China
###         Radboud University, Nijmegen, The Netherlands
### email: yugd@live.cn

import numpy as np
from scipy.constants import golden_ratio
from numpy import pi
import json
from monty.json import jsanitize
import time
import copy
#from mv import *

tau = golden_ratio - 1.
    
def rotate_operator(theta):
    """
    [0, 0] is the rotation centor
    """
    rad = theta*np.pi/180.
    mat = np.array([[np.cos(rad), -np.sin(rad)],[np.sin(rad), np.cos(rad)]])
    return mat

def rotate_on_vec(theta, vec):
    mat = rotate_operator(theta)
    return np.dot(mat, vec)

def rotate_angle(vec):
    """
    the rotate angle of the vec relative to the x axis
    """
    if np.abs(vec[0]) <= 1.0e-12:
        if vec[1] > 0.:
            return 90.
        elif vec[1] < 0.:
            return 270.
    if np.abs(vec[1]) <= 1.0e-12:
        if vec[0] > 0.:
            return 0.
        elif vec[0] < 0.:
            return 180.
    theta = np.arctan(vec[1]/vec[0]) /np.pi * 180.
    if vec[0]*vec[1]>0.:
        if vec[0] > 0.:
            return theta
        elif vec[0] <0.:
            return 180.+theta
    elif vec[0]*vec[1] <0.:
        if vec[0] >0.:
            return theta + 360.
        elif vec[0] <0.:
            return theta + 180. 

def plt_set(plt,nrow=1,ncolumn=1,width=30,height=30,scale=10):
    fig_width = width*ncolumn
    fig_height = height*nrow
    fig_size = [fig_width, fig_height]
    params = {'backend': 'ps',
              'axes.labelsize': 4*scale,
              'font.size': 3*scale,
              'font.weight': 'normal',
              'legend.fontsize': 4*scale,
              'xtick.labelsize': 4*scale,
              'ytick.labelsize': 4*scale,
              'text.usetex': False,
              'figure.figsize': fig_size,
              'lines.markersize': 2*scale,
              'lines.linewidth':  0.7*scale
              }
    plt.rcParams.update(params)

def deflate_tile(vertexes, tile):
    """
    Description:
        Deflate one tile.
    Args:
        vertexes: the coordinates of all vertexes
        tile: the tile that will be deflated
              tile is given in list ******
    comments: after deflation, the side length of the rombus will shrink 
    """              
    
    begin = vertexes[tile[1]]
    end = vertexes[tile[0]]
    tile_type = tile[-1]
    a = np.linalg.norm(np.array(end) - np.array(begin))
    vec_BE = np.array(end) - np.array(begin)
    theta = rotate_angle(vec_BE)
    if tile_type == 'thin':
        t0 = rotate_on_vec(theta, [a*(1. - np.cos(36.*pi/180.)), a*np.sin(36.*pi/180.)]) + np.array(begin)
        t1 = rotate_on_vec(theta, [-a*np.cos(36.*pi/180.), a*np.sin(36.*pi/180.)]) + np.array(begin)
        t2 = rotate_on_vec(theta, [a*(1.+ np.cos(72.*pi/180.)), a*np.sin(72.*pi/180.)]) + np.array(begin)
        new_pnts = [t0, t1, t2] 

        tile1 = [tile[1], '0', tile[0], '1', 'thin']
        tile2 = [tile[2], tile[0], '0', '2', 'fat']
        tiles = [tile1, tile2]

        if True:
            t0m = rotate_on_vec(theta, [a*(1-np.cos(36.*pi/180.)), -a*np.sin(36.*pi/180.)]) + np.array(begin)
            t1m = rotate_on_vec(theta, [-a*np.cos(36.*pi/180.), -a*np.sin(36.*pi/180.)]) + np.array(begin)
            t2m = rotate_on_vec(theta, [a*(1.+ np.cos(72.*pi/180.)), -a*np.sin(72.*pi/180.)]) +  np.array(begin)
            new_pnts = [t0, t1, t2, t0m, t1m, t2m]
            tile1m = [tile[1], '3', '4', tile[0], 'thin']
            tile2m = [tile[3], tile[0], '5', '3', 'fat']
            tiles.append(tile1m)
            tiles.append(tile2m)
        return {'new_pnts':new_pnts, 'new_tiles':tiles}

    elif tile_type == 'fat':
        l = a / ( 2.*np.sin(54.*pi/180.) + 1. )
        t0 = rotate_on_vec(theta, [2.*l*np.sin(54.*pi/180.), 0.]) +  np.array(begin)
        t1 = rotate_on_vec(theta, [l*np.sin(54.*pi/180), l*np.cos(54.*pi/180.)])+ np.array(begin)
        t2 = rotate_on_vec(theta, [a-(l/tau+l)*np.cos(36.*pi/180.), (l/tau+l)*np.sin(36.*pi/180.)])+ np.array(begin)
        t3 = rotate_on_vec(theta, [a-l*np.cos(72.*pi/180), l*np.sin(72.*pi/180)])+ np.array(begin)
        
        t1m = rotate_on_vec(theta, [l*np.sin(54.*pi/180), -l*np.cos(54.*pi/180.)])+ np.array(begin)
        new_pnts = [t0, t1, t2, t3, t1m]
        tile1 = [tile[1], '0', '4', '1', 'fat']
        tile2 = [tile[2], '1', '2', '0', 'thin']
        tile3 = [tile[2], tile[0], '0', '3', 'fat']
        tiles = [tile1, tile2, tile3]
        if True:
            t2m = rotate_on_vec(theta, [a-(l/tau+l)*np.cos(36.*pi/180.), -(l/tau+l)*np.sin(36.*pi/180.)]) + np.array(begin)
            t3m = rotate_on_vec(theta, [a-l*np.cos(72.*pi/180), -l*np.sin(72.*pi/180)]) + np.array(begin)
            new_pnts = [t0, t1, t2, t3, t1m, t2m, t3m]
            tile2m = [tile[3], '4', '0', '5', 'thin']
            tile3m = [tile[3], tile[0], '6', '0', 'fat']
            tiles.append(tile2m)
            tiles.append(tile3m)
        return {'new_pnts':new_pnts, 'new_tiles': tiles}  

def deflate_tiling(tiling, n_processes=1):
    """
    Description:
        Deflate the whole tiling one time
    Args:
        tiling: the tiling, that will be deflated 
        it should be the instantiation of class TilingSetup or Tiling below
    """
    def pnt_in(vertexes, pnt):
        pnt = np.array(pnt)
        ind = None
        if not vertexes:
            return (False, ind)
        else:
            include = False
            for i in range(len(vertexes)):
                v = np.array(vertexes[i])
                if np.linalg.norm(v - pnt) <= 1.0e-5:
                    include = True
                    ind = i
                    break
            return (include, ind)
        
    def tile_in(tiles, tile):
        if not tiles:
            return False
        inside = False
        for i in tiles:
            if tile == i:
                inside = True
                break
        return inside
            
    vertexes = copy.deepcopy(tiling.vertices)
    tiles = copy.deepcopy(tiling.tiles)
    added_tiles = []
    if n_processes ==1:
        for tile in tiles:
            new_part = deflate_tile(vertexes, tile)
            new_pnts = new_part['new_pnts']
            new_tiles = new_part['new_tiles']
            ind_pnts = {}
            
            for i in range(len(new_pnts)):
                pt = new_pnts[i]
                inside, ind = pnt_in(vertexes, pt)
                if inside:
                    ind_pnts['%i' % i] = ind
                else:
                    vertexes.append(new_pnts[i])
                    ind_pnts['%i' % i] =  len(vertexes)-1
                    
            for t in new_tiles:
                p0 = t[0] if type(t[0]) is int else ind_pnts[t[0]]
                p1 = t[1] if type(t[1]) is int else ind_pnts[t[1]]
                p2 = t[2] if type(t[2]) is int else ind_pnts[t[2]]
                p3 = t[3] if type(t[3]) is int else ind_pnts[t[3]]
                tile_new = [p0, p1, p2, p3, t[-1]]
                if not tile_in(added_tiles, tile_new):
                    added_tiles.append(tile_new)
    else:
        tiles_div, N = grouper(tiles, n_processes)
        processes = [None for i in range(N)]
        pipes = [mpg.Pipe() for i in range(N)]
        for i, tiles_sub in enumerate(tiles_div):
            pipe = pipes[i]
            processes[i] = mpg.Process(target=self.__get_neighs, \
                                      args=(cells_sub, 'intra', 'site0' , pipe[1]))
            processes[i].start()
        scan = [True for i in range(N)]
        while any(scan):
            for i in range(N):
                pipe = pipes[i]
                if scan[i] and pipe[0].poll():
                    # get data, close process
                    neigh_pairs_div, neigh_dists_div, neigh_types_div = pipe[0].recv()
                    neigh_pairs = neigh_pairs + neigh_pairs_div
                    neigh_dists = neigh_dists + neigh_dists_div
                    neigh_types = neigh_types + neigh_types_div
                    scan[i] = False
                    pipe[0].close()
                    processes[i].join()

    out_dict = {}
    out_dict['seed'] = tiling.seed
    out_dict['n'] = tiling.n + 1
    out_dict['tiles'] = added_tiles
    out_dict['vertices'] = (np.array(vertexes)/tau).tolist()
    out_dict['nsites'] = len(out_dict['vertices'])
    out_dict['a'] = tiling.a
    return Tiling.from_dict(out_dict)

class TilingFromSeed(object):
    def __init__(self, n, a=1.0, seed='star'):
        """
        Args:
            from_dict: whether read the tiling from a list
            if from_dict:
                kws: the list output by Tiling.as_dict()
            else:
                kws:
                    n: the deflation time from seed, no default 
                    a: the side length for all tiles (default 1.0)
                    seed: the seed type 'sun' or 'star'(default) 
        """
        self.n = n
        self.a = a
        if seed == 'star':
            self.seed = self._seed_star()
        elif seed == 'sun':
            self.seed = self._seed_sun()
        self._make_tiling_from_seed()
        self.sides = self._sides()
        self.thin_diags, self.fat_diags = self._tile_diags()

    def _sides(self):
        sides_tag=[]
        for tile in self.tiles:
            s0 = str(tile[0])+'.'+str(tile[2]) if tile[0] < tile[2] else str(tile[2])+'.'+str(tile[0])
            s1 = str(tile[0])+'.'+str(tile[3]) if tile[0] < tile[3] else str(tile[3])+'.'+str(tile[0])
            s2 = str(tile[1])+'.'+str(tile[2]) if tile[1] < tile[2] else str(tile[2])+'.'+str(tile[1])
            s3 = str(tile[1])+'.'+str(tile[3]) if tile[1] < tile[3] else str(tile[3])+'.'+str(tile[1])
            sides_tag.append(s0)
            sides_tag.append(s1)
            sides_tag.append(s2)
            sides_tag.append(s3)
        sides_reduced = set(sides_tag)
        sides=[]
        for i in sides_reduced:
            s = [int(i) for i in i.split('.')]
            sides.append(s)
        return sides

    def _tile_diags(self):
        thins = [i for i in self.tiles if i[-1]=='thin']
        fats = [i for i in self.tiles if i[-1]=='fat']

        fat_diags = []
        thin_diags = []

        for tile in thins:
            orb0 = tile[0]
            orb1 = tile[1]
            if orb0 < orb1:
                thin_diags.append([orb0, orb1])
            else:
                thin_diags.append([orb1, orb0])

        for tile in fats:
            orb0 = tile[2]
            orb1 = tile[3]
            if orb0 < orb1:
                fat_diags.append([orb0, orb1])
            else:
                fat_diags.append([orb1, orb0])
        return thin_diags, fat_diags


    def as_dict(self):
        out={}
        out['seed'] = self.seed
        out['nsites'] = self.nsites
        out['vertices'] = self.vertices
        out['tiles'] = self.tiles
        out['n'] = self.n
        out['a'] = self.a
        out['sides'] = self.sides
        out['fat_diags'] = self.fat_diags
        out['thin_diags'] = self.fat_diags
        return out

    def save_to(self, fname='Penrose_tiling.json'):
        out={}
        out['seed'] = self.seed
        out['nsites'] = self.nsites
        out['vertices'] = self.vertices
        out['tiles'] = self.tiles
        out['n'] = self.n
        out['a'] = self.a
        out['sides'] = self.sides
        out['fat_diags'] = self.fat_diags
        out['thin_diags'] = self.fat_diags
        with open(fname, 'w') as f:
            f.write(json.dumps(jsanitize(out)))
 
    def _seed_star(self):
        a = self.a
        aa = np.array([a, 0])
        bb = np.array([a*np.cos(72.*pi/180.), a*np.sin(72.*pi/180.)])
        orig = np.array([0.,0.])
        v0 = np.array([1., 0.])
        v1 = np.array([0., 1.])
        v2 = np.array([-1., 2.*np.cos(72.*pi/180.)])
        v3 = np.array([-0.5/np.sin(54.*pi/180.), -0.5/np.sin(54.*pi/180.)])
        v4 = np.array([2.*np.cos(72.*pi/180.), -1.])
        
        v5 = v0 + v1
        v6 = v1 + v2
        v7 = v2 + v3
        v8 = v3 + v4
        v9 = v0 + v4
        
        ##### vertexes ###
        vertexes = []
        vertexes.append(orig.tolist())
        for i in [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9]:
            veci = i[0]*aa + i[1]*bb
            vertexes.append(veci.tolist())
            
        tiles = []
        ##### arrowed tiles ####
        tiles.append([ 0, 6, 1, 2, 'fat']) # tile arrow 6->0 
        tiles.append([ 0, 7, 2, 3, 'fat'])
        tiles.append([ 0, 8, 3, 4, 'fat'])
        tiles.append([ 0, 9, 4, 5, 'fat'])
        tiles.append([ 0, 10, 5, 1, 'fat'])
        return  {'vertices': vertexes, 'tiles':tiles, 'a':a, 'nsites':len(vertexes)}

    def _seed_sun(self):
        a = self.a
        seed = self._seed_star(a)
        pt = [np.array(i) for i in seed['vertices']]
        tiles = seed['tiles']
        a = seed['a']
        n0 = pt[2] + (pt[6]-pt[2]) + (pt[7]-pt[2])
        n1 = pt[3] + (pt[7]-pt[3]) + (pt[8]-pt[3])
        n2 = pt[4] + (pt[8]-pt[4]) + (pt[9]-pt[4])
        n3 = pt[5] + (pt[10]-pt[5]) + (pt[9]-pt[5])
        n4 = pt[1] + (pt[6]-pt[1]) + (pt[10]-pt[1])
        for i in [n0, n1, n2, n3, n4]:
            pt.append(i)
        tiles.append([11, 2, 7, 6, 'thin'])
        tiles.append([12, 3, 8, 7, 'thin'])
        tiles.append([13, 4, 9, 8, 'thin'])
        tiles.append([14, 5, 10, 9, 'thin'])
        tiles.append([15, 1, 6, 10, 'thin'])
        return {'vertices': [i.tolist() for i in pt], 'tiles': tiles, 'a': a, 'nsites':len(pt)}
                
    def _seed2tiling(self):
        tiling = {}
        seed = self.seed
        tiling['seed'] = seed
        tiling['a'] = self.a
        tiling['n'] = 0
        tiling['nsites'] = seed['nsites']
        tiling['vertices'] = seed['vertices']
        tiling['tiles'] = seed['tiles']
        tiling = Tiling.from_dict(tiling)
        return tiling
    
    def _make_tiling_from_seed(self):
        tiling = self._seed2tiling()
        n = 0
        while n < self.n:
            tiling = deflate_tiling(tiling)
            n += 1
        self.nsites = tiling.nsites
        self.vertices = tiling.vertices
        self.tiles = tiling.tiles

    def plot(self, fig_name='Penrose_tiling.pdf',thin_diags=False, fat_diags=False, draw_dpi = 600):
        import matplotlib.pyplot as plt
        import matplotlib.collections as mc
        plt_set(plt)
        fig, ax = plt.subplots()
        pnts = self.vertices
        for i in pnts:
            ax.scatter(i[0],i[1])
        sides = []
        site_x = [i[0] for i in pnts]
        site_y = [i[1] for i in pnts]
        xmin = np.min(site_x)
        xmax = np.max(site_x)
        ymin = np.min(site_y)
        ymax = np.max(site_y)
        for s in self.sides:
            side = [pnts[s[0]],pnts[s[1]]]
            sides.append(side)
        lines = mc.LineCollection(sides, [0.1]*len(sides),colors='black')
        ax.add_collection(lines)
        if thin_diags:
            thin_diags_coord = []
            for i in self.thin_diags:
                thin_diags_coord.append([pnts[i[0]],pnts[i[1]]])
            lines = mc.LineCollection(thin_diags_coord, [0.1]*len(self.thin_diags),colors='blue')
            ax.add_collection(lines)
        if fat_diags:
            fat_diags_coord = []
            for i in self.fat_diags:
                fat_diags_coord.append([pnts[i[0]],pnts[i[1]]])
            lines = mc.LineCollection(fat_diags_coord, [0.1]*len(self.fat_diags),colors='red')
            ax.add_collection(lines)
        ax.set_aspect('equal')
        plt.axis('off')
        plt.xlim((xmin,xmax))
        plt.ylim((ymin,ymax))
        plt.draw()
        plt.savefig(fig_name, bbox_inches='tight', draw_dpi=draw_dpi)
        plt.close()
    
class Tiling(TilingFromSeed):
    def __init__(self, **kws):
        """
        Args:
            from_dict: whether read the tiling from a list
            if from_dict:
                kws: the list output by Tiling.as_dict()
            else:
                kws:
                    n: the deflation time from seed, no default 
                    a: the side length for all tiles (default 1.0)
                    seed: the seed type 'sun' or 'star'(default) 
        """
        self.n = kws['n'] # the deflated time of the tiling from seed
        self.a = kws['a'] # the side length of the tiles
        try:
            self.nsites = kws['npts'] # the number of vertexes
        except:
            self.nsites = kws['nsites']
        self.seed = kws['seed'] # the seed of the tiling
        try:
            self.vertices = kws['vertexes'] # All the vertexes' coords of the tiling
        except:
            self.vertices = kws['vertices']
        self.tiles = kws['tiles'] # All tiles of the tiling
        try:
            self.sides = self._sides() #kws['sides']
        except:
            self.sides = kws['sides']
        try:
            self.fat_diags = kws['fat_diags']
            self.thin_diags = kws['thin_diags']
        except:
            self.thin_diags, self.fat_diags = self._tile_diags()
    @classmethod
    def from_dict(cls, d):
        """
        Args:
            d: the dict of self.as_dict()
        """
        return cls(**d)
    
    @staticmethod
    def from_file(fname):
        j = json.load(open(fname))
        return Tiling.from_dict(j)
 
