# coding: utf-8

### author: Guodong Yu
###         Wuhan University, Wuhan, China
###         Radboud University, Nijmegen, The Netherlands
### email: yugd@live.cn

import numpy as np
import time
#import tipsi
from scipy.constants import golden_ratio
import random
import json
from monty.json import jsanitize
from scipy.linalg.lapack import zheev
import copy

def fibo(n):
    """
    Return Fibonacci number F(n)
    """
    if n<2:
        return n
    else:
        last = 1
        last2 = 0
        result= 0
        for i in range(2,n+1):
            result = last + last2
            last2 = last
            last = result
        return result

def insects_to_facets(grid_pair, insects):
    insects= np.array(np.floor(insects), dtype=int)
    i, j = grid_pair
    for i_add, j_add in [(0,0),(0,-1),(-1,0),(-1,-1)]:
        insts = copy.deepcopy(insects)
        insts[:,i] = insts[:,i] + i_add        
        insts[:,j] = insts[:,j] + j_add    
        try:
            facets = np.append(facets, insts, axis=0)
        except:
            facets = insts
    return facets    

class TilingRead(object):
    def __init__(self, **d):
        self.vertices = d['vertices']
        self.nsites = d['nsites']
        self.sides = d['sides']
        self.fat_diags = d['fat_diags']
        self.thin_diags = d['thin_diags']

    @staticmethod
    def from_dict(d):
        return TilingRead(**d)

    @staticmethod
    def from_file(fname):
        d = json.load(open(fname))
        return TilingRead(**d)




class Tiling(object):
    """
    Class for constructure p3 penrose tiling by multigrid(pentagrid) method
    Consolt paper: The Empire Problem in Penrose Tilings
    http://cs.williams.edu/~bailey/06le.pdf
    """

    def __init__(self, tau=golden_ratio):
        grid_vec0 = np.array([np.cos(0*2.*np.pi/5), np.sin(0*2.*np.pi/5)])
        grid_vec1 = np.array([np.cos(1*2.*np.pi/5), np.sin(1*2.*np.pi/5)])
        grid_vec2 =  1./tau * grid_vec1 - grid_vec0
        grid_vec3 = -1./tau * grid_vec1 - 1./tau* grid_vec0
        grid_vec4 = -grid_vec1 + 1./tau * grid_vec0
        self.grid_vecs = np.array([grid_vec0, grid_vec1, grid_vec2, grid_vec3, grid_vec4])
        self.grid_shifts = [0.01, 0.02, 0.03, -0.025, -0.035]

    def make_tiling(self, nlines):
        """
        Args:
            nlines: the number of parallel lines in one grid

        Key Attributes:
            self.grid_vecs: the grid vectors of the pentagrid
            self.grid_ks: the integer numbers labeling parallel lines in each grid
            self.grid_shifts: the shift values of each grid
            self.grid_interactions: the interactions coordinates of parallel lines of all grids
            self.grid_faces: all enclosed faces of the pentagrid
            self.vertices: the vertices coords of penrose tiling
            self.sides: the sides of penrose tiling 
            self.thin_diags: the shorter diagnol lines of all thin tiles
            self.fat_diags: the shorter diagnol lines of all fat tiles
        """
        if type(nlines) is int:
            self.nlines = [nlines, nlines, nlines, nlines, nlines]
        else:
            self.nlines = nlines
        self.grid_ks = [[-int(self.nlines[j]/2)+i for i in range(self.nlines[j])] for j in range(5)]
        self.grid_intersections, self.grid_intersections_detail = self._get_grid_intersections()
        self.grid_facets = self._get_grid_facets()
        self.vertices = self._get_vertices(self.grid_facets)
        self.sides, self.thin_diags, self.fat_diags = self._get_bonds()
        self.grid_facets_inds = self._get_facets_inds(self.grid_facets)
        self._bonds_inds()

    def _latt_vec(self, i, j):
        """
        The intersections between i-th and j-th grids construct a perodic lattice.
        This function return the original coordinate and the lattice vectors.
        
        Args:
            i: the i-th grid  0 <= i <= 4
            j: the j-th grid  0 <= j <=4 and j > i

        Return (orig, veci, vecj) tuple
            orig: the coordinate of origin
            veci: the lattice vector corresponding to the i-th grid
            vecj: the lattice vector cooresponding to the j-th grid   
        """
        shifts = self.grid_shifts
        ki_min = self.grid_ks[i][0]
        ki_sec_min = self.grid_ks[i][1]
        kj_min = self.grid_ks[j][0]
        kj_sec_min = self.grid_ks[j][1]

        A = np.array([ self.grid_vecs[i], self.grid_vecs[j]])
        b_min = np.array([-shifts[i]+ki_min, -shifts[j]+kj_min])
        b_i = np.array([-shifts[i]+ki_sec_min, -shifts[j]+kj_min])
        b_j = np.array([-shifts[i]+ki_min, -shifts[j]+kj_sec_min])
        orig = np.array(np.linalg.solve(A, b_min))
        pi1 = np.linalg.solve(A, b_i)
        pj1 = np.linalg.solve(A, b_j)
        veci = np.array(pi1) - orig
        vecj = np.array(pj1) - orig
        return (orig, veci, vecj)            
          
    def _get_grid_intersections(self):
        """
        The function will get all necessary info for penrose tiling which grids give:
        such as faces, intersections etc..
        """
        def get_nrange(m, i, j, v0, v1, orig, other_grids):
            """
            Desceription:
                (1) The lattices made of v0 and v1 (namely the intersections between grid0 and grid1) contain m*v0 + n*v1 +orig
                    and m = 0, 1, 2 ... self.nlines_per_grid-1
                        n = 0, 1, 2 ... self.nlines_per_grid-1
                (2) Only the points(intersections) enclosed by the lowest and highest lines of all grids make sense for 
                    generating the penrose tiling
            For a set of points of (m, n) with fixed m and n = 0, 1, 2 ... self.nlines_per_grid-1, this function will calculate
            and return the n range (n0, n1) for only the points in the region enclosed by lowest and highest lines of all grids 
            """
            
            n0 = 0
            n1 = self.nlines[j]-1
            o0 = orig[0]
            o1 = orig[1]
            for i in other_grids:
                k_min = self.grid_ks[i][0]
                k_max = self.grid_ks[i][-1]
                cos = self.grid_vecs[i][0]
                sin = self.grid_vecs[i][1]
                shift = self.grid_shifts[i]
                v1_dot_vi = v1[0]*cos + v1[1]*sin
                v0_dot_vi = v0[0]*cos+v0[1]*sin
                o_dot_vi = o0*cos + o1*sin
                if v1_dot_vi > 0.:
                    n0_i = int(np.ceil((k_min - shift -o_dot_vi -m*v0_dot_vi)/ v1_dot_vi))
                    n1_i = int(np.floor((k_max - shift -o_dot_vi -m*v0_dot_vi)/ v1_dot_vi))
                elif v1_dot_vi < 0.:
                    n1_i = int(np.floor((k_min - shift -o_dot_vi -m*v0_dot_vi)/ v1_dot_vi))
                    n0_i = int(np.ceil((k_max - shift -o_dot_vi -m*v0_dot_vi)/ v1_dot_vi))
                if n0_i > n0:
                    n0 = n0_i
                if n1_i < n1:
                    n1 = n1_i
            return (n0, n1)
            
        intersections = {}
        intersections_detail = {}
        for i in range(0,5):
            ki_min = self.grid_ks[i][0]
            for j in range(i+1,5):
                intersections[(i,j)]=[]
                intersections_detail[(i,j)]=[]
                kj_min = self.grid_ks[j][0]
                orig, veci, vecj = self._latt_vec(i, j)
                other_grids = [k for k in range(5) if k not in [i,j]]
                for m in range(self.nlines[i]):
                    n0, n1 = get_nrange(m, i, j, veci, vecj, orig, other_grids)
                    for n in range(n0,n1+1):
                        coord = m*veci+n*vecj+orig # intersection (m,n)
                        intersect = [None]*5
                        intersect[i] = m + ki_min
                        intersect[j] = n + kj_min 
                        for w in other_grids:
                            indw = np.dot(coord,self.grid_vecs[w]) + self.grid_shifts[w]
                            intersect[w] = indw
                        intersections_detail[(i,j)].append(intersect)
                        intersections[(i,j)].append([int(np.floor(k)) for k in intersect])
        return intersections, intersections_detail

    def _get_grid_facets(self):
        for grid_pair in self.grid_intersections:
            i, j = grid_pair
            insect = np.array(self.grid_intersections[grid_pair])
            facet = insects_to_facets(grid_pair, insect)
            try:
                facets = np.append(facets, facet, axis=0)
            except:
                facets = facet
        return np.unique(facets, axis=0)

    def _get_vertices(self, grid_facets):
        return np.matmul(grid_facets, self.grid_vecs)

    def _get_facets_inds(self, grid_facets):
        return dict(zip([tuple(i) for i in grid_facets], ((0,0,i) for i in range(len(grid_facets))) ))

    def get_vertice(self, facet):
        coord = 0.
        for i in range(5):
            coord = coord + facet[i]*self.grid_vecs[i]
        return coord

    def _get_bonds(self):

        def sort_side(side):
            i = tuple(side[0])
            j = tuple(side[1])
            return sorted([i,j], key=lambda x:(x[0],x[1],x[2],x[3],x[4]))

        fats = [(0,1),(0,4),(1,2),(2,3),(3,4)]
        thins = [(0,3),(1,3),(1,4),(2,4),(0,2)]
        tps = {'fat':fats, 'thin':thins}
        for tp in tps:
            for grid_pair in tps[tp]:
                i,j = grid_pair 
                intsects = np.array(self.grid_intersections[grid_pair])
                #facets = insects_to_facets(grid_pair, intsects)
                facets = []
                facets.append(intsects)
                for i_add,j_add in [(0,-1),(-1,0),(-1,-1)]:
                    intsects_tmp = copy.deepcopy(intsects)
                    intsects_tmp[:,i] = intsects_tmp[:,i] + i_add
                    intsects_tmp[:,j] = intsects_tmp[:,j] + j_add
                    facets.append(intsects_tmp)
                sides_00_01 = [sort_side([facets[0][m],facets[1][m]]) for m in range(len(intsects))]
                sides_00_10 = [sort_side([facets[0][m],facets[2][m]]) for m in range(len(intsects))]
                sides_10_11 = [sort_side([facets[2][m],facets[3][m]]) for m in range(len(intsects))]
                sides_01_11 = [sort_side([facets[1][m],facets[3][m]]) for m in range(len(intsects))]
                try:
                    sides = np.concatenate((sides, sides_00_01, sides_00_10, sides_10_11, sides_01_11))
                except:
                    sides = np.concatenate((sides_00_01, sides_00_10, sides_10_11, sides_01_11))
                if tp == 'fat':
                    sides_01_10 = [[tuple(facets[1][m]),tuple(facets[2][m])] for m in range(len(intsects))]
                    try:
                        fat_diags = np.concatenate((fat_diags, sides_01_10))
                    except:
                        fat_diags = np.array(sides_01_10)
                elif tp == 'thin':
                    sides_00_11 = [[tuple(facets[0][m]), tuple(facets[3][m])] for m in range(len(intsects))]
                    try:
                        thin_diags = np.concatenate((thin_diags, sides_00_11))
                    except:
                        thin_diags = np.array(sides_00_11)

        sides = np.unique(sides, axis=0) 
        #sides = np.array([i for i in sides if not np.sum(i[0][:-1])])
        #thin_diags = np.array([i for i in thin_diags if not np.sum(i[0][:-1])])
        #fat_diags = np.array([i for i in fat_diags if not np.sum(i[0][:-1])])
        return sides, thin_diags, fat_diags

    def _bonds_inds(self):
        inds = self.grid_facets_inds
        self.sides = np.array([[inds[tuple(side[0])], inds[tuple(side[1])]] for side in self.sides])
        self.thin_diags = np.array([[inds[tuple(side[0])], inds[tuple(side[1])]] for side in self.thin_diags])
        self.fat_diags = np.array([[inds[tuple(side[0])], inds[tuple(side[1])]] for side in self.fat_diags])
         
    
    def save_to(self, fname='tiling.json'):
        out = {}
        out['vertices'] = self.vertices
        out['nsites'] = self.nsites
        out['sides'] = self.sides
        out['fat_diags'] = self.fat_diags
        out['thin_diags'] = self.thin_diags
        with open(fname, 'w') as f:
            f.write(json.dumps(jsanitize(out)))
        
    def plot(self, ax=None, side=True, thin=False, fat=False, fig_name='Penrose_tiling.pdf',site_size=0, draw_dpi=600):
        if not ax:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            plot = True
        else:
            plot = False
        import matplotlib.collections as mc
        coords = self.vertices
        ax.scatter([i[0] for i in coords], [i[1] for i in coords], s=site_size)
        if side:
            lines = mc.LineCollection([[coords[i[0][-1]],coords[i[1][-1]]] for i in self.sides],\
                                                            [1.0]*len(self.sides),colors='black')
            ax.add_collection(lines)
        if thin:
            lines = mc.LineCollection([[coords[i[0][-1]],coords[i[1][-1]]] for i in self.thin_diags], \
                                                                 [1.0]*len(self.thin_diags),colors='blue')
            ax.add_collection(lines)
        if fat:
            lines = mc.LineCollection([[coords[i[0][-1]],coords[i[1][-1]]] for i in self.fat_diags], \
                                                                 [1.0]*len(self.fat_diags),colors='red')
            ax.add_collection(lines)

        site_x = self.vertices[:,0]
        site_y = self.vertices[:,1]


        xmin = np.min(site_x)
        xmax = np.max(site_x)
        ymin = np.min(site_y)
        ymax = np.max(site_y)
        #ax.set_xlim((xmin,xmax))
        #ax.set_ylim((ymin,ymax))
        ax.set_aspect('equal')
        if plot:
            plt.axis('off')
            plt.savefig(fig_name, bbox_inches='tight', draw_dpi=draw_dpi)
            plt.close()

class PeriodicTiling(Tiling):
    def make_tiling(self, fibo_idx):
        self.fibo_idx = fibo_idx
        Fn_ = fibo(fibo_idx+1)
        Fn = fibo(fibo_idx)
        tau = Fn_/Fn
        self._Fn_ = Fn_
        self._Fn = Fn
        nlines = Fn_+3
        Tiling.__init__(self, tau=tau)
        self.nlines = [nlines, nlines, nlines*2, nlines*2, nlines*2]
        self.grid_ks = [[-int(self.nlines[j]/2)+i for i in range(self.nlines[j])] for j in range(5)]
        self.grid_intersections, self.grid_intersections_in_detail = self._get_grid_intersections()
        self.grid_facets = self._get_grid_facets()
        self._grid_facets_classify()
        self.vertices = self._get_vertices(self.grid_facets_classified[(0,0)])
        self._get_facets_inds_ppt()
        self.sides, self.thin_diags, self.fat_diags = self._get_bonds()
        self._bonds_inds_ppt()

    def grid_latt_vec(self):
        theta = 2*np.pi/5
        latt_cont = 1/np.sin(theta) * fibo(self.fibo_idx+1)
        vec0 = latt_cont* np.array([np.sin(theta), -np.cos(theta)])
        vec1 = latt_cont*np.array([0, 1.])
        return np.array([vec0, vec1])

    def _plot_grids_lines(self, ax):
        cs = ['black','red','blue','green','purple']
        for i in range(5):
            ei = self.grid_vecs[i]
            ax.arrow(-0.5,0, ei[0], ei[1], color=cs[i],width=0.1, alpha=0.2)
            ax.text(-0.5+ei[0], ei[1], str(i),  color=cs[i], fontsize=20)

        def write_ks(i, y, color):
            for k in self.grid_ks[i]:
                x = (k-self.grid_shifts[i]-y*self.grid_vecs[i][1])/self.grid_vecs[i][0]
                ax.text(x, y, str(k), color=color, fontsize=30)
        for i in range(5):
            y = i - 3
            write_ks(i, y, cs[i])        

        def f(x, grid, k):
            ei = self.grid_vecs[grid]
            shift = self.grid_shifts[grid]
            return (k-shift-x*ei[0])/ei[1]
        for i in range(1,5):
            ei = self.grid_vecs[i]
            x = [-10/ei[0],10/ei[0]]
            x = [j*ei[0] for j in x]
            for k in self.grid_ks[i]:
                y = [f(j, i, k) for j in x]
                ax.plot(x,y,c=cs[i])
        for k in self.grid_ks[0]:
            y = [-10, 10]
            x = (k - self.grid_shifts[0])/self.grid_vecs[0][0]
            if k==self.grid_ks[0][0]:
                xmin = x
            if k==self.grid_ks[0][-1]:
                xmax = x
            ax.plot([x,x],y, c='black')

        latt = self.grid_latt_vec()
        kmin = self.grid_ks[0][0]
        shift0 = self.grid_shifts[0]
        shift1 = self.grid_shifts[1]
        e0 = self.grid_vecs[0]
        e1 = self.grid_vecs[1]
        x0 = (kmin-shift0)/e0[0]
        y0 = (kmin-shift1-x0*e1[0])/e1[1]
        O = np.array([x0,y0])
        A = O + latt[0]
        B = O + latt[1]
        C = O + latt[0] + latt[1]
        #ax.plot([O[0], A[0],C[0],B[0],O[0]], [O[1], A[1],C[1],B[1],O[1]], ls='dashed', c='black', lw=3)
        ax.set_xlim(xmin-0.01,xmax+0.01)
        ax.set_ylim(xmin-1,xmax+1)

    def _plot_grids(self):
        from matplotlib import pyplot as plt
        #plt.rcParams.update({'figure.figsize':[10,10]})
        fig, ax = plt.subplots(1,1)
        ax.axis('equal')
        self._plot_grids_lines(ax)
        self._plot_intersections(ax)
        plt.show()

    def _plot_intersections(self, ax):
        for grid_pair in self.grid_intersections_in_detail:
            i, j = grid_pair
            A = np.array([self.grid_vecs[i], self.grid_vecs[j]])
            shifti = self.grid_shifts[i]
            shiftj = self.grid_shifts[j]
            for intsect in self.grid_intersections_in_detail[grid_pair]:
                b = np.array([intsect[i]-shifti, intsect[j]-shiftj])
                X = np.linalg.solve(A, b)
                ax.scatter(X[0], X[1], c='red',marker='*',s=100, alpha=0.5)

    def _grid_facets_classify(self):
        kmin = self.grid_ks[0][0]
        kmax = self.grid_ks[0][-1]
        ind0 = np.where(self.grid_facets[:,0]==kmin-1)[0]
        ind1 = np.where(self.grid_facets[:,0]==kmax)[0]
        ind2 = np.where(self.grid_facets[:,1]==kmin-1)[0]
        ind3 = np.where(self.grid_facets[:,1]==kmax)[0]
        ind = range(len(self.grid_facets))
        ind_rm = np.unique(np.concatenate((ind0, ind1, ind2, ind3)))
        ind_left = np.setdiff1d(ind, ind_rm)
        grid_facets = self.grid_facets[ind_left]
        self.grid_facets = grid_facets

        grid_facets_classified = {}
        inds11 = np.intersect1d(np.where(grid_facets[:,0]==kmax-1)[0], np.where(grid_facets[:,1]==kmax-1)[0])
        inds_1_1 = np.intersect1d(np.where(grid_facets[:,0]==kmin)[0], np.where(grid_facets[:,1]==kmin)[0])
        inds1_1 = np.intersect1d(np.where(grid_facets[:,0]==kmax-1)[0], np.where(grid_facets[:,1]==kmin)[0])
        inds_11 = np.intersect1d(np.where(grid_facets[:,0]==kmin)[0], np.where(grid_facets[:,1]==kmax-1)[0])
        inds10 = np.intersect1d( np.where(grid_facets[:,0]==kmax-1)[0], \
                     np.intersect1d(np.where(grid_facets[:,1]<kmax-1)[0], np.where(grid_facets[:,1]>kmin)[0]))
        inds_10 = np.intersect1d( np.where(grid_facets[:,0]==kmin)[0], \
                     np.intersect1d(np.where(grid_facets[:,1]<kmax-1)[0], np.where(grid_facets[:,1]>kmin)[0]))
        inds01 = np.intersect1d( np.where(grid_facets[:,1]==kmax-1)[0], \
                     np.intersect1d(np.where(grid_facets[:,0]<kmax-1)[0], np.where(grid_facets[:,0]>kmin)[0]))
        inds0_1 = np.intersect1d( np.where(grid_facets[:,1]==kmin)[0], \
                     np.intersect1d(np.where(grid_facets[:,0]<kmax-1)[0], np.where(grid_facets[:,0]>kmin)[0]))
        inds_out = np.unique(np.concatenate((inds11, inds_1_1, inds1_1, inds_11, inds10, inds_10, inds01, inds0_1)))
        inds00 = np.setdiff1d(range(len(grid_facets)), inds_out)
        grid_facets_classified = {(0,0):grid_facets[inds00], (0,1):grid_facets[inds01],\
                                       (1,0):grid_facets[inds10], (1,1):grid_facets[inds11], \
                                       (-1,1):grid_facets[inds_11]}

        for i,j in [(1,0),(0,1),(1,1),(-1,1)]:
            facets = grid_facets_classified[(i,j)]
            facets_folded = self.facet_shift(facets, (-i,-j))
            grid_facets_classified[(i,j)] = {tuple(facets[i]):tuple(facets_folded[i]) for i in range(len(facets))}
        self.grid_facets_classified = grid_facets_classified
        
                
    def _get_facets_inds_ppt(self):
        self.grid_facets_inds = self._get_facets_inds(self.grid_facets_classified[(0,0)])
        for uc in [(0,1),(1,0),(1,1),(-1,1)]:
            for i in self.grid_facets_classified[uc]:
                self.grid_facets_inds[i] = (uc[0],uc[1],self.grid_facets_inds[self.grid_facets_classified[uc][i]][-1])

    def _bonds_inds_ppt(self):
        inds = self.grid_facets_inds
        sides = []
        thins = []
        fats = []
        def bond_sort(side):
            return sorted(side, key=lambda x:(x[1],x[0]))
        for side in self.sides:
            try:
                sides.append(bond_sort([inds[tuple(side[0])], inds[tuple(side[1])]]))
            except:
                pass
        for thin in self.thin_diags:
            try:
                thins.append(bond_sort([inds[tuple(thin[0])], inds[tuple(thin[1])]]))
            except:
                pass
        for fat in self.fat_diags:
            try:
                fats.append(bond_sort([inds[tuple(fat[0])], inds[tuple(fat[1])]]))
            except:
                pass
        sides = np.array([i for  i in sides if not ((i[0][0] or i[0][1]))])
        thin_diags = np.array([i for  i in thins if not ((i[0][0] or i[0][1]))])
        fat_diags = np.array([i for  i in fats if not ((i[0][0] or i[0][1]))])

        def bonds_post(bonds, uc):
            n = len(bonds)
            bond0 = bonds[:,0][:,-1].reshape(n,1)
            bond1 = bonds[:,1]
            ind0 = np.where(bond1[:,0]==uc[0])[0]
            ind1 = np.where(bond1[:,1]==uc[1])[0]
            ind = np.intersect1d(ind0, ind1)
            bond1 = bond1[:,-1].reshape(n,1)
            return np.append(bond0[ind], bond1[ind], axis=1)
        self.sides = {}
        self.thin_diags = {}
        self.fat_diags = {}
        for uc in [(0,0),(0,1),(1,0),(-1,1),(1,1)]:
            side = bonds_post(sides, uc)
            if len(side):
                self.sides[uc] = side
            thin = bonds_post(thin_diags, uc)
            if len(thin):
                self.thin_diags[uc] = thin
            fat = bonds_post(fat_diags, uc)
            if len(fat):
                self.fat_diags[uc] = fat

    def facet_shift(self, facet, direct):
        """
        direct: [i,j] means facet move along i*vec0 + j*vec1 
                vec0 and vec1 are the lattice vectors of the unit cell of the grid (enclosed by grid0 and grid1)
        """
        shift = direct[0]*np.array([self._Fn_,0,-self._Fn_,-self._Fn,self._Fn]) + \
                direct[1]*np.array([0,self._Fn_,self._Fn,-self._Fn, -self._Fn_])
        return facet + shift

    def get_orig_coord(self):
        kmin = self.grid_ks[0][0]
        O_grid = np.linalg.solve(np.array([self.grid_vecs[0],self.grid_vecs[1]]), \
                   np.array([kmin-self.grid_shifts[0], kmin-self.grid_shifts[1]]))
        return O_grid

    def tiling_latt_vec(self):
        kmin = self.grid_ks[0][1]
        O_grid = np.linalg.solve(np.array([self.grid_vecs[0],self.grid_vecs[1]]), \
                   np.array([kmin-self.grid_shifts[0], kmin-self.grid_shifts[1]]))
        facet_O = [kmin-1, kmin-1]
        for i in range(2,5):
            w = np.floor(np.dot(O_grid, self.grid_vecs[i])+self.grid_shifts[i])
            facet_O.append(int(w))
        facet_O = np.array(facet_O)
        facet_A = self.facet_shift(facet_O, [1,0])
        facet_B = self.facet_shift(facet_O, [0,1])
        facet_C = self.facet_shift(facet_O, [1,1])

        vs = []
        for facet in [facet_O, facet_A, facet_B, facet_C]:
            vs.append(np.matmul(facet, self.grid_vecs))
        return vs

    def plot_ppt(self, size=[2,2], thin=False, fat=False, site_size=0):
        from matplotlib import pyplot as plt
        import matplotlib.collections as mc
        fig, ax = plt.subplots(1,1)
        O,A,B,C = self.tiling_latt_vec()
        v0 = A - O
        v1 = B - O
        cs = {(0,1):'red',(1,0):'green',(1,1):'yellow'}
        ms = {(0,1):'^',(1,0):'>',(1,1):'*'}
        carts = {}
        for i in range(size[0]):
            for j in range(size[1]):
                coords = self.vertices + i*v0 + j*v1
                carts[(i,j)] = coords
                ax.scatter(coords[:,0], coords[:,1], s=site_size, color='black')                
        for i in range(size[0]+1):
            vj = v1*size[1]
            Xi = O + i*v0
            Xi_ = Xi + vj
            ax.plot([Xi[0], Xi_[0]],[Xi[1], Xi_[1]], ls='dashed', color='yellow')
        for j in range(size[1]+1):
            vi = v0*size[0]
            Xi = O + j*v1
            Xi_ = Xi + vi
            ax.plot([Xi[0], Xi_[0]],[Xi[1], Xi_[1]], ls='dashed', color='yellow')

        for i in range(size[0]):
            for j in range(size[1]):
                for uc in self.sides:
                    side = self.sides[uc]
                    coord0 = carts[(i,j)]
                    try:
                        coord1 = carts[(i+uc[0],j+uc[1])]
                        line = mc.LineCollection([[coord0[k[0]],coord1[k[1]]] for k in side], color='black')
                        ax.add_collection(line)
                    except:
                        pass
                    if thin:
                        thins = self.thin_diags[uc]
                        coord0 = carts[(i,j)]
                        try:
                            coord1 = carts[(i+uc[0],j+uc[1])]
                            line = mc.LineCollection([[coord0[k[0]],coord1[k[1]]] for k in thins], color='red')
                            ax.add_collection(line)
                        except:
                            pass
                    if fat:
                        fats = self.fat_diags[uc]
                        coord0 = carts[(i,j)]
                        try:
                            coord1 = carts[(i+uc[0],j+uc[1])]
                            line = mc.LineCollection([[coord0[k[0]],coord1[k[1]]] for k in fats], color='blue')
                            ax.add_collection(line)
                        except:
                            pass
        ax.set_aspect('equal')                
        plt.show()

