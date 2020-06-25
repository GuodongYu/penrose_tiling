import numpy as np
import time
import tipsi
from scipy.constants import golden_ratio
import random
import json
from monty.json import jsanitize
from scipy.linalg.lapack import zheev

def random_nelms_in_list(l, n):
    """
    Random pick up n elements from a list
    Args:
        l: the list
        n: the number of choosen elements
    """
    if n == 0:
        return []
    if n == len(l):
        return l
    data = [i for i in l]
    size = len(data)
    
    data_pick = []
    
    left = False
    if int(size/2.) < n:
        n = len(l) - n
        left = True
       
    while len(data_pick) < n:
        index = random.randrange(0, size)
        elem = data[index]
        data[index] = data[size-1]
        data_pick.append(elem)
        size = size - 1
    if left:
        return [i for i in data if i not in data_pick]
    else:
        return data_pick
class SparseHopDict:
    """Sparse HopDict class

    A hopping dictionary contains relative hoppings.
    
    Attributes
    ----------
    dict : list of dictionaries
        dictionaries containing hoppings
    """

    def __init__(self, n_orb):
        """Initialize hop_dict object
        """
        self.dict = [{} for i in range(n_orb)]

    def set_element(self, rel_unit_cell, element, hop):
        """Add single hopping to hopping matrix.
        
        Parameters
        ----------
        rel_unit_cell : 3-tuple of integers
            relative unit cell coordinates
        element : 2-tuple of integers
            element indices
        hop : complex float
            hopping value
        """
        self.dict[element[0]][rel_unit_cell + (element[1],)] = hop
    
    def add_conjugates(self):
        """Adds hopping conjugates to self.dict."""
        
        # declare new dict
        self.new_dict = copy.deepcopy(self.dict)
        
        # iterate over items
        for i in range(len(self.dict)):
            for rel_tag, hopping in self.dict[i].items():
                x, y, z, j = rel_tag
                reverse_tag = (-x, -y, -z, i)
                reverse_hopping = np.conjugate(np.transpose(hopping))
                if reverse_tag not in self.new_dict[j]:
                    self.new_dict[j][reverse_tag] = reverse_hopping
                
        # done
        self.dict = self.new_dict
        
    def sparse(self, nr_processes=1):

        #self.add_conjugates()
        return self.dict
    
class Penrose(object):
    def __init__(self, tiling, model=1):
        self.tiling = tiling 
        self.model = model 
        self.Eside = -1.0
        self.rescale_energy = 5.0
        if model == 2:
            self.Efat = -0.85
            self.Ethin = -1.618
            self.rescale_energy = 7.0

    def _site_set(self, vac_con=0.):
        time0 = time.time()

        try:
            sites = self.tiling.vertices
        except:
            sites = self.tiling.vertexes
        if len(sites[0]) == 2:
            sites = np.append(sites, np.array([[0]]*len(sites)), axis=1)
        x_min, y_min, z_min = np.min(sites, axis=0)
        x_max, y_max, z_max = np.max(sites, axis=0)
        lat_vecs = [[x_max-x_min, 0., 0.],[0., y_max-y_min, 0.]]
        lattice = tipsi.Lattice(lat_vecs, sites)

        nsites = len(sites)
        nvac = int(nsites*vac_con)
        site_set = tipsi.SiteSet()
        vacancies = set()
        while len(vacancies) < nvac:
            orb = random.randrange(nsites)
            vacancies.add(((0, 0, 0), orb))

        for i in range(nsites):
            orbi = ((0,0,0),i)
            if orbi not in vacancies:
                site_set.add_site((0,0,0),i)
        time1 = time.time()
        print('Time for siteset: %s s' % (time1-time0))
        return tipsi.Sample(lattice, site_set)
   
    def _hop_dict(self, W=0., alpha=0.):
        """
        Description:
            W and alpha control the amplitude of diagonal disorder and off-diagonal disorder, respectively.
            For onsite energy: -W/2. < Eon_random < W/2.
            For hopping energy:  Ehop(1-alpha) < Ehop_random < Ehop(1+alpha).
            Any value in this range appears equivalentlly and the probability is 1/W and 1/2e*alpha respectively.
            **The whole procedure is implemented using numpy.random.uniform function.**
        Args:
            W: width for diagonal(Anderson) disorder
            alpha: the turn amplitude of hopping energy, that should be in (0,1)
        """
        time0 = time.time()

        try:
            nsite = len(self.tiling.vertexes)
        except:
            nsite = len(self.tiling.vertices)
        hop_dict = SparseHopDict(nsite)

        sides = self.tiling.sides
        nsides = len(sides)
        Esides = np.random.uniform(self.Eside*(1-alpha), self.Eside*(1+alpha), nsides)
        for i in range(nsides):
            s = sides[i]
            hop_dict.set_element((0,0,0), (s[0],s[1]), Esides[i])
            hop_dict.set_element((0,0,0), (s[1],s[0]), Esides[i])

        if self.model == 2:
            thins = self.tiling.thin_diags
            fats = self.tiling.fat_diags
            nthins = len(thins)
            nfats = len(self.tiling.fat_diags)
            Ethins = np.random.uniform(self.Ethin*(1-alpha), self.Ethin*(1+alpha), nthins)
            Efats = np.random.uniform(self.Efat*(1-alpha), self.Efat*(1+alpha), nfats)
            for i in range(nthins):
                thin = thins[i]
                hop_dict.set_element((0,0,0), (thin[0],thin[1]), Ethins[i])
                hop_dict.set_element((0,0,0), (thin[1],thin[0]), Ethins[i])
            for i in range(nfats):
                fat = fats[i]
                hop_dict.set_element((0,0,0), (fat[0],fat[1]), Efats[i])
                hop_dict.set_element((0,0,0), (fat[1],fat[0]), Efats[i])

        Eons = np.random.uniform(-W/2., W/2., nsite)
        for i in range(nsite):
            hop_dict.set_element((0,0,0),(i,i), Eons[i])
        time1 = time.time()
        print('Time for hop_dict: %s s' % (time1-time0))
        return hop_dict

    @staticmethod
    def _save_sample(tipsi_sample, fname, **kws):
        stru = kws
        stru['site_x'] = tipsi_sample.site_x
        stru['site_y'] = tipsi_sample.site_y
        stru['hop'] = np.real(tipsi_sample.hop)
        stru['indices'] = tipsi_sample.indices
        stru['indptr'] = tipsi_sample.indptr
        with open(fname, 'w') as f:
            f.write(json.dumps(jsanitize(stru)))

    def disorder_free(self):
        tipsi_sample = self._site_set()
        hop_dict = self._hop_dict()
        tipsi_sample.add_hop_dict(hop_dict)
        tipsi_sample.rescale_H(self.rescale_energy)
        return tipsi_sample

    def vacancy_disorder(self, con, save_to='sample_vacancy_disorder.json'):
        tipsi_sample = self._site_set(con)
        hop_dict = self._hop_dict()
        tipsi_sample.add_hop_dict(hop_dict)
        tipsi_sample.rescale_H(self.rescale_energy)
        self._save_sample(tipsi_sample, save_to)
        return tipsi_sample

    def Anderson_disorder(self, W):
        tipsi_sample = self._site_set()
        hop_dict = self._hop_dict(W=W, alpha=0.)
        tipsi_sample.add_hop_dict(hop_dict)
        tipsi_sample.rescale_H(self.rescale_energy)
        return tipsi_sample

    def off_diag_disorder(self, alpha):
        tipsi_sample = self._site_set()
        hop_dict = self._hop_dict(W=0., alpha=alpha)
        tipsi_sample.add_hop_dict(hop_dict)
        tipsi_sample.rescale_H(self.rescale_energy)
        return tipsi_sample

    def hopping_disorder(self, hop_type, con, Enew=0.0, save_to='sample_hopping_disorder.json'):
        """
        change the existed hopping with concentration con
        Args:
            hop_type:  hopping type 'side', 'fat' or 'thin'
            con: hopping concentration for reset
            Enew: the new hopping energy
        """
        Enew = -Enew
        tipsi_sample = self._site_set()
        hop_dict = self._hop_dict()
        tipsi_sample.add_hop_dict(hop_dict)
        time0 = time.time()
        if hop_type == 'side':
            sides = self.tiling.sides
            nsides = len(sides)
            npick = int(con * nsides)
            pairs_pickup = random_nelms_in_list(sides, npick)

        if hop_type == 'thin':
            if self.model == 1:
                raise ValueError('No hopping along thin diag for model1')
            thin_diags = self.tiling.thin_diags
            nthins = len(thin_diags)
            npick = int(con * nthins)
            pairs_pickup = random_nelms_in_list(thin_diags, npick)

        if hop_type == 'fat':
            if self.model == 1:
                raise ValueError('No hopping along fat diag for model1')
            fat_diags = self.tiling.fat_diags
            nfats = len(fat_diags)
            npick = int(con * nfats)
            pairs_pickup = random_nelms_in_list(fat_diags, npick)

        for i in pairs_pickup:
            tipsi_sample.set_hopping(Enew,(0,0,0),(0,0,0),i[0],i[1])
            tipsi_sample.set_hopping(Enew,(0,0,0),(0,0,0),i[1],i[0])
        tipsi_sample.rescale_H(self.rescale_energy)
        time1 = time.time()
        print('Time for adding hopping disorder: %s s' % (time1-time0))
        self._save_sample(tipsi_sample, save_to)
        return tipsi_sample

def diag_hamilt(tipsi_sample, diag_func='zheev', eigvecs_save=True, save_to='eigen_values_vecs_zheev.json'):
    time0 = time.time()
    Hamiltn = np.array(tipsi_sample.Hk((0.,0.,0.))*tipsi_sample.rescale, order='F')
    if diag_func == 'zheev':
        val, vec, inf = zheev(Hamiltn)
    elif diag_func == 'zhpevx':
        from diag_hamilt_zhpevx import diag_hamilt_zhpevx
        val, vec, inf = diag_hamilt_zhpevx(Hamiltn)
    if inf>=1:
        raise ValueError('%s failed, info > 0' % diag_func)
    elif inf <=-1:
        raise ValueError('%s failed, info < 0' % diag_func)
    if eigvecs_save:
        eigen={'eigen_energies':val, 'eigen_states':vec}
    else:
        eigen={'eigen_energies':val}
    with open(save_to, 'w') as f:
        f.write(json.dumps(jsanitize(eigen)))
    time1 = time.time()
    print('Time for diag: %s s' % (time1-time0))

