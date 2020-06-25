from tiling_grid import PeriodicTiling
import numpy as np
ppt = PeriodicTiling()
ppt.make_tiling(3)
for i in ppt.grid_intersections.values():
    try:
        intecs = np.append(intecs, i, axis=0)
    except:
        intecs = np.array(i)

for grid_pair in ppt.intersections_out_fold:
    for uc in ppt.intersections_out_fold[grid_pair]:
        try:
            insects_out = np.append(insects_out, ppt.intersections_out_fold[grid_pair][uc], axis=0)
        except:
            insects_out = ppt.intersections_out[grid_pair][uc]


intecs = set([tuple(i) for  i in intecs])
intecs_out = set([tuple(i) for  i in insects_out])
for i in intecs_out:
    if i not in intecs:
        print(i)
