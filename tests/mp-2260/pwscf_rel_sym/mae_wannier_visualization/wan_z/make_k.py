import numpy as np

for x in np.linspace(0,1,9,endpoint=False):
    for y in np.linspace(0,1,9,endpoint=False):
        for z in np.linspace(0,1,6,endpoint=False):
            if x >= 0.5:
                x = x-1
            if y >= 0.5:
                y = y-1
            if z >= 0.5:
                z = z-1
            print("{:.10f}\t{:.10f}\t{:.10f}\t1.0".format(x,y,z))