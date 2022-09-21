from mmap import mmap
import sre_compile
import numpy as np

def simpson_integration(startval, midval, endval, dx):
    #simpson's rule definition
    return (startval + 4*midval + endval)*dx/6

def adaptive_integrator(fun, a, b, extra=None):
    #set up parameters
    tol = 1e-7 
    singularity = 1e9

    #need to calculate simpson's rule at least twice to check tolerance
    #so set up all the midpoints: (a-mid-b); (a-mid1-mid), (mid-mid2-b)
    mid = (a+b)/2
    mid1 = (a + mid)/2
    mid2 = (mid + b)/2

    #calculate initial required function values
    if extra == None:
        extra = {
            'numcalls':1,
            a:fun(a),
            b:fun(b),
            mid:fun(mid),
            mid1:fun(mid1),
            mid2:fun(mid2),
        }
    #calculate any new required function calls
    else:
        if extra['numcalls'] > singularity:
            print('Too many function calls. Probably a singularity...')
        else:
            extra['numcalls'] += 1
        extra[mid1] = fun(mid1)
        extra[mid2] = fun(mid2)
        #all the other points are already known

    int1 = simpson_integration(extra[a], extra[mid], extra[b], np.abs(b-a))
    int2 = simpson_integration(extra[a], extra[mid1], extra[mid], np.abs(mid-a)) + simpson_integration(extra[mid], extra[mid2], extra[b], np.abs(b-mid))
    if np.abs(int1 - int2) < tol:
        return int2
    else:
        return adaptive_integrator(fun, a, mid, extra) + adaptive_integrator(fun, mid, b, extra)

