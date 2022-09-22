from mmap import mmap
import sre_compile
import numpy as np

def simpson_integration(startval, midval, endval, dx):
    #simpson's rule definition
    return (startval + 4*midval + endval)*dx/6

def adaptive_integrator(fun, a, b, extra=None):
    #set up parameters
    tol = 1e-8 

    #need to calculate simpson's rule at least twice to check tolerance
    #so set up all the midpoints: (a-mid-b); (a-mid1-mid), (mid-mid2-b)
    mid = (a+b)/2
    mid1 = (a + mid)/2
    mid2 = (mid + b)/2

    #calculate initial required function values
    if extra == None:
        extra = {
            a:fun(a),
            b:fun(b),
            mid:fun(mid),
            mid1:fun(mid1),
            mid2:fun(mid2),
        }
    #calculate any new required function calls
    else:
        extra[mid1] = fun(mid1)
        extra[mid2] = fun(mid2)
        #all the other points are already known
    
    dx = np.abs(b-a)
    #want to stop integration before we get rounding error in the intervals
    if dx/2 < 1e-15:
        print('Integration stopped. dx too small - probably a singularity...')
        return None

    int1 = simpson_integration(extra[a], extra[mid], extra[b], dx)
    int2 = simpson_integration(extra[a], extra[mid1], extra[mid], dx/2) + simpson_integration(extra[mid], extra[mid2], extra[b], dx/2)
    if np.abs(int1 - int2) < tol:
        return int2
    else:
        return adaptive_integrator(fun, a, mid, extra) + adaptive_integrator(fun, mid, b, extra)

