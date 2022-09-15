import numpy as np

def centraldiff(fun, x, dx):
    #central derivative formula
    #generalise to arrays for x, dx:
    if hasattr(x, '__len__') or hasattr(dx, '__len__'):
        x = np.atleast_1d(x)
        dx = np.atleast_1d(dx)
        tmp = np.ones((len(x), len(dx)))
        xArr = x[:, np.newaxis]*tmp
        dxArr = dx*tmp
        XplusDx = xArr + dxArr
        XminusDx = xArr - dxArr
        #not sure if this works
        return np.squeeze( (fun(XplusDx) - fun(XminusDx)) / (2*dxArr) )
    else:
        return np.atleast_1d( (fun(x+dx) - fun(x-dx)) / (2*dx) )

def ndiff(fun, x, full=False):
    #get dx by trying a lot of points
    dx10 = 10.**np.arange(-16, 1, dtype=np.float64) #get order of magnitude
    deriv = centraldiff(fun, x, dx10)
    if len(deriv.shape) > 1:
        diffs = np.sum(np.abs(np.diff(deriv, axis=1)), axis=0)
    else:
        diffs = np.abs(np.diff(deriv))
    bestdx = dx10[np.nanargmin(diffs)+1]

    dxrefine = bestdx * 10.**np.linspace(-1, 1, 10) #refine previous guess.
    deriv = centraldiff(fun, x, dxrefine)
    if len(deriv.shape) > 1:
        diffs = np.sum(np.abs(np.diff(deriv, axis=1)), axis=0)
    else:
        diffs = np.abs(np.diff(deriv))
    dx = dxrefine[np.argmin(diffs)+1] #choose this one.

    deriv = centraldiff(fun, x, dx)

    if not full:
        return deriv
    else:
        vals = np.atleast_1d(x)
        R = np.zeros(vals.shape)
        for i in range(len(R)):
            #calculate derivatives around each point in x using central derivative formula
            deriv1 = deriv[i]
            deriv2 = centraldiff(fun, vals[i]+dx, dx)
            deriv3 = centraldiff(fun, vals[i]-dx, dx)

            #calculate second derivative at x (+/-) dx
            second1 = (deriv2 - deriv1)/(2*dx)
            second2 = (deriv1 - deriv3)/(2*dx)

            #calculate third derivative at x
            third = (second1 - second2)/(2*dx)
            #central derivative error formula (neglect errors in higher derivatives)
            R[i] = (third*(dx**2)/6) + vals[i]*1e-16 
        return deriv, dx, R


#example test
deriv, dx, err = ndiff(np.exp, np.linspace(-10, 10, 20), full=True)
truth = np.exp(np.linspace(-10, 10, 20))
trutherr = np.abs(truth - deriv)

print('dx', dx)
print()
print('derivative ... estimated error ... True error')
print()
for i in range(len(deriv)):
    print(deriv[i], '...', err[i], '...', trutherr[i])

