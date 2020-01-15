import numpy as np
from fatiando.gridder import regular
from fatiando.utils import ang2vec, vec2ang
from fatiando.constants import CM, T2NT, G, SI2MGAL, SI2EOTVOS
from scipy.optimize import nnls

def kernelxx(x,y,z,xs,ys,zs):
    '''
    Calculate the second derivative in relation x of a function 1/r.

    input
    x,y,z : float - Cartesian coordinates (in m) of the i-th observation point
    xs,ys,zs : numpy arrays - Cartesian coordinates of equivalent sources

    output
    phi_xx : numpy array - Values with the second derivatives at one point.
    '''
    assert xs.size == ys.size and ys.size == zs.size and xs.size == zs.size, \
    'All arrays must have the same size'
    xa = x - xs
    ya = y - ys
    za = z - zs 
    r_sqr = xa**2 + ya**2 + za**2
    r = np.sqrt(r_sqr)
    r_5 = r*r*r*r*r
       
    phi_xx = (3*xa**2 - r_sqr)/r_5

    return phi_xx

def kernelxy(x,y,z,xs,ys,zs):
    '''
    Calculate the second derivative in relation x and y of a function 1/r.

    input
    x,y,z : float - Cartesian coordinates (in m) of the i-th observation point
    xs,ys,zs : numpy arrays - Cartesian coordinates of equivalent sources

    output
    phi_xy : numpy array - Values with the second derivatives.
    '''
    assert x.size == y.size and y.size == z.size, \
    'All arrays must have the same size'
    xa = x - xs
    ya = y - ys
    za = z - zs
    
    r_sqr = xa**2 + ya**2 + za**2
    r = np.sqrt(r_sqr)
    r_5 = r*r*r*r*r
   
    phi_xy = 3*xa*ya/r_5 

    return phi_xy

def kernelxz(x,y,z,xs,ys,zs):
    '''
    Calculate the second derivative in relation x and z of a function 1/r.

    input
    x,y,z : float - Cartesian coordinates (in m) of the i-th observation point
    xs,ys,zs : numpy arrays - Cartesian coordinates of equivalent sources

    output
    phi_xz : numpy array - Values with the second derivatives.
    '''
    assert x.size == y.size and y.size == z.size, \
    'All arrays must have the same size'

    xa = x - xs
    ya = y - ys
    za = z - zs 
    
    r_sqr = xa**2 + ya**2 + za**2
    r = np.sqrt(r_sqr)
    r_5 = r*r*r*r*r

    phi_xz = 3*xa*za/r_5

    return phi_xz

def kernelyy(x,y,z,xs,ys,zs):
    '''
    Calculate the second derivative in relation y of a function 1/r.

    input
    x,y,z : float - Cartesian coordinates (in m) of the i-th observation point
    xs,ys,zs : numpy arrays - Cartesian coordinates of equivalent sources

    output
    phi_yy : numpy array - Values with the second derivatives.
    '''
    assert x.size == y.size and y.size == z.size and x.size == z.size, \
    'All arrays must have the same size'
    
    xa = x - xs
    ya = y - ys
    za = z - zs 
    
    r_sqr = xa**2 + ya**2 + za**2
    r = np.sqrt(r_sqr)
    r_5 = r*r*r*r*r    

    phi_yy = (3*ya**2 - r_sqr)/r_5

    return phi_yy

def kernelyz(x,y,z,xs,ys,zs):
    '''
    Calculate the second derivative in relation y and z of a function 1/r.

    input
    x,y,z : float - Cartesian coordinates (in m) of the i-th observation point
    xs,ys,zs : numpy arrays - Cartesian coordinates of equivalent sources

    output
    phi_yz : numpy array - Values with the second derivatives.
    '''
    assert x.size == y.size and y.size == z.size, \
    'All arrays must have the same size'
    xa = x - xs
    ya = y - ys
    za = z - zs
    
    r_sqr = xa**2 + ya**2 + za**2
    r = np.sqrt(r_sqr)
    r_5 = r*r*r*r*r

    phi_yz = 3*ya*za/r_5

    return phi_yz

def kernelzz(x,y,z,xs,ys,zs):
    '''
    Calculate the second derivative in relation z of a function 1/r.

    input
    x,y,z : float - Cartesian coordinates (in m) of the i-th observation point
    xs,ys,zs : numpy arrays - Cartesian coordinates of equivalent sources

    output
    phi_zz : numpy array - Values with the second derivatives at one point.
    '''
    assert xs.size == ys.size and ys.size == zs.size and xs.size == zs.size, \
    'All arrays must have the same size'
    xa = x - xs
    ya = y - ys
    za = z - zs
    
    r_sqr = xa**2 + ya**2 + za**2
    r = np.sqrt(r_sqr)
    r_5 = r*r*r*r*r

    phi_zz = (3*za**2 - r_sqr)/r_5

    return phi_zz

def bz_layer(x,y,z,xs,ys,zs,p,inc,dec):
    '''
    Calculate the bz produced by a layer

    input

    x,y,z : float - Cartesian coordinates (in m) of the i-th observation point
    xs,ys,zs : numpy arrays - Cartesian coordinates of equivalent sources
    p: numpy array - Vector composed by magnetic moment and magnetization direction of the
                     equivalent sources

    return
    bz : numpy array - bz of the equivalent layer
    '''
    N = x.size
    
    bz = np.empty(N,dtype=float)

    mx,my,mz = ang2vec(1.,inc,dec)

    for i in range(N):
        phi_xz = kernelxz(x[i],y[i],z[i],xs,ys,zs)
        phi_yz = kernelyz(x[i],y[i],z[i],xs,ys,zs)
        phi_zz = kernelzz(x[i],y[i],z[i],xs,ys,zs)

        gi = phi_xz*mx + phi_yz*my + phi_zz*mz

        bz[i] = np.dot(p.T,gi)

    bz *= CM*T2NT
    return bz

def bx_layer(x,y,z,xs,ys,zs,p,inc,dec):
    '''
    Calculate the bx produced by a layer

    input

    x,y,z : float - Cartesian coordinates (in m) of the i-th observation point
    xs,ys,zs : numpy arrays - Cartesian coordinates of equivalent sources
    p: numpy array - Vector composed by magnetic moment and magnetization direction of the
                     equivalent sources

    return
    bx : numpy array - bz of the equivalent layer
    '''
    N = x.size
    
    bx = np.empty(N,dtype=float)

    mx,my,mz = ang2vec(1.,inc,dec)

    for i in range(N):
        phi_xx = kernelxx(x[i],y[i],z[i],xs,ys,zs)
        phi_xy = kernelxy(x[i],y[i],z[i],xs,ys,zs)
        phi_xz = kernelxz(x[i],y[i],z[i],xs,ys,zs)
        
        gi = phi_xx*mx + phi_xy*my + phi_xz*mz

        bx[i] = np.dot(p.T,gi)

    bx *= CM*T2NT
    return bx

def by_layer(x,y,z,xs,ys,zs,p,inc,dec):
    '''
    Calculate the by produced by a layer

    input

    x,y,z : float - Cartesian coordinates (in m) of the i-th observation point
    xs,ys,zs : numpy arrays - Cartesian coordinates of equivalent sources
    p: numpy array - Vector composed by magnetic moment and magnetization direction of the
                     equivalent sources

    return
    by : numpy array - bz of the equivalent layer
    '''
    N = x.size
    
    by = np.empty(N,dtype=float)

    mx,my,mz = ang2vec(1.,inc,dec)

    for i in range(N):
        phi_xy = kernelxy(x[i],y[i],z[i],xs,ys,zs)
        phi_yy = kernelyy(x[i],y[i],z[i],xs,ys,zs)
        phi_yz = kernelyz(x[i],y[i],z[i],xs,ys,zs)
        
        gi = phi_xy*mx + phi_yy*my + phi_yz*mz

        by[i] = np.dot(p.T,gi)

    by *= CM*T2NT
    return by


def sensitivity_bz(x,y,z,xs,ys,zs,inc,dec):
    '''
    Calculate the sensitivity matrix

    input

    return
    '''
    N = x.size # number of data
    M = xs.size # number of parameters

    A = np.empty((N,M)) # sensitivity matrix

    mx,my,mz = ang2vec(1.,inc,dec) # magnetization direction in Cartesian coordinates

    for i in range(N):
        phi_xz = kernelxz(x[i],y[i],z[i],xs,ys,zs)
        phi_yz = kernelyz(x[i],y[i],z[i],xs,ys,zs)
        phi_zz = kernelzz(x[i],y[i],z[i],xs,ys,zs)        
        gi = phi_xz*mx + phi_yz*my + phi_zz*mz

        A[i,:] = gi

    A *= CM*T2NT
    return A

def sensitivity_by(x,y,z,xs,ys,zs,inc,dec):
    '''
    Calculate the sensitivity matrix

    input

    return
    '''
    N = x.size # number of data
    M = xs.size # number of parameters

    A = np.empty((N,M)) # sensitivity matrix

    mx,my,mz = ang2vec(1.,inc,dec) # magnetization direction in Cartesian coordinates

    for i in range(N):
        phi_xy = kernelxy(x[i],y[i],z[i],xs,ys,zs)
        phi_yy = kernelyy(x[i],y[i],z[i],xs,ys,zs)
        phi_yz = kernelyz(x[i],y[i],z[i],xs,ys,zs)        
        gi = phi_xy*mx + phi_yy*my + phi_yz*mz

        A[i,:] = gi

    A *= CM*T2NT
    return A

def sensitivity_bx(x,y,z,xs,ys,zs,inc,dec):
    '''
    Calculate the sensitivity matrix

    input

    return
    '''
    N = x.size # number of data
    M = xs.size # number of parameters

    A = np.empty((N,M)) # sensitivity matrix

    mx,my,mz = ang2vec(1.,inc,dec) # magnetization direction in Cartesian coordinates

    for i in range(N):
        phi_xx = kernelxx(x[i],y[i],z[i],xs,ys,zs)
        phi_xy = kernelxy(x[i],y[i],z[i],xs,ys,zs)
        phi_xz = kernelxz(x[i],y[i],z[i],xs,ys,zs)        
        gi = phi_xx*mx + phi_xy*my + phi_xz*mz

        A[i,:] = gi

    A *= CM*T2NT
    return A

 
