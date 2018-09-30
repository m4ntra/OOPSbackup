from __future__ import division,print_function

import pylab as pl
from pylab import pi,sin,cos,exp

from scipy.sparse import coo_matrix


def assembleMvv(A,B,M,P):
	#A	?
	#B	?
	#M Mesh object
	#P Physics object

	nodes = 4
	sa0,ma0 = localStiffness(M)

	etabflat = M.etab.reshape(M.nelx*M.nelz,nodes)
	Aflat = A.flatten()[pl.newaxis].T
	Bflat = B.flatten()[pl.newaxis].T

	inew = pl.kron(etabflat,pl.ones((1,nodes))).flatten()
	jnew = pl.kron(etabflat,pl.ones((nodes,1))).flatten()
	dnew = pl.dot(Aflat,sa0.reshape((1,nodes*nodes))).flatten() -\
				P.k0**2*pl.dot(Bflat,ma0.reshape((1,nodes*nodes))).flatten()
	i = list(inew)
	j = list(jnew)
	d = list(dnew)

	#Periodic Bloch-Floquet BC:
	pen = 1e8
	n1 = list(range(0,(M.nelz+1)*(M.nelx+1),M.nelx+1))
	n2 = list(range(M.nelx,(M.nelz+1)*(M.nelx+1)+M.nelx,M.nelx+1))
	i += n1+n2+n1+n2
	j += n1+n2+n2+n1
	d += [pen]*2*len(n1)+\
				[-pen*exp(1.j*P.kInx*M.lx)]*len(n1)+\
				[-pen*exp(1.j*P.kInx*M.lx).conj()]*len(n1)

	#Calculate the matrices given in Eq (43) in Dossou2006. See Fuchi2010 for a "nicer" way
	#	of writing them. The elements used are the same as in Andreassen2011
	#Mvv
	Mvv = coo_matrix((d,(i,j)),shape=(M.ndof,M.ndof)).tocsc()
	return Mvv


#P contains info on material at boundaries
def assembleMxxfx(M,P):
	#Instead of solving Eq (43) in Dossou2006 directly, it is faster to solve the reduced systems
	#	shown in Eq (47-52) in Dossou2006. This is what is done below. These matrices are not banded
	#  and therefore treated as dense matrices

	#Precalculate integrals for contstructing the matrices:
	iFDown, jFDown, dFDown = [], [], []
	iFUp  , jFUp  , dFUp   = [], [], []
	for m in range(-M.Nm,M.Nm+1):
		idx = m+M.Nm
		elx = pl.array(range(M.nelx))
		x1 = elx/M.nelx*M.lx
		x2 = (elx+1)/M.nelx*M.lx
		Fvec = FourierInt(x1,x2,P.alpha[idx])

		#Downstream modes (... or have I switched what's up and down)
		elzDown = 0	
		edofDown = M.etab[elx,elzDown]
		FvecDown = Fvec.copy()
		FvecDown[1:3,:] = 0.
		iFDown += list(edofDown.flatten())
		jFDown += [idx]*edofDown.size
		dFDown += list(FvecDown.T.flatten())

		#Upstream modes
		elzUp = M.nelz-1 
		edofUp = M.etab[elx,elzUp]
		FvecUp = Fvec.copy()
		FvecUp[0,:] = 0.
		FvecUp[3,:] = 0.
		iFUp += list(edofUp.flatten())
		jFUp += [idx]*edofUp.size
		dFUp += list(FvecUp.T.flatten())

	FFDown =	coo_matrix( (dFDown,(iFDown,jFDown)), shape=(M.ndof,M.NM),dtype='complex').toarray()
	FFUp   =	coo_matrix( (dFUp  ,(iFUp  ,jFUp  )), shape=(M.ndof,M.NM),dtype='complex').toarray()

	Mrv = -FFDown.T
	Mvr = 1.j*P.chiIn*P.AIn*FFDown.conj()
	fvn = pl.zeros((M.ndof,1),dtype='complex')
	fvn[:,0] = 1.j*P.chiIn[M.Nm]*P.AIn*FFDown[:,M.Nm].conj()

	Mtv = -FFUp.T
	Mvt = 1.j*P.chiOut*P.AOut*FFUp.conj()
	frn = pl.zeros((M.NM,1),dtype='complex')

	return Mrv.view(pl.matrix),Mvr.view(pl.matrix), \
			 Mtv.view(pl.matrix),Mvt.view(pl.matrix), \
			 fvn,frn


def localStiffness(M):
	#Jacobians (?) in x and z
	xJac = M.elmsize/2
	zJac = M.elmsize/2

	#Temporary variables
	k1 = (zJac**2+xJac**2)/(xJac*zJac)
	k2 = (zJac**2-2*xJac**2)/(xJac*zJac)
	k3 = (xJac**2-2*zJac**2)/(xJac*zJac)

	sa0 = 1/3*pl.array([
		[ k1  , k2/2,-k1/2, k3/2],
		[ k2/2, k1  , k3/2,-k1/2],
		[-k1/2, k3/2, k1  , k2/2],
		[ k3/2,-k1/2, k2/2, k1  ]])

	ma0 = xJac*zJac/9*pl.array([
		[ 4, 2, 1, 2],
		[ 2, 4, 2, 1],
		[ 1, 2, 4, 2],
		[ 2, 1, 2, 4]])
	return sa0,ma0


def FourierInt(x1,x2,alpha):
	xs = (x2-x1)/2
	xd = (x2+x1)/2
	N = 1 if type(xs) is float else xs.size

	Fvec = pl.zeros((4,N),dtype='complex')
	if abs(alpha)<1e-6:
		Fvec[0,:] = xs
		Fvec[1,:] = xs
		Fvec[2,:] = xs
		Fvec[3,:] = xs
	else:
		Fvec[0,:] = -(-2.j*alpha*xs*exp(-1.j*alpha*(-xd+xs)) - exp(-1.j*alpha*(-xd+xs)) 
						+exp(1.j*alpha*(xd+xs))) /(alpha**2 *2*xs)
		Fvec[1,:] = -(-2.j*alpha*xs*exp(-1.j*alpha*(-xd+xs)) - exp(-1.j*alpha*(-xd+xs)) 
						+exp(1.j*alpha*(xd+xs))) /(alpha**2 *2*xs)
		Fvec[2,:] = -( 2.j*alpha*xs*exp( 1.j*alpha*( xd+xs)) + exp(-1.j*alpha*(-xd+xs)) 
						-exp(1.j*alpha*(xd+xs))) /(alpha**2 *2*xs)
		Fvec[3,:] = -( 2.j*alpha*xs*exp( 1.j*alpha*( xd+xs)) + exp(-1.j*alpha*(-xd+xs)) 
						-exp(1.j*alpha*(xd+xs))) /(alpha**2 *2*xs)
	return Fvec


if __name__ == "__main__":
	pass
