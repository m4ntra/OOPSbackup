## Module containing functions to draw standard geometries



def makeCircle(x,M,x0,z0,r,mat):
	## draw a circle in x with material properties mat. Updates x directly.
	#
	#@param	x	Design domain to draw on
	#@param	M	Model object that fits x
	#@param	x0 Center of circle in x-direction
	#@param	z0 Center of circle in z-direction
	#@param	r	radius of circle
	#@param	mat	Material (or material properties)
   X,Z = M.getElementCoordinates()
   idx = pl.sqrt((X-x0)**2+(Z-z0)**2) <r
   x[idx] = mat
   return


def makeSlab(x,M,zstart,zstop,mat):
	## draw a slab in x with material properties mat. Updates x directly.
	#
	#@param	x	Design domain to draw on
	#@param	M	Model object that fits x
	#@param	zstart	start z-coordinate of slab
	#@param	zstop		stop z-coordinate of slab
	#@param	mat	Material (or material properties)
   X,Z = M.getElementCoordinates()
   idx = ((Z-zstart)>=0) * ((Z-zstop)<=0)
   x[idx] = mat
   return
