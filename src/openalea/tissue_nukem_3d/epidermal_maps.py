import numpy as np

def nuclei_density_function(nuclei_positions,cell_radius,k=0.1):
    
    def density_func(x,y,z,return_potential=False):
        
        max_radius = cell_radius
        # max_radius = 0.

        points = np.array(nuclei_positions.values())


        if len((x+y+z).shape) == 0:
            cell_distances = np.power(np.power(x[np.newaxis] - points[:,0],2) +  np.power(y[np.newaxis] - points[:,1],2) + np.power(z[np.newaxis] - points[:,2],2),0.5)
        elif len((x+y+z).shape) == 1:
            cell_distances = np.power(np.power(x[np.newaxis] - points[:,0,np.newaxis],2) +  np.power(y[np.newaxis] - points[:,1,np.newaxis],2) + np.power(z[np.newaxis] - points[:,2,np.newaxis],2),0.5)
        elif len((x+y+z).shape) == 2:
            cell_distances = np.power(np.power(x[np.newaxis] - points[:,0,np.newaxis,np.newaxis],2) +  np.power(y[np.newaxis] - points[:,1,np.newaxis,np.newaxis],2) + np.power(z[np.newaxis] - points[:,2,np.newaxis,np.newaxis],2),0.5)
        elif len((x+y+z).shape) == 3:
            cell_distances = np.power(np.power(x[np.newaxis] - points[:,0,np.newaxis,np.newaxis,np.newaxis],2) +  np.power(y[np.newaxis] - points[:,1,np.newaxis,np.newaxis,np.newaxis],2) + np.power(z[np.newaxis] - points[:,2,np.newaxis,np.newaxis,np.newaxis],2),0.5)


        density_potential = 1./2. * (1. - np.tanh(k*(cell_distances - (cell_radius+max_radius)/2.)))

        if len(density_potential.shape)==1 and density_potential.shape[0]==1:
            density = density_potential.sum()
        else:
            density = density_potential.sum(axis=0)

        if return_potential:
            return density, density_potential
        else:
            return density
    return density_func


def compute_local_2d_signal(positions, points, signal_values, cell_radius=5.0, density_k=0.33):
    X = positions[:,0]
    Y = positions[:,1]
    projected_positions = dict(zip(range(len(positions)),np.transpose([X,Y,np.zeros_like(X)])))

    x = points[...,0]
    y = points[...,1]
    z = 0*points[...,0]

    potential = np.array([nuclei_density_function(dict([(p,projected_positions[p])]),cell_radius=cell_radius,k=density_k)(x,y,z) for p in xrange(len(positions))])
    if potential.ndim == 2:
        potential = np.transpose(potential)
    elif potential.ndim == 3:
        potential = np.transpose(potential,(1,2,0))
    density = np.sum(potential,axis=-1)
    membership = potential/density[...,np.newaxis]

    signal = np.sum(membership*signal_values[np.newaxis,:],axis=-1)

    return signal