import ecco_v4_py as e4p
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

def array2mask(xa,write_path = False,form = '>f4'):
    """
    Create a mask for emu using xarray dataarray

    Parameter:
    ----------
    xa: xr.DataArray
        The dataarray to be converted into emu mask
    write_path: False or string
        The path for saving mask file. If not provided, return the numpy array
    form: string
        The data format, use >f8 for tracer.
    """
    transformed = e4p.llc_tiles_to_compact(xa)
    mask = np.array(transformed).astype(form).ravel()
    if not write_path:
        return mask
    else:
        mask.tofile(write_path)

def quickplotmask(mask):
    to_plot = nat2globe(mask)
    a = plt.pcolormesh(to_plot)
    return a
    
def quickplotarray(array):
    mask = array2mask(array)
    return quickplotmask(mask)

def nat2globe(llc):
    """
    Reorder a native 1170-by-90 format array (llc) to a geographically
    contiguous global 360-by-360 array (glb) for visualization.
    
    Parameters:
    llc (np.ndarray): Input array of shape (1170, 90).
    
    Returns:
    np.ndarray: Output array of shape (360, 360).

    """
    # Get the size of the input array
    nx = llc.shape[1]

    # Calculate extended dimensions
    nx2 = nx * 2
    nx3 = nx * 3
    nx4 = nx * 4

    # Initialize the global array
    glb = np.zeros((nx4, nx4), dtype=np.float32)

    # Face 1
    glb[0:nx3, 0:nx] = llc[0:nx3, :]

    # Face 2
    ioff = nx
    glb[0:nx3, nx:nx2] = llc[nx3:nx3*2, :]

    # Face 3
    glb[nx3:, 0:nx] = np.rot90(llc[2*nx3:2*nx3+nx, :], k=3)

    # Face 4
    dum = np.zeros((nx, nx3), dtype=np.float32)
    dum[:, :] = llc[2*nx3+nx:3*nx3+nx, :].reshape(nx, nx3)
    glb[0:nx3, nx2:nx3] = np.rot90(dum, k=1)

    # Face 5
    dum[:, :] = llc[3*nx3+nx:, :].reshape(nx, nx3)
    glb[0:nx3, nx3:] = np.rot90(dum, k=1)

    return glb