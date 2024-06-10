import astra
import numpy as np

def create_sinogram(volume_data):

    vol_geom = astra.create_vol_geom(volume_data.shape)
    

    angles = np.linspace(0, np.pi, 180, False)
    proj_geom = astra.create_proj_geom('cone', 1.0, 1.0, volume_data.shape[0], volume_data.shape[1], angles)


    proj_id, sinogram = astra.create_sino3d_gpu(volume_data, proj_geom, vol_geom)

    astra.data3d.delete(proj_id)

    return sinogram

def reconstruct_volume(sinogram, proj_geom, vol_geom):


    rec_id = astra.data3d.create('-vol', vol_geom)


    cfg = astra.astra_dict('CGLS3D_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram


    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, 50) 

 
    rec = astra.data3d.get(rec_id)

    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)

    return rec

# volume_data = np.random.rand(128, 128, 128)  
# proj_geom = ... 
# vol_geom = ...   

# sinogram = create_sinogram(volume_data)
# reconstructed_volume = reconstruct_volume(sinogram, proj_geom, vol_geom)