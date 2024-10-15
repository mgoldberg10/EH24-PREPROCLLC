import numpy as np
import datetime
from datetime import date, timedelta
from scipy.spatial import KDTree
import xarray as xr


def MITprof_from_fields(prof_depth, prof_descr, prof_YYYYMMDD,
                prof_HHMMSS, prof_lon, prof_lat,
                prof_point, prof_T, prof_Tweight,
                prof_Terr, prof_S, prof_Sweight,
                prof_Serr, prof_interp_XC11, prof_interp_YC11,
                prof_interp_XCNINJ, prof_interp_YCNINJ, prof_interp_i,
                prof_interp_j, prof_interp_lon, prof_interp_lat):

    mitprof = MITprof(prof_depth, prof_descr, prof_YYYYMMDD,
                prof_HHMMSS, prof_lon, prof_lat,
                prof_point, prof_T, prof_Tweight,
                prof_Terr, prof_S, prof_Sweight,
                prof_Serr, prof_interp_XC11, prof_interp_YC11,
                prof_interp_XCNINJ, prof_interp_YCNINJ, prof_interp_i,
                prof_interp_j, prof_interp_lon, prof_interp_lat)
    return mitprof.assemble_dataset()

def MITprof_from_metadata(grid_ds,
                  start_date, end_date, freq,
                  ungridded_lat, ungridded_lon,
                  sNx, sNy, fld_data_dict=None, **args):

    # load grid dict
    xc = grid_ds.XC.values
    yc = grid_ds.YC.values
    prof_depth = abs(grid_ds.Z.values)
    nz = len(prof_depth)

    # get gridded lat/lon, prof_point
    latlon_dict = get_nearest_latlon(xc, yc, ungridded_lat, ungridded_lon, **args)
    prof_point = latlon_dict['prof_point']
    ns = len(prof_point)
   
    # get temporal dict
    time_dict, nt = get_time_dict(ns, start_date, end_date, freq)
    
    # fill data fields
    flds_dict, fld_strs, fld_exts = get_fields_dict(ns, nt, nz, fld_data_dict)

    # get tile data
    tile_dict = get_tile_dict(xc, yc, prof_point, sNx, sNy)

    # compile all dicts into prof_dict
    prof_dict = get_prof_dict(latlon_dict, time_dict, flds_dict, tile_dict)
    prof_dict['prof_depth'] = prof_depth

    return MITprof(fld_strs=fld_strs, fld_exts=fld_exts, **prof_dict).assemble_dataset()


class MITprof:
    def __init__(self,
            prof_depth=None, prof_descr=None, prof_YYYYMMDD=None,
            prof_HHMMSS=None, prof_lon=None, prof_lat=None,
            prof_point=None, prof_T=None, prof_Tweight=None,
            prof_Terr=None, prof_S=None, prof_Sweight=None,
            prof_Serr=None, prof_interp_XC11=None, prof_interp_YC11=None,
            prof_interp_XCNINJ=None, prof_interp_YCNINJ=None, prof_interp_i=None,
            prof_interp_j=None, prof_interp_lon=None, prof_interp_lat=None,
            prof_interp_weights=None, fld_strs=None, fld_exts=None,
        ):
    
        """
        Initialize the MITprof class.

        Parameters
        ----------
        prof_depth : numpy.array
            Depth values for each profile
        prof_descr : numpy.array
            Description for each profile
        prof_YYYYMMDD : numpy.array
            Date of the profile in YYYYMMDD format
        prof_HHMMSS : numpy.array
            Time of the profile in HHMMSS format
        prof_lon : numpy.array
            Longitude (ungridded) of the profile locations
        prof_lat : numpy.array
            Latitude (ungridded) of the profile locations
        prof_point : numpy.array
            Indices of the nearest grid points for each profile
        prof_T : numpy.array
            Temperature values for each profile
        prof_Tweight : numpy.array
            Weight values for temperature measurements
        prof_Terr : numpy.array
            Error values for temperature measurements
        prof_S : numpy.array
            Salinity values for each profile
        prof_Sweight : numpy.array
            Weight values for salinity measurements
        prof_Serr : numpy.array
            Error values for salinity measurements
        prof_interp_XC11 : numpy.array
            Gridded longitude lower-left corner index
        prof_interp_YC11 : numpy.array
            Gridded latitude lower-left corner index
        prof_interp_XCNINJ : numpy.array
            Gridded longitude upper-right corner index
        prof_interp_YCNINJ : numpy.array
            Gridded latitude upper-right corner index
        prof_interp_i : numpy.array
            Parallel partition index
        prof_interp_j : numpy.array
            Parallel partition index
        prof_interp_lon : numpy.array
            Gridded longitude values
        prof_interp_lat : numpy.array
            Gridded latitude values
        prof_interp_weights : numpy.array
            Interpolation weights
        fld_strs : numpy.array
            List of observed field names
        fld_exts : numpy.array
            List of observed field extensions
        """

        self.prof_depth=prof_depth
        self.prof_descr=prof_descr
        self.prof_YYYYMMDD=prof_YYYYMMDD
        self.prof_HHMMSS=prof_HHMMSS
        self.prof_lon=prof_lon
        self.prof_lat=prof_lat
        self.prof_point=prof_point
        self.prof_T=prof_T
        self.prof_Tweight=prof_Tweight
        self.prof_Terr=prof_Terr
        self.prof_S=prof_S
        self.prof_Sweight=prof_Sweight
        self.prof_Serr=prof_Serr
        self.prof_interp_XC11=prof_interp_XC11
        self.prof_interp_YC11=prof_interp_YC11
        self.prof_interp_XCNINJ=prof_interp_XCNINJ
        self.prof_interp_YCNINJ=prof_interp_YCNINJ
        self.prof_interp_i=prof_interp_i
        self.prof_interp_j=prof_interp_j
        self.prof_interp_lon=prof_interp_lon
        self.prof_interp_lat=prof_interp_lat
        self.prof_interp_weights=prof_interp_weights
        self.fld_strs=fld_strs
        self.fld_exts = fld_exts

    def assemble_dataset(self):
        """
        Assemble a dataset using the class attributes.

        Returns:
        - mitprof: xarray Dataset containing the profile data
        """
        mitprof = xr.Dataset(
                data_vars=dict(
                    prof_depth=(['iDEPTH'],self.prof_depth),
                    prof_descr=(['iPROF'],self.prof_descr),
                    prof_YYYYMMDD=(['iPROF'],self.prof_YYYYMMDD),
                    prof_HHMMSS=(['iPROF'],self.prof_HHMMSS),
                    prof_lon=(['iPROF'],self.prof_lon),
                    prof_lat=(['iPROF'],self.prof_lat),
                    prof_point=(['iPROF'],self.prof_point),
                    prof_interp_XC11=(['iPROF'],self.prof_interp_XC11),
                    prof_interp_YC11=(['iPROF'],self.prof_interp_YC11),
                    prof_interp_XCNINJ=(['iPROF'],self.prof_interp_XCNINJ),
                    prof_interp_YCNINJ=(['iPROF'],self.prof_interp_YCNINJ),
                    prof_interp_i=(['iPROF', 'iINTERP'],self.prof_interp_i[:, None]),
                    prof_interp_j=(['iPROF', 'iINTERP'],self.prof_interp_j[:, None]),
                    prof_interp_lon=(['iPROF', 'iINTERP'],self.prof_interp_lon[:, None]),
                    prof_interp_lat=(['iPROF', 'iINTERP'],self.prof_interp_lat[:, None]),
                    prof_interp_weights=(['iPROF', 'iINTERP'],self.prof_interp_weights[:, None]),
                )
        )

        # populate observation fields
        for fld in self.fld_strs:
            for fld_ext in self.fld_exts:
                fld_tmp = getattr(self, fld+fld_ext)
                mitprof[fld+fld_ext] = (['iPROF', 'iDEPTH'], fld_tmp)

        return mitprof


def get_nearest_latlon(xc, yc, prof_lat, prof_lon, get_unique=True, verbose=False):
    # get gridded/llc coordinates

    llc_coords = np.c_[yc.ravel(), xc.ravel()]
    sensor_coords = np.c_[prof_lat, prof_lon]

    kd_tree = KDTree(llc_coords)
    distance, nearest_grid_idx = kd_tree.query(sensor_coords, k=1)

    assert((nearest_grid_idx>np.prod(xc.shape)).sum()==0)

    prof_lat_out = prof_lat
    prof_lon_out = prof_lon

    nearest_grid_idx_uniq, index = np.unique(nearest_grid_idx, return_index=True)

    ns = len(prof_lat)
    diff_indices = np.setdiff1d(np.arange(ns), index)

    prof_interp_lat = yc.ravel()[nearest_grid_idx]
    prof_interp_lon = xc.ravel()[nearest_grid_idx]

    latlon_dict = dict()

    if get_unique:
        if len(nearest_grid_idx) > len(nearest_grid_idx_uniq):
            print("Warning: Mapping from ungridded to gridded was not one-to-one!")
            if verbose:

                for i, (lat0, lon0, lat1, lon1) in enumerate(zip(prof_lat, prof_lon,
                                    prof_interp_lat, prof_interp_lon)):
                    print("({:.2f},{:.2f}) -> ({:.2f},{:.2f})".format(lat0, lon0, lat1, lon1))
            print("Returning nearest UNIQUE (lat,lon) pairs")
            print("{} ungridded (lat,lon) pairs -> {} gridded (lat,lon) pairs".format(len(prof_lat),len(nearest_grid_idx_uniq)))
            print("Dropped indices")
            print(", ".join(map(str, diff_indices)))

            prof_lat_out = np.array(prof_lat)[index]
            prof_lon_out = np.array(prof_lon)[index]
            prof_interp_lat = prof_interp_lat[index]
            prof_interp_lon = prof_interp_lon[index]
            distance_uniq = distance[index]

        latlon_dict['prof_point']=nearest_grid_idx_uniq
    else:
        latlon_dict['prof_point']=nearest_grid_idx


    latlon_dict['prof_lat']=prof_lat_out
    latlon_dict['prof_lon']=prof_lon_out
    latlon_dict['prof_interp_lat']=prof_interp_lat
    latlon_dict['prof_interp_lon']=prof_interp_lon
    latlon_dict['prof_interp_weights']=np.ones_like(prof_lat_out)

    return latlon_dict

def get_time_dict(ns, start_date, end_date, freq='daily'):

    supported_freqs = ['daily', 'hourly']
    delta = end_date - start_date
    n_yyyymmdd = delta.days + 1

    if freq == 'hourly':
        # this assumes user wants data right on the hour
        prof_HHMMSS = np.arange(0,24e4,1e4).astype(int)
        fac_yyyymmdd = 24
    elif freq == 'daily':
        # this assumes user wants data on first hour of each day
        prof_HHMMSS = 0
        fac_yyyymmdd = 1
    else: 
        raise Exception('invalid frequency \'{}\'\nCurrently supported frequencies: {}'.\
                        format(freq, '\'' + '\', \''.join(supported_freqs) + '\''))

    
    prof_YYYYMMDD = np.zeros((n_yyyymmdd,))
    
    for i, day in enumerate(range(n_yyyymmdd)):
        curr_day = start_date + timedelta(days=i)
        yyyymmdd_str = curr_day.strftime('%Y%m%d')
        prof_YYYYMMDD[i] = int(yyyymmdd_str)

    prof_HHMMSS = np.tile(prof_HHMMSS, n_yyyymmdd)
    prof_YYYYMMDD = np.repeat(prof_YYYYMMDD, fac_yyyymmdd)

    if freq == 'hourly':
        # Calculate total number of hours
        total_hours = int((end_date - start_date).total_seconds() // 3600) + 1

        # Trim the arrays to the total number of hours
        prof_HHMMSS = prof_HHMMSS[:total_hours]
        prof_YYYYMMDD = prof_YYYYMMDD[:total_hours]

    nt = len(prof_HHMMSS)
    assert len(prof_HHMMSS)==len(prof_YYYYMMDD)

    time_dict=dict(zip(['prof_YYYYMMDD', 'prof_HHMMSS'], [prof_YYYYMMDD, prof_HHMMSS]))
    return time_dict, nt
    
def get_fields_dict(ns, nt, nz, fld_data_dict=None):
 
    fld_exts = ['', 'err', 'weight']

    if fld_data_dict is None:
        dummy_data = np.zeros((ns*nt, nz))
        dummy_data[:,0] = 1

        fld_strs = ['prof_T', 'prof_S']

        fld_data_keys = [fld+fld_ext for fld in fld_strs for fld_ext in fld_exts]
        fld_data_vals = [dummy_data] * len(fld_data_keys)
        fld_data_dict = dict(zip(fld_data_keys, fld_data_vals))

    return fld_data_dict, fld_strs, fld_exts

def get_prof_dict(latlon_dict, time_dict, flds_dict, tile_dict):

    ns = len(latlon_dict['prof_point'])
    nt = len(time_dict['prof_HHMMSS'])
    prof_dict = {}
    all_dicts = [latlon_dict, time_dict, flds_dict, tile_dict]
    for d in all_dicts:
        for k, v in d.items():
            if v.ndim == 1:  # 1-dimensional case
                if len(v) == nt:
                    v = np.tile(v, ns)
                elif len(v) == ns:
                    v = np.repeat(v, nt)
            elif v.ndim == 2:  # 2-dimensional case
                if v.shape[0] == ns:
                    v = np.repeat(v, nt, axis=0)
            prof_dict[k] = v

    prof_dict['prof_descr'] = get_prof_descr(prof_dict, nt, ns)
    return prof_dict

def get_tile_dict(xc, yc, prof_point, sNx, sNy):
    def make_empty_5f(xgrid):
        tmp = dict();
        for face in range(1, 6): tmp[face] = np.zeros_like(xgrid[face])
        return tmp

    # transform xc and yc from worldmap back to 5faces
    xc_5faces = patchface3D_wrld_to_5f(xc[np.newaxis, :, :])
    yc_5faces = patchface3D_wrld_to_5f(yc[np.newaxis, :, :])

    xgrid = xc_5faces.copy()
    ygrid = yc_5faces.copy()

    XC11 = make_empty_5f(xgrid)
    YC11 = make_empty_5f(xgrid)
    XCNINJ = make_empty_5f(xgrid)
    YCNINJ = make_empty_5f(xgrid)
    iTile = make_empty_5f(xgrid)
    jTile = make_empty_5f(xgrid)
    tileNo = make_empty_5f(xgrid)

    tileCount=0
    for iF in range(1, len(xgrid)+1):
        face_XC = xgrid[iF][0]
        face_YC = ygrid[iF][0]
        for ii in range(face_XC.shape[0] // sNx):
            for jj in range(face_XC.shape[1] // sNy):
                tileCount += 1
                tmp_i = slice(sNx * ii, sNx * (ii + 1))
                tmp_j = slice(sNy * jj, sNy * (jj + 1))
                tmp_XC = face_XC[tmp_i, tmp_j]
                tmp_YC = face_YC[tmp_i, tmp_j]
                XC11[iF][0][tmp_i, tmp_j] = tmp_XC[0, 0]
                YC11[iF][0][tmp_i, tmp_j] = tmp_YC[0, 0]
                XCNINJ[iF][0][tmp_i, tmp_j] = tmp_XC[-1, -1]
                YCNINJ[iF][0][tmp_i, tmp_j] = tmp_YC[-1, -1]
                iTile[iF][0][tmp_i, tmp_j] = np.outer(np.ones(sNx), np.arange(1, sNy + 1))
                jTile[iF][0][tmp_i, tmp_j] = np.outer(np.arange(1, sNx + 1), np.ones(sNy))
                tileNo[iF][0][tmp_i, tmp_j] = tileCount * np.ones((sNx, sNy))


    tile_keys = ['XC11', 'YC11', 'XCNINJ', 'YCNINJ', 'i', 'j']


    XC11 = patchface3D_5f_to_wrld(XC11)[0,:,:]
    YC11 = patchface3D_5f_to_wrld(YC11)[0,:,:]
    XCNINJ = patchface3D_5f_to_wrld(XCNINJ)[0,:,:]
    YCNINJ = patchface3D_5f_to_wrld(YCNINJ)[0,:,:]
    iTile = patchface3D_5f_to_wrld(iTile)[0,:,:]
    jTile = patchface3D_5f_to_wrld(jTile)[0,:,:]

    tile_vals = [XC11, YC11, XCNINJ, YCNINJ, iTile, jTile]

    tile_dict = dict()
    for key, val in zip(tile_keys, tile_vals):
        tile_dict['prof_interp_' + key] = val.ravel()[prof_point]
    return tile_dict

def patchface3D_5f_to_wrld(faces):
    # Extract dimensions from one of the faces
    nz, _, nx = faces[1].shape

    array_out = np.zeros((nz, 4*nx, 4*nx))

    # Face 1
    array_out[:, :3*nx, :nx] = faces[1]

    # Face 2
    array_out[:, :3*nx, nx:2*nx] = faces[2]

    # Face 4
    face4 = np.transpose(np.flip(faces[4], 2), (0,2,1))
    array_out[:, :3*nx, 2*nx:3*nx] = face4

    # Face 5
    face5 = np.transpose(np.flip(faces[5], 2), (0,2,1))
    array_out[:, :3*nx, 3*nx:4*nx] = face5

    # Face 3
    face3 = np.rot90(faces[3][0,:,:], 3)
    array_out[:, 3*nx:4*nx, :nx] = face3

    return array_out

def patchface3D_wrld_to_5f(array_in):

    nz, ny, nx = np.shape(array_in)
    nx = int(nx/4)
    faces = dict()

    # face 1
    faces[1] = array_in[:,:3*nx,:nx]

    # face 2
    faces[2] = array_in[:,:3*nx,nx:2*nx]

    # face 4
    face4 = array_in[:,:3*nx,2*nx:3*nx]
    faces[4] = sym_g_mod(face4, 5)

    # face 5
    face5 = array_in[:,:3*nx,3*nx:4*nx]
    faces[5] = sym_g_mod(face5, 5)

    # face 3
    face3 = array_in[:,3*nx:4*nx,:nx]
    faces[3] = sym_g_mod(face3, 7)

    return faces

def sym_g_mod(field_in, sym_in):
    field_out = field_in
    for icur in range(sym_in-4):
        field_out = np.flip(np.transpose(field_out, (0, 2, 1)), axis=2)

    return field_out

def compact2worldmap(fldin,nx,nz):
    #add a new dimension in case it's only 2d field:
    if nz == 1:
        fldin=fldin[np.newaxis, :, :]
    #defining a big face:
    a=np.zeros((nz,4*nx,4*nx))       #(50,270,360)
    #face1
    tmp=fldin[:,0:3*nx,0:nx]        #(50,270,90)
    a[:,0:3*nx,0:nx]=tmp
    #face2
    tmp=fldin[:,(3*nx):(6*nx),0:nx] #(50, 270,90)
    a[:,0:3*nx,nx:2*nx]=tmp
    #face3
    tmp=fldin[:,(6*nx):(7*nx),0:nx] #(50, 90, 90)
    tmp=np.transpose(tmp, (1,2,0))  #(90, 90, 50)
    ##syntax to rotate ccw:
    tmp1=list(zip(*tmp[::-1]))
    tmp1=np.asarray(tmp1)
    tmp1=np.transpose(tmp1,[2,0,1]) #(50, 90, 90)
    a[:,3*nx:4*nx,0:nx]=tmp1
    #face4
    tmp=np.reshape(fldin[:,7*nx:10*nx,0:nx],[nz,nx,3*nx]) #(50,90,270)
    tmp=np.transpose(tmp, (1,2,0))
    print(tmp.shape)                                      #(90,270,50)
    #syntax to rotate cw:
    tmp1=list(zip(*tmp))[::-1]      #type is <class 'list'>
    tmp1=np.asarray(tmp1)           #type <class 'numpy.ndarray'>, shape (270,90,50)
    tmp1=np.transpose(tmp1,[2,0,1]) #(50,270,90)
    a[:,0:3*nx,2*nx:3*nx]=tmp1
    #face5
    tmp=np.reshape(fldin[:,10*nx:13*nx,0:nx],[nz,nx,3*nx]) #(50,90,270)
    tmp=np.transpose(tmp, (1,2,0))                         #(90,270,50)
    tmp1=list(zip(*tmp))[::-1]      #type is <class 'zip'> --> <class 'list'>
    tmp1=np.asarray(tmp1)           #type <class 'numpy.ndarray'>, shape (270,90,50)
    tmp1=np.transpose(tmp1,[2,0,1]) #(50,270,90)
    a[:,0:3*nx,3*nx:4*nx]=tmp1
    return a


def get_prof_descr(prof_dict, nt, ns):
    prof_descr = []
    suffix = ['sensor000'] * nt * ns
#    suffix2 = [x+'_' for x in ''.join([str(x)*nt for x in range(ns)])]
    suffix2 = [str(x) + '_' for x in np.repeat(range(ns), nt)]
    suffix = np.char.add(suffix, suffix2)
            
    yyyymmdd_str = prof_dict['prof_YYYYMMDD'].astype(int).astype(str)
    yyyymmdd_str = np.char.add(yyyymmdd_str, np.array(['_'] * nt * ns))
    hhmmss_str = prof_dict['prof_HHMMSS'].astype(int).astype(str)
    date_array = np.char.add(yyyymmdd_str, hhmmss_str)
    
    prof_descr = np.char.add(suffix, date_array)
    return prof_descr


if __name__ == "__main__":
    ungridded_lon = [167.34443606, 167.82297157, 168.68691826,
                     169.27300269, 168.48230111, 168.37412213]
    ungridded_lat = [-20.91171539, -20.64166884, -20.01145874,
                     -19.5513735,  -18.66460884, -17.80332068]
    # example below doesnt work, need to load a ds 
    # improvement: take either direct fields [xc, yc, Z] or ds with those fields
    start_date = date(1992, 1, 1)
    end_date = date(1992, 1, 31)
    freq='hourly'
    (sNx, sNy) = (6, 6)
    mitprof =  MITprof_from_metadata(grid_ds,
                      start_date, end_date, freq,
                      ungridded_lat, ungridded_lon,
                      sNx, sNy, fld_data_dict=None)

    #mitprof.tonetcdf()
