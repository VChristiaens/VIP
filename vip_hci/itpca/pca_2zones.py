#! /usr/bin/env python

"""
Full-frame PCA algorithm for ADI, ADI+RDI and ADI+mSDI (IFS data) cubes.
"""

__author__ = 'Valentin Christiaens'
__all__ = ['pca_2regimes_it']

import inspect
import numpy as np
from scipy.interpolate import interp2d
import os
from os.path import isdir, isfile
from .pca_local import pca_annular_it, _prepare_matrix_ann
from .pca_fullfr import pca_it
from ..conf import time_ini, timing
from ..fits import write_fits, open_fits
from ..preproc import cube_derotate, cube_crop_frames
from ..pca import pca, pca_annular
from ..metrics import (cube_inject_companions, frame_inject_companion, 
                       contrast_curve)
from ..specfit import find_nearest
from ..var import (prepare_matrix, cart_to_pol, frame_center, 
                   get_annulus_segments)


def pca_2regimes_it(cube, angle_list, cube_ref=None, fwhm=4, r_lim=3, buffer=1, 
                    strategy='ADI', ncomp_range=range(1,11), n_it_max=20, 
                    thr=1., n_neigh=0, add_res=False, thru_corr=True, n_br=6, 
                    psfn=None, starphot=1., plsc=1., svd_mode='lapack', 
                    scaling=None, delta_rot=1, mask_center_px=None, n_segments=1, 
                    imlib='opencv', imlib2='opencv', interpolation='lanczos4', 
                    collapse='median', mask_rdi=None, check_memory=True, 
                    nproc=1, full_output=False, verbose=True, weights=None, 
                    debug=False, path='', overwrite=True):
    """
    Iterative version of PCA, combining the annular and full-frame pca for two
    different regimes/zones defined by 'r_lim'. A final image is made by 
    combining the sections of images obtained with reduction parameters 
    achieving the highest contrast (number of pc subtracted, number of 
    iterations and rotation threshold for PCA library construction). 
    
    This method is relevant when the innermost stellar residuals vary on a 
    different timescale and amplitude (e.g. leakage from a mask) than the
    outer speckle pattern (quasi-static).
    
    See complete description of the algorithm in Christiaens et al. (2021b):
    *** INSERT LINK ***    
    See also description of 'pca_annular_it' and 'pca_it' to know how each zone 
    is dealt with.
    
    Note: The iterative PCA can only be used in ADI, RDI or mSDI+ADI 
    (adimsdi='double') modes.

    Parameters
    ----------
    cube : str or numpy ndarray, 3d or 4d
        Input cube (ADI or ADI+mSDI). If a string is given, it must correspond
        to the path to the fits file to be opened in memmap mode (for PCA
        incremental of ADI 3d cubes).
    angle_list : numpy ndarray, 1d
        Corresponding parallactic angle for each frame.
    cube_ref : numpy ndarray, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    scale_list : numpy ndarray, 1d, optional
        Scaling factors in case of IFS data (ADI+mSDI cube). Usually, the
        scaling factors are the central channel wavelength divided by the
        shortest wavelength in the cube (more thorough approaches can be used
        to get the scaling factors). This scaling factors are used to re-scale
        the spectral channels and align the speckles.
    fwhm : float, optional
        Known size of the FHWM in pixels to be used. Default value is 4.
    r_lim: float, optional
        Radius in terms of FWHM for the transition between the 2 regimes (i.e
        outer radius of annular PCA, and inner radius of full frame PCA).
    buffer: float, option
        Buffer in terms of FWHM used to smooth the transition. In practice, the
        annular (resp. full-frame) PCA will be performed several times, with an 
        outer radius (resp. inner radius) ranging between (r_lim-buffer/2)*FWHM
        and (r_lim+buffer/2)*FWHM, per steps of 1 px.
    strategy: str, opt
        Whether to do iterative ADI only ('ADI'), iterative RDI only ('RDI'), 
        or iterative RDI and iterative ADI consecutivey ('RADI'). A reference 
        cube needs to be provided for 'RDI' and 'RADI'.
    ncomp_range : list or tuple of int/None/float, optional
        Test list of number of PC values used as a lower-dimensional subspace 
        to project the target frames. The format of each test ncomp in the list 
        should be compatible with the pca.pca() function, that is:

        * ADI (``cube`` is a 3d array): if an int is provided, ``ncomp`` is the
          number of PCs extracted from ``cube`` itself. If ``ncomp`` is a float
          in the interval (0, 1] then it corresponds to the desired cumulative
          explained variance ratio (the corresponding number of components is
          estimated). If ``ncomp`` is a tuple of two integers, then it
          corresponds to an interval of PCs in which final residual frames are
          computed (optionally, if a tuple of 3 integers is passed, the third
          value is the step). When``source_xy`` is not None, then the S/Ns
          (mean value in a 1xFWHM circular aperture) of the given
          (X,Y) coordinates are computed.

        * ADI+RDI (``cube`` and ``cube_ref`` are 3d arrays): ``ncomp`` is the
          number of PCs obtained from ``cube_ref``. If ``ncomp`` is a tuple,
          then it corresponds to an interval of PCs (obtained from ``cube_ref``)
          in which final residual frames are computed. If ``source_xy`` is not
          None, then the S/Ns (mean value in a 1xFWHM circular aperture) of the
          given (X,Y) coordinates are computed.

    n_it_max: int, opt
        Maximum number of iterations for the iterative PCA. Note that if no 
        improvement is found from one iteration to the next, the algorithm 
        will stop automatically.
    thr: float, tuple of floats or tuple or tuples/lists, opt
        Threshold used to (iteratively) identify significant signals in the 
        final PCA image obtained in each regime. If a float, the same value 
        will be used for both regimes. If a tuple, can either contain 1) two 
        floats corresponding to the resp. thresholds to be used for inner and 
        outer regions; or ii) two lists/tuples of float values which will be 
        tested. This threshold is a minimum pixel intensity in the STIM map 
        computed with the PCA residual cube (Pairet et al. 2019), as expressed 
        in units of max. pixel intensity obtained in the inverse STIM map. 
        Recommended value: 1. But for discs with bright axisymmetric signals, a 
        lower value may be necessary. 
    n_neigh: int, opt
        If larger than 0, number of neighbouring pixels to the ones included
        as significant to be included in the mask. A larger than zero value can
        make the convergence faster but also bears the risk of including 
        non-significant signals.
    thru_corr: bool, opt
        Whether to correct the significant signals by the algorithmic 
        throughput before subtraction to the original cube at the next
        iteration. 
        Deprecated: If None, this is not performed. If 'psf', throughput is 
        estimated in a classical way with injected psfs. If 'map', the map of
        identified significant signals is used directly for an estimate of the
        2D throughput.
    n_br: int, opt
        Number of branches on which the fake planets are injected to compute 
        the throughput.
    psfn: 2d numpy array, opt
        If thru_corr is set to True, psfn should be a normalised and centered 
        unsaturated psf.
    starphot: float, opt
        Integrated flux of the star in a 1FWHM aperture. Only relevant for
        accurate output contrast curves in full_output mode.
    plsc: float, opt
        Plate scale in arcsec/px. Only relevant for accurate output contrast 
        curves in full_output mode.
    svd_mode : {'lapack', 'arpack', 'eigen', 'randsvd', 'cupy', 'eigencupy',
        'randcupy', 'pytorch', 'eigenpytorch', 'randpytorch'}, str optional
        Switch for the SVD method/library to be used.

        ``lapack``: uses the LAPACK linear algebra library through Numpy
        and it is the most conventional way of computing the SVD
        (deterministic result computed on CPU).

        ``arpack``: uses the ARPACK Fortran libraries accessible through
        Scipy (computation on CPU).

        ``eigen``: computes the singular vectors through the
        eigendecomposition of the covariance M.M' (computation on CPU).

        ``randsvd``: uses the randomized_svd algorithm implemented in
        Sklearn (computation on CPU).

        ``cupy``: uses the Cupy library for GPU computation of the SVD as in
        the LAPACK version. `

        `eigencupy``: offers the same method as with the ``eigen`` option
        but on GPU (through Cupy).

        ``randcupy``: is an adaptation of the randomized_svd algorithm,
        where all the computations are done on a GPU (through Cupy). `

        `pytorch``: uses the Pytorch library for GPU computation of the SVD.

        ``eigenpytorch``: offers the same method as with the ``eigen``
        option but on GPU (through Pytorch).

        ``randpytorch``: is an adaptation of the randomized_svd algorithm,
        where all the linear algebra computations are done on a GPU
        (through Pytorch).

    scaling : {None, "temp-mean", spat-mean", "temp-standard",
        "spat-standard"}, None or str optional
        Pixel-wise scaling mode using ``sklearn.preprocessing.scale``
        function. If set to None, the input matrix is left untouched. Otherwise:

        ``temp-mean``: temporal px-wise mean is subtracted.

        ``spat-mean``: spatial mean is subtracted.

        ``temp-standard``: temporal mean centering plus scaling pixel values
        to unit variance. HIGHLY RECOMMENDED FOR ASDI AND RDI CASES!

        ``spat-standard``: spatial mean centering plus scaling pixel values
        to unit variance.

    mask_center_px : None or int
        If None, no masking is done. If an integer > 1 then this value is the
        radius of the circular mask. If buffer is larger than 0, the effective 
        mask_center_px will be increased accordingly with the outer radius of 
        the annulus, for the different reductions used for edge smoothing.
        
    source_xy : tuple of int, optional
        For ADI-PCA, this triggers a frame rejection in the PCA library, with
        ``source_xy`` as the coordinates X,Y of the center of the annulus where
        the PA criterion is estimated. When ``ncomp`` is a tuple, a PCA grid is
        computed and the S/Ns (mean value in a 1xFWHM circular aperture) of the
        given (X,Y) coordinates are computed.
    delta_rot : int/float, tuple of int/float, or tuple of 2 lists, optional
        Factor for tunning the parallactic angle threshold, expressed in FWHM.
        Default is 1 (excludes 1xFHWM on each side of the considered frame).
        If int/float: same value used for inner and outer zones.
        If tuple of int/floats: two values used for inner and outer zones.
        If tuple of 2 lists: values to be tested for the inner and outer zones.
    adimsdi : {'single', 'double'}, str optional
        Changes the way the 4d cubes (ADI+mSDI) are processed. Basically it
        determines whether a single or double pass PCA is going to be computed.

        ``single``: the multi-spectral frames are rescaled wrt the largest
        wavelength to align the speckles and all the frames (n_channels *
        n_adiframes) are processed with a single PCA low-rank approximation.

        ``double``: a first stage is run on the rescaled spectral frames, and a
        second PCA frame is run on the residuals in an ADI fashion.
    crop_ifs: bool, optional
        [adimsdi='single'] If True cube is cropped at the moment of frame
        rescaling in wavelength. This is recommended for large FOVs such as the
        one of SPHERE, but can remove significant amount of information close to
        the edge of small FOVs (e.g. SINFONI).
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    imlib2 : str, optional
        See the documentation of the ``vip_hci.preproc.cube_rescaling_wavelengths`` 
        function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    ifs_collapse_range: str 'all' or tuple of 2 int
        If a tuple, it should contain the first and last channels where the mSDI 
        residual channels will be collapsed (by default collapses all channels).
    check_memory : bool, optional
        If True, it checks that the input cube is smaller than the available
        system memory.
    nproc : None or int, optional
        Number of processes for parallel computing. If None the number of
        processes will be set to (cpu_count()/2). Defaults to ``nproc=1``.
    full_output: bool, optional
        Whether to return the final median combined image only or with other
        intermediate arrays.
    verbose : bool, optional
        If True prints intermediate info and timing.
    weights: 1d numpy array or list, optional
        Weights to be applied for a weighted mean. Need to be provided if 
        collapse mode is 'wmean'.   
    debug: bool, opt
        Whether to save intermediate fits files in directory given by path.
    path: str, opt
        Where to save intermediate fits files if debug is True. 

    Returns
    -------
    master_frame: 2d numpy ndarray
        2D array obtained from combining images obtained at optimal contrast 
        at each radius.
    master_cube: 3d numpy ndarray
        [full_output=True] Master cube of all nanmedian-combined images used 
        to produce master_frame.
    drot_opt_arr: 1d numpy ndarray
        [full_output=True] Array containing the delta_rot values used to 
        produce each image of master_cube. 
    npc_opt_arr: 1d numpy ndarray
        [full_output=True] Array containing the optimal npc value used to 
        produce each image of master_cube. 
    nit_opt_arr: 1d numpy ndarray
        [full_output=True] Array containing the optimal number of iterations 
        used to produce each image of master_cube. 
    cc_ss_opt_arr: 1d numpy ndarray
        [full_output=True] Array containing the optimal contrast value achieved 
        at each radius (used to select optimal images), when significant signal
        is subtracted from the cube.
    cc_ws_opt_arr: 1d numpy ndarray
        [full_output=True] Array containing the optimal contrast value achieved 
        at each radius (used to select optimal images), when significant signal
        is NOT subtracted from the cube.    
    cc_rad_arr: 1d numpy ndarray
        [full_output=True] Array containing the radii where the contrasts are 
        achieved (used to select optimal images).       
    """
    
    if buffer/2.>=int(r_lim):
        raise TypeError("buffer should be smaller than half r_lim") 
    if not isinstance(ncomp_range, (list,tuple)):
        raise TypeError("ncomp_range can only be list or tuple.")
    if not isinstance(thr, tuple):
        thrs = ([thr],[thr])
    elif len(thr) != 2:
        raise TypeError("If a tuple, thr must have a length of 2.")
    elif not isinstance(thr[0], (tuple,list)):
        thrs = ([thr[0]],[thr[1]])
    else:
        thrs=thr
    if mask_center_px is None:
        mask_inner_sz = 0
    else:
        mask_inner_sz = mask_center_px
    if isinstance(delta_rot, (int, float)):
        delta_rot_test = ([delta_rot],[delta_rot])
    elif isinstance(delta_rot, (tuple, list)):
        if len(delta_rot) != 2:
            raise TypeError("If a tuple/list, delta_rot must have 2 elements")
        if isinstance(delta_rot[0], (int, float)):
            delta_rot_test = ([delta_rot[0]],[delta_rot[1]])
        elif not isinstance(delta_rot[0], (tuple, list)):
            msg = "If delta_rot is a tuple/list, its elements can only be "
            msg+= "tuple, list, int or float"
            raise TypeError(msg)
        else:
            delta_rot_test=delta_rot
    else:
        raise TypeError("Format of delta_rot not int, float, list or tuple")
    if thru_corr and psfn is None:
        msg = "psf should be provided for throughput correction or convolution"
        raise TypeError(msg)
    
    if strategy == 'ADI':
        ref_cube = None
    elif strategy == 'RDI' or strategy == 'RADI':
        if cube_ref is None:
            raise ValueError("cube_ref should be provided for RDI or RADI")
        ref_cube = cube_ref.copy() 
    else:
        raise ValueError("strategy not recognized: should be ADI, RDI or RADI")
     
    if path:
        if not isdir(path):
            os.makedirs(path)
     
    nframes = cube.shape[0]
    buff = max(int(buffer*fwhm),1)
    asize = int(r_lim*fwhm-mask_inner_sz)
    if asize<fwhm:
        msg = "asize too small: increase r_lim ({:.1f} FWHM) or decrease "
        msg+= "mask_center_px {:.0f} px"
        raise ValueError(msg.format(r_lim,mask_inner_sz))
    master_crop_cube = []
    
    # prepare labels for intermediate files saved
    if debug:
        if n_neigh==0:
            neigh_lab=''
        else:
            neigh_lab='_{:.0f}neigh'.format(n_neigh)
        fn_tmp = "TMP_PCA-{}{}_{:.1f}rlim_{:.0f}bb_{:.0f}nit_{:.1f}thr{}"
        fn_tmp += "_{:.1f}drot_{:.0f}npc"
    
    ## loop over buffer inner/outer rads
    drot_opt = []
    thr_opt = []
    npc_opt = []
    nit_opt = []
    cc_opt = []
    cc_opt_ws = []
    rad_opt = []
        
    # loop on inner and outer parts
    for io in range(2):
        # 1. it. PCA in annulus
        # 2. it. PCA in full frame
        for bb in range(buff):
            if io == 0:
                mask_sz = mask_inner_sz+bb
                opt_rads = [[mask_sz+0.5*fwhm, mask_sz+asize-0.5*fwhm]]
                lab_io = '_ann'
            else:
                mask_sz = mask_inner_sz+bb+asize
                lab_io = '_full'
                cy, cx = frame_center(cube[0])
                source_xy_tmp = (cx, cy+mask_sz+fwhm)
                opt_rads = [[mask_sz+0.5*fwhm, mask_sz+1.5*fwhm]]
                rr = 0.5
                while opt_rads[-1][-1]<cy-2*fwhm:
                    rr+=1
                    opt_rads.append([mask_sz+rr*fwhm,mask_sz+(rr+1)*fwhm])
            n_rads = len(opt_rads)
            cube_tmp = cube.copy()
            if ref_cube is None:
                ref_cube_tmp = None
            else:
                ref_cube_tmp = ref_cube.copy()
            if io == 0:
                ## crop the cube to just larger than annulus
                crop_sz = max((mask_sz+asize)*2+1,int(9*fwhm))
                # note: 8 fwhm required to avoid bug in contrast curve function
                if not crop_sz%2:
                    crop_sz+=1
                if crop_sz < cube.shape[-1]:
                    cube_tmp = cube_crop_frames(cube, crop_sz)
                    if ref_cube is not None:
                        ref_cube_tmp = cube_crop_frames(ref_cube, crop_sz)
                elif crop_sz > cube.shape[-1]:
                    msg = "Input cube dimensions too small for annulus"
                    raise ValueError(msg)
                
            ## loop over delta_rot_test
            cc_bb = []
            cc_bb_ws = []
            bb_drot = []
            bb_npc = []
            bb_thr = []
            bb_it = []
            bb_frames = []
            for delta_rot_tmp in delta_rot_test[io]:
                ## loop over pcs
                tmp_imgs = np.zeros([len(ncomp_range), cube_tmp.shape[-2], 
                                     cube_tmp.shape[-1]])
                for tt, thr_t in enumerate(thrs[io]):
                    fn = "TMP_PCA-{}{}_{:.1f}rlim_{:.0f}bb_{:.0f}nit"
                    fn+="_{:.1f}thr{}_{:.1f}drot_imgs.fits"
                    filename_tmp = path+fn.format(strategy, lab_io, r_lim, bb, 
                                                  n_it_max, thr_t, neigh_lab,
                                                  delta_rot_tmp)

                    for pp, ncomp in enumerate(ncomp_range):
                        if debug:
                            fn = path+fn_tmp.format(strategy, lab_io, r_lim, bb, 
                                                    n_it_max, thr_t, neigh_lab, 
                                                    delta_rot_tmp, ncomp)
                        else:
                            fn=''
                        if not debug or not isfile(fn+"_norm_stim_cube.fits") or overwrite: # change below
                            if io==0:
                                ## it. PCA-ann with full_output
                                res = pca_annular_it(cube_tmp, angle_list, 
                                                     n_it=n_it_max, thr=thr_t, 
                                                     cube_ref=ref_cube_tmp, 
                                                     thru_corr=thru_corr, 
                                                     n_neigh=n_neigh, psfn=psfn, 
                                                     strategy=strategy, 
                                                     n_br=n_br, fwhm=fwhm, 
                                                     radius_int=mask_sz, 
                                                     asize=asize, ncomp=ncomp, 
                                                     delta_rot=delta_rot_tmp, 
                                                     interpolation=interpolation,
                                                     imlib=imlib, 
                                                     full_output=True,
                                                     interp_order=None,
                                                     add_res=add_res)
                                it_cube_nd = res[-4]
                                norm_stim_cube = res[-3]
                                stim_cube = res[-2]
                                inv_stim_cube = res[-1]
                            else:
                                ## it. PCA-full frame with full_output
                                res = pca_it(cube_tmp, angle_list, n_it=n_it_max,
                                             cube_ref=ref_cube_tmp, fwhm=fwhm, 
                                             thru_corr=thru_corr, mask_rdi=mask_rdi, 
                                             mask_center_px=mask_sz, psfn=psfn,  
                                             source_xy=source_xy_tmp, thr=thr_t,
                                             delta_rot=delta_rot_tmp, ncomp=ncomp, 
                                             n_neigh=n_neigh, scaling=scaling,
                                             imlib=imlib, imlib2=imlib2, 
                                             interpolation=interpolation,
                                             full_output=True, add_res=add_res)
                                norm_stim_cube = res[-2]
                                it_cube_nd = res[-1]
                            tmp_imgs[pp] = res[0]
                            it_cube = res[1]
                            sig_cube = res[2]
                            if debug:
                                fn = path+fn_tmp.format(strategy, lab_io, r_lim, bb, 
                                                        n_it_max, thr_t, neigh_lab, 
                                                        delta_rot_tmp, ncomp)
                                write_fits(fn+"_it_cube.fits", it_cube)
                                write_fits(fn+"_sig_cube.fits", sig_cube)
                                write_fits(fn+"_norm_stim_cube.fits", 
                                           norm_stim_cube)
                                write_fits(fn+"_it_cube_nd.fits", it_cube_nd)
                                if io==0:
                                    write_fits(fn+"_stim_cube.fits", stim_cube)
                                    write_fits(fn+"_inv_stim_cube.fits", 
                                               inv_stim_cube)
                        else:
                            it_cube = open_fits(fn+"_it_cube.fits")
                            sig_cube = open_fits(fn+"_sig_cube.fits")
                               
                        ## calculate contrast curve (after scaling cubes if needed)
                        if io == 0:
                            res = _prepare_matrix_ann(cube_tmp, ref_cube_tmp, 
                                                      scaling, angle_list, fwhm, 
                                                      mask_sz, asize, delta_rot, 
                                                      n_segments)
                        else:
                            res = _prepare_matrix_full(cube_tmp, ref_cube_tmp, 
                                                       scaling, mask_sz, 
                                                       'fullfr', False)
                        cube_tmp_scal, ref_cube_scal = res
                        for ii, n_it in enumerate(range(n_it_max)):
                            if n_it==0:
                                # no disc subtraction
                                sig_cube_rot = np.zeros_like(cube_tmp_scal)
                            else:
                                sig_image = sig_cube[ii].copy()
                                if np.all(sig_image==sig_cube[ii-1]):
                                    break
                                sig_cube_rot = np.repeat(sig_image[np.newaxis, :, :], 
                                                         nframes, axis=0)
                                sig_cube_rot = cube_derotate(sig_cube_rot, 
                                                             -angle_list, 
                                                             nproc=1, 
                                                             border_mode='constant')
                            if io == 0:
                                cc = contrast_curve(cube_tmp_scal-sig_cube_rot, 
                                                    angle_list, pxscale=plsc, 
                                                    psf_template=psfn, 
                                                    fwhm=fwhm, starphot=starphot, 
                                                    plot=False, nbranch=n_br, 
                                                    algo=pca_annular, 
                                                    inner_rad=1, verbose=False, 
                                                    radius_int=mask_sz,  
                                                    delta_rot=delta_rot_tmp, 
                                                    interp_order=None, 
                                                    scaling=None, ncomp=ncomp,
                                                    imlib=imlib, smooth=False,
                                                    cube_ref=ref_cube_scal, 
                                                    interpolation=interpolation)   
                            else:
                                cc = contrast_curve(cube_tmp_scal-sig_cube_rot, 
                                                    angle_list, pxscale=plsc, 
                                                    psf_template=psfn, 
                                                    fwhm=fwhm, plot=False, 
                                                    starphot=starphot, 
                                                    algo=pca, nbranch=n_br, 
                                                    inner_rad=1, verbose=False, 
                                                    mask_center_px=mask_sz, 
                                                    source_xy=source_xy_tmp, 
                                                    ncomp=ncomp, scaling=None, 
                                                    delta_rot=delta_rot_tmp, 
                                                    cube_ref=ref_cube_scal,
                                                    mask_rdi=mask_rdi, 
                                                    imlib=imlib, imlib2=imlib2, 
                                                    interpolation=interpolation)  
                                
                            rad_vec = np.array(cc['distance'])
                            cc_tmp = np.array(cc['sensitivity_student'])
                            cc_mean = []
                            for rr in range(n_rads):
                                ir0 = find_nearest(rad_vec, opt_rads[rr][0])
                                ir1 = find_nearest(rad_vec, opt_rads[rr][1])
                                cc_mean.append(float(np.mean(cc_tmp[ir0:ir1+1])))
                            cc_bb.append(cc_mean)
                            if ii==0:
                                cc_mean_0 = cc_mean.copy()
                            cc_bb_ws.append(cc_mean_0)
                            bb_drot.append(delta_rot_tmp)
                            bb_npc.append(ncomp)
                            bb_thr.append(thr_t)
                            bb_it.append(n_it)
                            bb_frames.append(it_cube[ii])
                    ### save if debug
                    if debug and not isfile(filename_tmp):
                        fn = "TMP_PCA-{}{}_{:.1f}rlim_{:.0f}bb_{:.0f}nit"
                        fn+="_{:.1f}thr{}_{:.1f}drot_imgs.fits"
                        write_fits(path+fn.format(strategy, lab_io, r_lim, bb, 
                                                  n_it_max, thr_t, neigh_lab,
                                                  delta_rot_tmp),
                                   tmp_imgs)
            idx_opt = np.argmin(np.array(cc_bb), axis=1)
            for rr in range(n_rads):
                drot_opt.append(bb_drot[idx_opt[rr]])
                thr_opt.append(bb_thr[idx_opt[rr]])
                npc_opt.append(bb_npc[idx_opt[rr]])
                nit_opt.append(bb_it[idx_opt[rr]])
                cc_opt.append(cc_bb[idx_opt[rr]][rr])
                cc_opt_ws.append(cc_bb_ws[idx_opt[rr]][rr])
                rad_opt.append(np.mean(np.array(opt_rads[rr])))
                ## append to master_list frame obtained with optimal drot, pc and it.
                irad = opt_rads[rr][0]-0.5*fwhm+1
                width = opt_rads[rr][1]-opt_rads[rr][0]+fwhm-2
                good_ann_fr = get_annulus_segments(bb_frames[idx_opt[rr]],
                                                   inner_radius=irad,
                                                   width=width, 
                                                   mode='mask')[0]
                master_crop_cube.append(good_ann_fr)

    # 3. produce master final frame with np.nanmedian()
    ## place in a master cube with same dimensions as original cube
    ncombi = len(master_crop_cube)
    master_cube = np.zeros([ncombi, cube.shape[-2], cube.shape[-1]])
    cy, cx = frame_center(master_cube[0])
    for mm in range(ncombi):
        cy_tmp, cx_tmp = frame_center(master_crop_cube[mm])
        idx_y0 = int(cy-cy_tmp)
        idx_yN = int(cy+cy_tmp+1)
        idx_x0 = int(cx-cx_tmp)
        idx_xN = int(cx+cx_tmp+1)
        master_cube[mm,idx_y0:idx_yN,idx_x0:idx_xN] = master_crop_cube[mm]
    master_cube[np.where(master_cube==0)] = np.nan   
    master_frame= np.nanmedian(master_cube,axis=0)

    if full_output:
        drot_opt_arr = np.array(drot_opt)
        thr_opt_arr = np.array(thr_opt)
        npc_opt_arr = np.array(npc_opt)
        nit_opt_arr= np.array(nit_opt)
        cc_ss_opt_arr = np.array(cc_opt)
        cc_ws_opt_arr = np.array(cc_opt_ws)
        cc_rad_arr = np.array(rad_opt)
        return (master_frame, master_cube, drot_opt_arr, thr_opt_arr, 
                npc_opt_arr, nit_opt_arr, cc_ss_opt_arr, cc_ws_opt_arr, 
                cc_rad_arr)
    else:
        return master_frame
    

def _interp2d_rad(thruput_arr, radii, frame_sz, theta_0=0):
    """
    Converts 1d throughput curve into a 2d image.
    
    Parameters
    ----------
    thruput_arr : 2d numpy array
        Matrix of estimated throughput as a function of azimuth and radius.
    radii : 1d numpy array
        Vector of radii corresponding to the throughput
    frame_sz : int
        Size of the 2d image to be returned
    theta_0: float, opt
        Polar (trigonometric) angle of the first branch used for fake planet
        injections.

    Returns
    -------
    thru_2d : 2d numpy array
        2D throughput image.

    """
    n_br = thruput_arr.shape[0]
    # median over 3 azimuthally shifted thruput maps to smooth 2pi transition
    thru_2d_list = []
    for ii in range(4):
        thetas = np.linspace(theta_0, theta_0+360, n_br, endpoint=False)
        sh=ii*90
        sh_idx = np.where(thetas>theta_0+360-sh) 
        thetas[sh_idx] = thetas[sh_idx]-360
        
        f_thru = interp2d(radii, thetas, thruput_arr)
        
        yy, xx = np.ogrid[:frame_sz, :frame_sz]  
        cy, cx = (frame_sz-1)/2, (frame_sz-1)/2
        r, theta = cart_to_pol(xx, yy, cx, cy)
        theta[np.where(theta<np.amin(thetas))]+=360
        theta[np.where(theta>np.amax(thetas))]-=360
    
        thru_2d = np.zeros([frame_sz,frame_sz])
        for i in range(frame_sz):
            for j in range(frame_sz):
                thru_2d[i,j] = f_thru(r[i,j], theta[i,j])
        thru_2d[np.where(thru_2d<=0)] = 1
        thru_2d[np.where(thru_2d>1)] = 1
        thru_2d_list.append(thru_2d)
    
    final_thru_2d = np.median(np.array(thru_2d_list),axis=0)
    
    return final_thru_2d


def throughput_ext(cube, angle_list, sig_image, fwhm, pxscale, algo, #nbranch=1, 
                   #theta=0, inner_rad=1, fc_rad_sep=3, wedge=(0,360),  
                   imlib='opencv', interpolation='lanczos4', verbose=True, 
                   full_output=False, **algo_dict):
    """ Measures the throughput for chosen algorithm and input dataset (ADI or
    ADI+mSDI). The final throughput is the average of the same procedure
    measured in ``nbranch`` azimutally equidistant branches.

    Parameters
    ---------_
    cube : numpy ndarray
        The input cube, 3d (ADI data) or 4d array (IFS data), without fake
        companions.
    angle_list : numpy ndarray
        Vector with the parallactic angles.
    sig_image : numpy ndarray
        Frame containing the identified significant signal, with the rest of 
        the image masked to 0.
    fwhm: int or float or 1d array, optional
        The the Full Width Half Maximum in pixels. It can handle a different
        FWHM value for different wavelengths (IFS data).
    pxscale : float
        Plate scale in arcsec/px.
    algo : callable or function
        The post-processing algorithm, e.g. vip_hci.pca.pca. Third party Python
        algorithms can be plugged here. They must have the parameters: 'cube',
        'angle_list' and 'verbose'. Optionally a wrapper function can be used.
    #nbranch : int optional
    #    Number of branches on which to inject fakes companions. Each branch
    #    is tested individually.
    # theta : float, optional
    #     Angle in degrees for rotating the position of the first branch that by
    #     default is located at zero degrees. Theta counts counterclockwise from
    #     the positive x axis.
    inner_rad : int, optional
        Innermost radial distance to be considered in terms of FWHM.
    # fc_rad_sep : int optional
    #     Radial separation between the injected companions (in each of the
    #     patterns) in FWHM. Must be large enough to avoid overlapping. With the
    #     maximum possible value, a single fake companion will be injected per
    #     cube and algorithm post-processing (which greatly affects computation
    #     time).
    wedge : tuple of floats, optional
        Initial and Final angles for using a wedge. For example (-90,90) only
        considers the right side of an image.
    full_output : bool, optional
        If True returns intermediate arrays.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    verbose : bool, optional
        If True prints out timing and information.
    **algo_dict
        Parameters of the post-processing algorithms must be passed here.

    Returns
    -------
    thruput : numpy ndarray
        2d array containing which ratio of flux is preserved after 
        post-processing for each original significant intensity pixels.

    If full_output is True then the function returns: thruput, frame_fc and 
    frame_nofc.
    frame_fc : numpy ndarray
        2d array with the PCA processed frame with significant signal.
    frame_nofc : numpy ndarray
        2d array, PCA processed frame without significant signal.

    """
    nbranch=1
    theta=0
    array = cube
    parangles = angle_list

    if array.ndim != 3 and array.ndim != 4:
        raise TypeError('The input array is not a 3d or 4d cube')
    else:
        if array.ndim == 3:
            if array.shape[0] != parangles.shape[0]:
                msg = 'Input parallactic angles vector has wrong length'
                raise TypeError(msg)
            if sig_image.ndim != 2:
                raise TypeError('Template PSF is not a frame or 2d array')
            #maxfcsep = int((array.shape[1]/2.)/fwhm)-1
            # if fc_rad_sep < 3 or fc_rad_sep > maxfcsep:
            #     msg = 'Too large separation between companions in the radial '
            #     msg += 'patterns. Should lie between 3 and {}'
            #     raise ValueError(msg.format(maxfcsep))

        elif array.ndim == 4:
            if array.shape[1] != parangles.shape[0]:
                msg = 'Input vector or parallactic angles has wrong length'
                raise TypeError(msg)
            if sig_image.ndim != 3:
                raise TypeError('Template PSF is not a frame, 3d array')
            if 'scale_list' not in algo_dict:
                raise ValueError('Vector of wavelength not found')
            else:
                if algo_dict['scale_list'].shape[0] != array.shape[0]:
                    raise TypeError('Input wavelength vector has wrong length')
                # if isinstance(fwhm, float) or isinstance(fwhm, int):
                #     maxfcsep = int((array.shape[2] / 2.) / fwhm) - 1
                # else:
                #     maxfcsep = int((array.shape[2] / 2.) / np.amin(fwhm)) - 1
                # if fc_rad_sep < 3 or fc_rad_sep > maxfcsep:
                #     msg = 'Too large separation between companions in the '
                #     msg += 'radial patterns. Should lie between 3 and {}'
                #     raise ValueError(msg.format(maxfcsep))

        if sig_image.shape[1] % 2 == 0:
            raise ValueError("Only odd-sized PSF is accepted")
        if not hasattr(algo, '__call__'):
            raise TypeError('Parameter `algo` must be a callable function')
        # if not isinstance(inner_rad, int):
        #     raise TypeError('inner_rad must be an integer')
        #angular_range = wedge[1] - wedge[0]
        # if nbranch > 1 and angular_range < 360:
        #     msg = 'Only a single branch is allowed when working on a wedge'
        #     raise RuntimeError(msg)

    if isinstance(fwhm, (np.ndarray,list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if verbose:
        start_time = time_ini()
    #***************************************************************************
    # Compute noise in concentric annuli on the "empty frame"
    argl = inspect.getargspec(algo).args
    if 'cube' in argl and 'angle_list' in argl and 'verbose' in argl:
        if 'fwhm' in argl:
            frame_nofc = algo(cube=array, angle_list=parangles, fwhm=fwhm_med,
                              verbose=False, **algo_dict)
            # if algo_dict.pop('scaling',None):
            #     new_algo_dict = algo_dict.copy()
            #     new_algo_dict['scaling'] = None
            #     frame_nofc_noscal = algo(cube=array, angle_list=parangles, 
            #                              fwhm=fwhm_med, verbose=False, 
            #                              **new_algo_dict)
            # else:
            #     frame_nofc_noscal = frame_nofc
        else:
            frame_nofc = algo(array, angle_list=parangles, verbose=False,
                              **algo_dict)
            # if algo_dict.pop('scaling',None):
            #     new_algo_dict = algo_dict.copy()
            #     new_algo_dict['scaling'] = None
            #     frame_nofc_noscal = algo(cube=array, angle_list=parangles,
            #                              verbose=False, **new_algo_dict)
            # else:
            #     frame_nofc_noscal = frame_nofc
                
    if verbose:
        msg1 = 'Cube without fake companions processed with {}'
        print(msg1.format(algo.__name__))
        timing(start_time)

    # noise, res_level, vector_radd = noise_per_annulus(frame_nofc, 
    #                                                   separation=fwhm_med,
    #                                                   fwhm=fwhm_med, 
    #                                                   wedge=wedge)
    # noise_noscal, _, _ = noise_per_annulus(frame_nofc_noscal, 
    #                                        separation=fwhm_med, fwhm=fwhm_med, 
    #                                        wedge=wedge)                                       
    # vector_radd = vector_radd[inner_rad-1:]
    # noise = noise[inner_rad-1:]
    # res_level = res_level[inner_rad-1:]
    # noise_noscal = noise_noscal[inner_rad-1:]
    if verbose:
        print('Measured annulus-wise noise in resulting frame')
        timing(start_time)

    # We crop the PSF and check if PSF has been normalized (so that flux in
    # 1*FWHM aperture = 1) and fix if needed
    new_psf_size = int(round(3 * fwhm_med))
    if new_psf_size % 2 == 0:
        new_psf_size += 1

    if cube.ndim == 3:
        n, y, x = array.shape
        psf_template = sig_image.copy() 
        #normalize_psf(psf_template, fwhm=fwhm, verbose=verbose,
        #size=min(new_psf_size,
        #psf_template.shape[1]))

        # Initialize the fake companions
        #angle_branch = angular_range / nbranch
        #thruput_arr = np.zeros((nbranch, sig_image.shape[0]))
        cy, cx = frame_center(array[0])

        # each branch is computed separately
        for br in range(nbranch):
            # each pattern is computed separately. For each one the companions
            # are separated by "fc_rad_sep * fwhm", interleaving the injections
            #for irad in range(fc_rad_sep):
            radvec = np.array([0])#vector_radd[irad::fc_rad_sep]
            cube_fc = array.copy()
            # filling map with small numbers
            fc_map = np.ones_like(array[0]) * 1e-6
            fcy = []
            fcx = []
            for i in range(radvec.shape[0]):
                flux = 1 #fc_snr * noise_noscal[irad + i * fc_rad_sep]
                cube_fc = cube_inject_companions(cube_fc, psf_template,
                                                 parangles, flux, pxscale,
                                                 rad_dists=[radvec[i]],
                                                 theta=#br*angle_branch +
                                                       theta,
                                                 imlib=imlib, verbose=False,
                                                 interpolation=
                                                    interpolation)
                y = cy + radvec[i] * np.sin(np.deg2rad(#br * angle_branch +
                                                       theta))
                x = cx + radvec[i] * np.cos(np.deg2rad(#br * angle_branch +
                                                       theta))
                fc_map = frame_inject_companion(fc_map, psf_template, y, x,
                                                flux, imlib, interpolation)
                fcy.append(y)
                fcx.append(x)

            if verbose:
                msg2 = 'Fake sig image injected in branch {} '
                #msg2 += '(pattern {}/{})'
                print(msg2.format(br+1))
                timing(start_time)

            #***************************************************************
            arg = inspect.getargspec(algo).args
            if 'cube' in arg and 'angle_list' in arg and 'verbose' in arg:
                if 'fwhm' in arg:
                    frame_fc = algo(cube=cube_fc, angle_list=parangles,
                                    fwhm=fwhm_med, verbose=False, **algo_dict)
                else:
                    frame_fc = algo(cube=cube_fc, angle_list=parangles,
                                    verbose=False, **algo_dict)
            else:
                msg = 'Input algorithm must have at least 3 parameters: '
                msg += 'cube, angle_list and verbose'
                raise ValueError(msg)

            if verbose:
                msg3 = 'Cube with fake companions processed with {}'
                msg3 += '\nMeasuring its annulus-wise throughput'
                print(msg3.format(algo.__name__))
                timing(start_time)

            #***************************************************************
            injected_flux = fc_map
            recovered_flux = frame_fc - frame_nofc
            thruput = np.ones_like(injected_flux)
            num = recovered_flux[np.where(injected_flux > 0)]
            denom = injected_flux[np.where(injected_flux > 0)]
            thruput[np.where(injected_flux > 0)] = num/denom
            thruput[np.where(thruput <= 0)] = 1
            thruput[np.where(thruput > 1)] = 1
            #thruput_arr[br] = thruput

    elif cube.ndim == 4:
        w, n, y, x = array.shape
        if isinstance(fwhm, (int, float)):
            fwhm = [fwhm] * w
        psf_template = sig_image.copy()
                       #normalize_psf(psf_template, fwhm=fwhm, verbose=verbose,
                       #              size=min(new_psf_size,
                       #                       psf_template.shape[1]))

        # Initialize the fake companions
        #angle_branch = angular_range / nbranch
        #thruput_arr = np.zeros((nbranch, sig_image.shape))
        cy, cx = frame_center(array[0, 0])

        # each branch is computed separately
        for br in range(nbranch):
            # each pattern is computed separately. For each pattern the
            # companions are separated by "fc_rad_sep * fwhm"
            # radius = vector_radd[irad::fc_rad_sep]
            #for irad in range(fc_rad_sep):
            radvec = np.array([0])#vector_radd[irad::fc_rad_sep]
            thetavec = range(int(theta), int(theta) + 360,
                             360 // len(radvec))
            cube_fc = array.copy()
            # filling map with small numbers
            fc_map = np.ones_like(array[:, 0]) * 1e-6
            fcy = []
            fcx = []
            for i in range(radvec.shape[0]):
                flux = 1 #fc_snr * noise_noscal[irad + i * fc_rad_sep]
                cube_fc = cube_inject_companions(cube_fc, psf_template,
                                                 parangles, flux, pxscale,
                                                 rad_dists=[radvec[i]],
                                                 theta=thetavec[i],
                                                 verbose=False)
                y = cy + radvec[i] * np.sin(np.deg2rad(#br * angle_branch +
                                                       thetavec[i]))
                x = cx + radvec[i] * np.cos(np.deg2rad(#br * angle_branch +
                                                       thetavec[i]))
                fc_map = frame_inject_companion(fc_map, psf_template, y, x,
                                                flux)
                fcy.append(y)
                fcx.append(x)

            if verbose:
                msg2 = 'Fake companions injected in branch {} '
                #msg2 += '(pattern {}/{})'
                print(msg2.format(br + 1))#, irad + 1, fc_rad_sep))
                timing(start_time)

            # **************************************************************
            arg = inspect.getargspec(algo).args
            if 'cube' in arg and 'angle_list' in arg and 'verbose' in arg:
                if 'fwhm' in arg:
                    frame_fc = algo(cube=cube_fc, angle_list=parangles,
                                    fwhm=fwhm_med, verbose=False, **algo_dict)
                else:
                    frame_fc = algo(cube=cube_fc, angle_list=parangles,
                                    verbose=False, **algo_dict)

            if verbose:
                msg3 = 'Cube with fake companions processed with {}'
                msg3 += '\nMeasuring its annulus-wise throughput'
                print(msg3.format(algo.__name__))
                timing(start_time)

            # **************************************************************
            injected_flux = np.mean(fc_map, axis=0)
            recovered_flux = frame_fc - frame_nofc
            thruput = np.ones_like(injected_flux)
            num = recovered_flux[np.where(injected_flux > 0)]
            denom = injected_flux[np.where(injected_flux > 0)]
            thruput[np.where(injected_flux > 0)] = num/denom
            thruput[np.where(thruput <= 0)] = 1
            thruput[np.where(thruput > 1)] = 1
            #thruput_arr[br] = thruput

    if verbose:
        msg = 'Finished measuring the throughput in {} branches'
        print(msg.format(nbranch))
        timing(start_time)

    if full_output:
        return thruput, frame_fc, frame_nofc
    else:
        return thruput
    
    
def _prepare_matrix_full(cube, ref_cube, scaling, mask_center_px, mode,
                         verbose):
    if scaling is None:
        if ref_cube is None:
            return cube.copy(), None
        else:
            return cube.copy(), ref_cube.copy()
    cube_tmp = prepare_matrix(cube.copy(), scaling=scaling, 
                              mask_center_px=mask_center_px, mode='fullfr',
                              verbose=False)
    cube_tmp = np.reshape(cube_tmp, cube.shape)
    if ref_cube is not None:
        cube_ref_tmp = prepare_matrix(ref_cube.copy(), scaling=scaling, 
                                      mask_center_px=mask_center_px, 
                                      mode='fullfr', verbose=False)
        cube_ref_tmp = np.reshape(cube_ref_tmp, ref_cube.shape)
    else:
        cube_ref_tmp = None
    return cube_tmp, cube_ref_tmp