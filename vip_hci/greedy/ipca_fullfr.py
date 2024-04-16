#! /usr/bin/env python
"""Full-frame iterative PCA algorithm for ADI or ADI+RDI cubes.

The concept was proposed in [PAI18]_ and [PAI21]. Iterative PCA-ARDI was then
implemented and tested in [JUI24].

.. [PAI18]
   | Pairet et al. 2018
   | **Reference-less algorithm for circumstellar disks imaging**
   | *In Proceedings of iTWIST'18, 23*
   | `https://arxiv.org/abs/1812.01333
     <https://arxiv.org/abs/1812.01333>`_

.. [PAI21]
   | Pairet et al. 2021
   | **MAYONNAISE: a morphological components analysis pipeline for
   circumstellar discs and exoplanets imaging in the near-infrared**
   | *MNRAS 503, 3724*
   | `https://arxiv.org/abs/2008.05170
     <https://arxiv.org/abs/2008.05170>`_

.. [JUI24]
   | Juillard et al. 2024
   | **Combining reference-star and angular differential imaging for
   high-contrast imaging of extended sources**
   | *A&A in press*
   | `https://arxiv.org/abs/2008.05170
     <https://arxiv.org/abs/2008.05170>`_

"""

__author__ = 'Valentin Christiaens'
__all__ = ['ipca',
           'find_significant_signals']

from dataclasses import dataclass
import numpy as np
from typing import Union, List
from ..config.paramenum import ALGO_KEY
from ..config.utils_param import separate_kwargs_dict
from ..config import Progressbar
from ..psfsub import pca, PCA_Params
from ..preproc import cube_derotate, cube_collapse
from ..metrics import stim_map, inverse_stim_map
from ..var import prepare_matrix, mask_circle, frame_filter_lowpass


@dataclass
class IPCA_Params(PCA_Params):
    """
    Set of parameters for the iterative PCA routine.

    Inherits from PCA_Params.

    See function `ipca` for documentation.
    """

    mode: str = None
    strategy: str = "ADI"
    ncomp_step: int = 1
    nit: int = 1
    thr: Union[float, str] = 0.
    thr_mode: str = 'STIM'
    r_out: float = None
    r_max: float = None
    smooth_ker: Union[float, List] = None
    rtol: float = 1e-2
    atol: float = 1e-2
    continue_without_smooth_after_conv: bool = False


def ipca(*all_args: List, **all_kwargs: dict):
    """
    Run iterative version of full-frame PCA.

    The algorithm finds significant disc or planet signal in the final PCA
    image, then subtracts it from the input cube (after rotation) just before
    (and only for) projection onto the principal components. This is repeated
    n_it times, which progressively reduces geometric biases in the image
    (e.g. negative side lobes for ADI).
    This is similar to the algorithm presented in Pairet et al. (2020).

    The same parameters as pca() can be provided, except 'batch'. There are two
    additional parameters related to the iterative algorithm: the number of
    iterations (n_it) and the threshold (thr) used for the identification of
    significant signals.

    Note: The iterative PCA can only be used in ADI, RDI, ARDI or R+ADI modes.

    Parameters
    ----------
    cube : str or numpy ndarray, 3d or 4d
        Input cube (ADI).
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
    mode: str or None, opt {'Pairet18', 'Pairet21'}
        - If None: runs with provided value of 'n_comp' for 'n_it' iterations,
        and considering threshold 'thr'.
        - If 'Pairet18': runs for n_comp iterations, with n_comp=1,...,n_comp,
        at each iteration (if `nit` is provided it is ignored). thr set to 0
        (ignored if provided).
        - If 'Pairet21': runs with n_comp=1,...,n_comp, and n_it times for each
        n_comp (i.e. outer loop on n_comp, inner loop on n_it). thr set to 0
        (ignored if provided). 'thr' parameter discarded, always set to 0.
        - If 'Christiaens21': same as 'Pairet21', but with 'thr' parameter used.
    ncomp : int or tuple/list of int, optional
        How many PCs are used as a lower-dimensional subspace to project the
        target frames.
        - if mode is None:
            * ADI or RDI: if an int is provided, ``ncomp`` is the number of PCs
            extracted from ``cube`` itself.
            * RADI: can be a list/tuple of 2 elements corresponding to the ncomp
              to be used for RDI and then ADI.
         - if mode is not None:
             ncomp should correspond to the maximum number of principal
             components to be tested. The increment will be ncomp_step.
    ncomp_step: int, opt
        Incremental step for number of principal components - used when mode is
        not None.
    nit: int, opt
        Number of iterations for the iterative PCA.
        - if mode is None:
            total number of iterations
        - if mode is 'Pairet18':
            this parameter is ignored. Number of iterations will be ncomp.
        - if mode is 'Pairet21':
            iterations per tested ncomp.
    strategy: str {'ADI, 'RDI', 'ARDI', 'RADI'}, opt
        Whether to do iterative ADI only ('ADI'), iterative RDI only ('RDI'),
        iterative ADI and RDI together ('ARDI', i.e. with a combined PCA
        library), or iterative RDI followed by iterative ADI consecutively
        ('RADI'). A reference cube `cube_ref` must be provided for 'RDI', 'ARDI'
        and 'RADI'.
    thr: float or 'auto', opt
        Minimum threshold used to identify significant signals in the PCA image
        obtained at each iteration.
    thr_mode: str {'STIM', 'abs'}, opt
        How the threshold is expressed: whether based on the STIM map ('STIM')
        or expressed as absolute pixel intensity threshold ('abs'). The optimal
        choice may depend on whether you are speckle noise or sensitivity
        limited in your region of interest. For the 'STIM' criterion, the
        threshold corresponds to the minimum intensity in the STIM map computed
        from PCA residuals (Pairet et al. 2019), as expressed in units of
        maximum intensity obtained in the inverse STIM map (i.e. obtained from
        using opposite derotation angles).
    r_out: float or None, opt
        Outermost radius in pixels of circumstellar signals (estimated). This
        will be used if thr is set to 'auto'. The max STIM value beyond that
        radius will be used as minimum threshold. If r_out is set to None, a
        quarter of the frame width will be used.
    r_max: float or None, opt
        Max radius in pixels where the STIM map will be considered. The max STIM
        value beyond r_out but within r_max will be used as minimum threshold.
        If r_max is set to None, the half-width of the frames will be used.
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
        radius of the circular mask.
    source_xy : tuple of int, optional
        For ADI-PCA, this triggers a frame rejection in the PCA library, with
        ``source_xy`` as the coordinates X,Y of the center of the annulus where
        the PA criterion is estimated. When ``ncomp`` is a tuple, a PCA grid is
        computed and the S/Ns (mean value in a 1xFWHM circular aperture) of the
        given (X,Y) coordinates are computed.
    delta_rot : int or float, optional
        Factor for tunning the parallactic angle threshold, expressed in FWHM.
        Default is 1 (excludes 1xFHWM on each side of the considered frame).
    fwhm : float, optional
        Known size of the FHWM in pixels to be used. Default value is 4.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    imlib2 : str, optional
        See the documentation of the
        ``vip_hci.preproc.cube_rescaling_wavelengths`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
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
    rtol: float, optional
        Relative tolerance threshold element-wise in the significant signal
        image compared to the same image obtained either 1 or 2 iterations
        before, to consider convergence [more details in np.allclose].
    atol: float, optional
        Absolute tolerance threshold element-wise in the significant signal
        image compared to the same image obtained either 1 or 2 iterations
        before, to consider convergence [more details in np.allclose].
    smooth_ker: None or float or list/1darray of floats, optional
        If not None, size in pixels of the Gaussian kernel to use on
        post-processed image obtained at each iteration to include the expected
        spatial correlation of neighbouring pixels. Default is to consider a
        kernel size of 1 pixel. If a list/1d array, length should be equal to
        the number of iterations. Depending on your data, starting with a
        larger kernel (e.g. FWHM/2) and progressively decreasing it with
        iterations can provide better results in terms of signal recovery.
    continue_without_smooth_after_conv: bool, opt
        Whether to continue to iterate after convergence, but without applying
        the smoothing criterion. At this stage, with a good first guess of the
        circumstellar signals, this can lead to a very sharp final image.

    Returns
    -------
    frame : numpy ndarray
        2D array, median combination of the de-rotated/re-scaled residuals cube.
        Always returned. This is the final image obtained at the last iteration.
    it_cube: numpy ndarray
        [full_output=True] 3D array with final image from each iteration.
    sig_images: numpy ndarray
        [full_output=True] 3D array similar to it_cube, but only containing
        significant signals identified at each iteration.
    residuals_cube : numpy ndarray
        [full_output=True] Residuals cube from the last iteration.
    residuals_cube_ : numpy ndarray
        [full_output=True] Derotated residuals cube from the last iteration.
    stim_cube: numpy ndarray
        [full_output=True] 3D array with normalized STIM map from each iteration
        used for the identification of significant signals. If thr_mode is set
        to 'abs', this contains instead a binary mask of identified significant
        signals at each iteration.
    it_cube_nd: numpy ndarray
        [full_output=True] 3D array with image obtained with the (disk-)empty
        cube at each iteration, serves to show how much residual disk flux is
        not yet captured at each iteration.
    """

    def _blurring_2d(array, mask_center_sz, fwhm_sz=2):
        if mask_center_sz:
            frame_mask = mask_circle(array, radius=mask_center_sz+fwhm_sz,
                                     fillwith=np.nan, mode='out')
            frame_mask2 = mask_circle(array, radius=mask_center_sz,
                                      fillwith=np.nan, mode='out')
            frame_filt = frame_filter_lowpass(frame_mask, mode='gauss',
                                              fwhm_size=fwhm_sz,
                                              iterate=False)
            nonan_loc = np.where(np.isfinite(frame_mask2))
            array[nonan_loc] = frame_filt[nonan_loc]
        else:
            array = frame_filter_lowpass(array, mode='gauss',
                                         fwhm_size=fwhm_sz, iterate=False)
        return array

    def _blurring_3d(array, mask_center_sz, fwhm_sz=2):
        bl_array = np.zeros_like(array)
        for i in range(array.shape[0]):
            bl_array[i] = _blurring_2d(array[i], mask_center_sz, fwhm_sz)
        return bl_array

    # 0. Identify parameters
    # Separating the parameters of the ParamsObject from optional rot_options
    class_params, rot_options = separate_kwargs_dict(
        initial_kwargs=all_kwargs, parent_class=IPCA_Params
    )
    # Do the same to separate IROLL and ROLL params
    pca_params, ipca_params = separate_kwargs_dict(
        initial_kwargs=class_params, parent_class=PCA_Params
    )

    # Extracting the object of parameters (if any)
    algo_params = None
    if ALGO_KEY in rot_options.keys():
        algo_params = rot_options[ALGO_KEY]
        del rot_options[ALGO_KEY]

    if algo_params is None:
        algo_params = IPCA_Params(*all_args, **class_params)

    # force full_output
    pca_params['full_output'] = True
    pca_params['verbose'] = False  # too verbose otherwise

    # 1. Prepare/format additional parameters depending on chosen options
    mask_center_px = algo_params.mask_center_px  # None? what's better in pca?
    mask_rdi_tmp = None
    if algo_params.strategy == 'ADI' and algo_params.cube_ref is None:
        ref_cube = None
        mask_rdi_tmp = algo_params.mask_rdi
    elif algo_params.cube_ref is not None:
        if algo_params.strategy == 'ADI':
            msg = "WARNING: requested strategy is 'ADI' but reference cube "
            msg += "detected! Strategy automatically switched to 'ARDI'."
            print(msg)
            algo_params.strategy = 'ARDI'
        if algo_params.mask_rdi is not None:
            mask_rdi_tmp = algo_params.mask_rdi.copy()
        if algo_params.cube_ref is None:
            raise ValueError("cube_ref should be provided for RDI or RADI")
        if algo_params.strategy == 'ARDI':
            ref_cube = np.concatenate((algo_params.cube,
                                       algo_params.cube_ref), axis=0)
        else:
            ref_cube = algo_params.cube_ref.copy()
    else:
        msg = "strategy not recognized: must be ADI, RDI, ARDI or RADI"
        raise ValueError(msg)

    if isinstance(algo_params.ncomp, (float, int)):
        ncomp_list = [algo_params.ncomp]
        if algo_params.strategy == 'RADI':
            ncomp_list.append(algo_params.ncomp)
    elif isinstance(algo_params.ncomp, (tuple, list)):
        ncomp_list = algo_params.ncomp
        if len(algo_params.ncomp) == 1:
            if algo_params.strategy == 'RADI':
                ncomp_list.append(algo_params.ncomp)
        elif not len(algo_params.ncomp) == 2:
            raise ValueError("Length of npc list cannot be larger than 2")
    else:
        raise TypeError("ncomp should be float, int, tuple or list")

    ncomp_tmp = ncomp_list[0]
    nframes = algo_params.cube.shape[0]

    if algo_params.mode is not None:
        final_ncomp = list(range(1, ncomp_tmp+1, algo_params.ncomp_step))
        if algo_params.mode == 'Pairet18':
            algo_params.nit = ncomp_tmp
            final_ncomp = list(range(1, ncomp_tmp+1, algo_params.ncomp_step))
            algo_params.thr = 0
        elif algo_params.mode in ['Pairet21', 'Christiaens21']:
            final_ncomp = []
            for npc in range(1, ncomp_tmp+1, algo_params.ncomp_step):
                for ii in range(algo_params.nit):
                    final_ncomp.append(npc)
            algo_params.nit = len(final_ncomp)
            if algo_params.mode == 'Pairet21':
                algo_params.thr = 0
    else:
        final_ncomp = [ncomp_tmp]*algo_params.nit

    # Scale cube and cube_ref if necessary
    cube_tmp = prepare_matrix(algo_params.cube, scaling=algo_params.scaling,
                              mask_center_px=mask_center_px, mode='fullfr',
                              verbose=False)
    cube_tmp = np.reshape(cube_tmp, algo_params.cube.shape)
    if ref_cube is not None:
        cube_ref_tmp = prepare_matrix(ref_cube, scaling=algo_params.scaling,
                                      mask_center_px=mask_center_px,
                                      mode='fullfr', verbose=False)
        cube_ref_tmp = np.reshape(cube_ref_tmp, ref_cube.shape)
    else:
        cube_ref_tmp = None

    # 2. Get a first disc estimate, using PCA
    pca_params['cube_ref'] = ref_cube
    res = pca(**pca_params, **rot_options)
    frame = res[0]
    residuals_cube = res[-2]
    residuals_cube_ = res[-1]
    # smoothing and manual derotation if requested
    smooth_ker = algo_params.smooth_ker
    if smooth_ker is None or np.isscalar(smooth_ker):
        smooth_ker = np.array([smooth_ker]*algo_params.nit, dtype=object)
    else:
        smooth_ker = np.array(smooth_ker, dtype=object)
    # if smooth_ker[0] is not None:
    #     residuals_cube = _blurring_3d(residuals_cube, None,
    #                                   fwhm_sz=smooth_ker[0])
    #     residuals_cube_ = cube_derotate(residuals_cube,
    #                                     algo_params.angle_list,
    #                                     imlib=algo_params.imlib,
    #                                     nproc=algo_params.nproc)
    #     frame = cube_collapse(residuals_cube_, algo_params.collapse)
    if smooth_ker[0] is not None:
        frame = frame_filter_lowpass(frame, fwhm_size=smooth_ker[0])

    # 3. Identify significant signals with STIM map
    it_cube = np.zeros([algo_params.nit, frame.shape[0], frame.shape[1]])
    it_cube_nd = np.zeros_like(it_cube)
    stim_cube = np.zeros_like(it_cube)
    sig_images = np.zeros_like(it_cube)
    it_cube[0] = frame.copy()
    it_cube_nd[0] = frame.copy()
    if algo_params.thr_mode == 'STIM':
        sig_mask, nstim = find_significant_signals(residuals_cube,
                                                   residuals_cube_,
                                                   algo_params.angle_list,
                                                   algo_params.thr,
                                                   mask=mask_center_px,
                                                   r_out=algo_params.r_out,
                                                   r_max=algo_params.r_max)
    else:
        sig_mask = np.ones_like(frame)
        sig_mask[np.where(frame < algo_params.thr)] = 0
        nstim = sig_mask.copy()
    sig_image = frame.copy()
    sig_image[np.where(1-sig_mask)] = 0
    sig_image[np.where(sig_image < 0)] = 0
    sig_images[0] = sig_image.copy()
    stim_cube[0] = nstim.copy()
    mask_rdi_tmp = None  # after first iteration do not use it any more

    # 4.Loop, updating the reference cube before projection by subtracting the
    #   best disc estimate. This is done by providing sig_cube.
    for it in Progressbar(range(1, algo_params.nit), desc="Iterating..."):
        # Uncomment here (and comment below) to do like IROLL
        # if smooth_ker[it] is not None:
        #     frame = _blurring_2d(frame, None, fwhm_sz=smooth_ker[it])
        # create and rotate sig cube
        sig_cube = np.repeat(frame[np.newaxis, :, :], nframes, axis=0)
        sig_cube = cube_derotate(sig_cube, -algo_params.angle_list,
                                 imlib=algo_params.imlib,
                                 nproc=algo_params.nproc)

        if algo_params.thr_mode == 'STIM':
            # create and rotate binary mask
            mask_sig = np.zeros_like(sig_image)
            mask_sig[np.where(sig_image > 0)] = 1
            sig_mcube = np.repeat(mask_sig[np.newaxis, :, :], nframes, axis=0)
            sig_mcube = cube_derotate(sig_mcube, -algo_params.angle_list,
                                      imlib='skimage', interpolation='bilinear',
                                      nproc=algo_params.nproc)
            sig_cube[np.where(sig_mcube < 0.5)] = 0
            sig_cube[np.where(sig_cube < 0)] = 0
        else:
            sig_cube[np.where(sig_cube < algo_params.thr)] = 0

        if algo_params.strategy == 'ARDI':
            ref_cube = np.concatenate((algo_params.cube-sig_cube,
                                       algo_params.cube_ref), axis=0)
            cube_ref_tmp = prepare_matrix(ref_cube, scaling=algo_params.scaling,
                                          mask_center_px=mask_center_px,
                                          mode='fullfr', verbose=False)
            cube_ref_tmp = np.reshape(cube_ref_tmp, ref_cube.shape)

        # Run PCA on original cube
        # Update PCA PARAMS
        pca_params['cube'] = algo_params.cube
        pca_params['cube_ref'] = ref_cube
        pca_params['ncomp'] = final_ncomp[it]
        pca_params['scaling'] = algo_params.scaling
        pca_params['cube_sig'] = sig_cube
        pca_params['mask_rdi'] = mask_rdi_tmp

        res = pca(**pca_params, **rot_options)

        frame = res[0]
        residuals_cube = res[-2]
        it_cube[it] = frame.copy()

        # DON'T otherwise frame is smoothed twice!
        # smoothing and manual derotation if requested
        if smooth_ker[it] is not None:
            residuals_cube = _blurring_3d(residuals_cube, None,
                                          fwhm_sz=smooth_ker[it])
            residuals_cube_ = cube_derotate(residuals_cube,
                                            algo_params.angle_list,
                                            imlib=algo_params.imlib,
                                            nproc=algo_params.nproc)
            frame = cube_collapse(residuals_cube_, algo_params.collapse)

        # Run PCA on disk-empty cube
        # Update PCA PARAMS
        pca_params['cube'] = cube_tmp-sig_cube
        pca_params['cube_ref'] = cube_ref_tmp
        pca_params['cube_sig'] = None
        pca_params['scaling'] = None

        res_nd = pca(**pca_params, **rot_options)

        residuals_cube_nd = res_nd[-2]
        frame_nd = res_nd[0]

        if algo_params.thr_mode == 'STIM':
            sig_mask, nstim = find_significant_signals(residuals_cube_nd,
                                                       residuals_cube_,
                                                       algo_params.angle_list,
                                                       algo_params.thr,
                                                       mask=mask_center_px,
                                                       r_out=algo_params.r_out,
                                                       r_max=algo_params.r_max)
        else:
            sig_mask = np.ones_like(frame)
            sig_mask[np.where(frame < algo_params.thr)] = 0
            nstim = sig_mask.copy()
        inv_sig_mask = np.ones_like(sig_mask)
        inv_sig_mask[np.where(sig_mask)] = 0
        if mask_center_px:
            inv_sig_mask = mask_circle(inv_sig_mask, mask_center_px,
                                       fillwith=1)
        sig_image = frame.copy()
        sig_image[np.where(inv_sig_mask)] = 0
        sig_image[np.where(sig_image < 0)] = 0

        it_cube[it] = frame.copy()
        it_cube_nd[it] = frame_nd.copy()
        sig_images[it] = sig_image.copy()
        stim_cube[it] = nstim.copy()

        # check if improvement compared to last iteration
        if it > 1:
            cond1 = np.allclose(sig_image, sig_images[it-1],
                                rtol=algo_params.rtol, atol=algo_params.atol)
            cond2 = np.allclose(sig_image, sig_images[it-2],
                                rtol=algo_params.rtol, atol=algo_params.atol)
            if cond1 or cond2:
                if algo_params.strategy in ['ADI', 'RDI', 'ARDI']:
                    msg = "Convergence criterion met after {:.0f} iterations"
                    condB = algo_params.continue_without_smooth_after_conv
                    if smooth_ker[it] is not None and condB:
                        smooth_ker[it+1:] = None
                        msg2 = "... Switching smoothing off and iterating more!"
                        if algo_params.verbose:
                            print(msg.format(it)+msg2)
                    else:
                        if algo_params.verbose:
                            print("Final " + msg.format(it))
                        break
                if algo_params.strategy == 'RADI':
                    # continue to iterate with ADI
                    ncomp_tmp = ncomp_list[1]
                    algo_params.strategy = 'ADI'
                    ref_cube = None
                    if algo_params.verbose:
                        msg = "After {:.0f} iterations, PCA-RDI -> PCA-ADI."
                        print(msg.format(it))

    # mask everything last
    if mask_center_px is not None:
        frame = mask_circle(frame, mask_center_px)
        it_cube = mask_circle(it_cube, mask_center_px)
        residuals_cube = mask_circle(residuals_cube, mask_center_px)
        residuals_cube_ = mask_circle(residuals_cube_, mask_center_px)
        it_cube_nd = mask_circle(it_cube_nd, mask_center_px)

    if algo_params.full_output:
        return (frame, it_cube[:it+1], sig_images[:it+1], residuals_cube,
                residuals_cube_, stim_cube[:it+1], it_cube_nd[:it+1])
    else:
        return frame


def find_significant_signals(residuals_cube, residuals_cube_, angle_list,
                             thr, mask=0, r_out=None, r_max=None):
    """Identify significant signals with STIM map (outside a mask).

    Parameters
    ----------
    residuals_cube : numpy ndarray, 3d
        Cube of residual images.
    residuals_cube_ : numpy ndarray, 3d
        Cube of derotated residual images.
    angle_list : numpy ndarray, 1d
        Corresponding parallactic angle for each frame.
    thr: float or 'auto', opt
        Minimum threshold used to identify significant signals in the PCA image
        obtained at each iteration.
    mask : positive float
        (Inner) radius of a mask in pixels, if any, beyond which the maximum of
        the inverse STIM map is considered.
    r_out: float or None, opt
        Outermost radius in pixels of circumstellar signals (estimated). This
        will be used if thr is set to 'auto'. The max STIM value beyond that
        radius will be used as minimum threshold. If r_out is set to None, a
        quarter of the frame width will be used.
    r_max: float or None, opt
        Max radius in pixels where the STIM map will be considered. The max STIM
        value beyond r_out but within r_max will be used as minimum threshold.
        If r_max is set to None, the half-width of the frames will be used.

    Returns
    -------
    good_mask : numpy ndarray
        2D binary map corresponding to significant pixel intensities.
    norm_stim: numpy ndarray
        Normalized STIM map, i.e. STIM map normalized by the maximum intensity
        in the inverse STIM map.

    """
    stim = stim_map(residuals_cube_)
    inv_stim = inverse_stim_map(residuals_cube, angle_list)
    if mask:
        inv_stim = mask_circle(inv_stim, mask)
    max_inv = np.amax(inv_stim)
    if max_inv == 0:
        max_inv = 1  # np.amin(stim[np.where(stim>0)])
    if thr == 'auto':
        if r_out is None:
            r_out = residuals_cube.shape[-1]//4
        if r_max is None:
            r_max = residuals_cube.shape[-1]//2
        inv_stim_rout = mask_circle(inv_stim, r_out)
        inv_stim_rmax = mask_circle(inv_stim_rout, r_max, mode='out')
        thr = np.amax(inv_stim_rmax)/max_inv
    norm_stim = stim/max_inv
    good_mask = np.zeros_like(stim)
    good_mask[np.where(norm_stim > thr)] = 1

    return good_mask, norm_stim
