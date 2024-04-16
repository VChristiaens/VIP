"""
Subpackage ``greedy`` contains iterative implementations of stellar PSF
modelling + subtraction algorithms. The following methods have been implemented:
- iterative roll subtraction [HEA00]_ / [CHR24]_
- iterative PCA in full frame [PAI18]_ / [PAI21]_ / [JUI24]_
- iterative NMF in full frame (cite latest VIP paper if used).
- FEVES (Christiaens et al. in prep.).
"""
from .feves import *
from .feves_opt import *
from .inmf_fullfr import *
from .ipca_fullfr import *
from .ipca_local import *
from .iroll import *
from .utils_itpca import *
