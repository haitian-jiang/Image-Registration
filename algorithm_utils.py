import numpy as np
import scipy.signal
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1


def normalize(img: np.ndarray):
    return (img - img.min()) / (img.max() - img.min() + 1e-6)


def bilinear_interp(coord: np.ndarray, img: np.ndarray):
    """bilinear interpolation for `coord` from `img`

    The index of `coord` is the coordinates for the interpolated moved image, which is what
    we return, with the same shape as `coord`.
    The value in `coord` is the coordinate(float) of `img`(moving image), we interpolate it to get 
    the color of such float point coordinates to place in the correspoding index of the moved image.

    Args:
    -----
    coord: np.ndarray, dtype=float
        The index is the coordinates of the fixed image, the value is the float point coordinates of 
        the moving image. Its value is calculated by the transformation on its index.
    img: np.ndarray
        The moving image, where the color is get.

    Returns:
    --------
        The deformed version of moving image, registered to the fixed image.
    """
    # get the lower bound integer coordinates for calculating weights
    coord_trunc = np.floor(coord).astype(int)
    # get the the residual weights
    coord_res = coord - coord_trunc
    # padding for points outside with edge value by np.clip
    get_pix = lambda x, y: img[np.clip(x, 0, img.shape[0] - 1),
                               np.clip(y, 0, img.shape[1] - 1)]
    # bilinear interpolation
    return (
            (1 - coord_res[0]) * (1 - coord_res[1]) * get_pix(coord_trunc[0], coord_trunc[1]) +
            (1 - coord_res[0]) * coord_res[1] * get_pix(coord_trunc[0], coord_trunc[1] + 1) +
            coord_res[0] * (1 - coord_res[1]) * get_pix(coord_trunc[0] + 1, coord_trunc[1]) +
            coord_res[0] * coord_res[1] * get_pix(coord_trunc[0] + 1, coord_trunc[1] + 1)
    )


###################################################
#    loss functions
###################################################
def mse(moved_img: np.ndarray, fixed_img: np.ndarray):
    """Mean Squared Error.
    A more popular way of SSD (Sum of Squared Difference)
    """
    return ((moved_img - fixed_img) ** 2).mean()


def cc(moved_img: np.ndarray, fixed_img: np.ndarray):
    # correlation coefficient
    return -np.corrcoef(moved_img.flatten(), fixed_img.flatten())[0,1]


def ncc(moved_img: np.ndarray, fixed_img: np.ndarray, win=(25,25)):
    """Normalized Cross Correlation.

    Use negative mean local normalized cross correlation as loss function.
    Larger cross correlation means higher similarity, so we use the negative one
    to serve as a loss metric. Global cross correlation typically do not perform 
    as well as the local one. 
    Formula: ∑(x_i-μ_x)(y_i-μ_y) / [sqrt(∑(x_i-μ_x)²)*sqrt(∑(y_i-μ_y)²)]
    The range of this loss function is [-1, 1], the lower the better.
    """
    # get the necessary quantities
    I, J = moved_img, fixed_img
    sum_filt = np.ones(win)
    I2, J2 = I * I, J * J
    IJ = I * J

    # use convolution to get the value of local regions
    I_sum = scipy.signal.convolve2d(I, sum_filt, 'same')
    J_sum = scipy.signal.convolve2d(J, sum_filt, 'same')
    I2_sum = scipy.signal.convolve2d(I2, sum_filt, 'same')
    J2_sum = scipy.signal.convolve2d(J2, sum_filt, 'same')
    IJ_sum = scipy.signal.convolve2d(IJ, sum_filt, 'same')
    
    # normalization
    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    # calculate cross correlation
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    # calculate normalized cross correlation
    ncc = cross * cross / (I_var * J_var + 1e-6)
    return -ncc.mean()


def nmi(moved_img: np.ndarray, fixed_img: np.ndarray):
    """Negative Mutual Information.

    Mutual information shows the similarity of the distribution of gray scale values.
    Mutual information is non-negative and symmetric for the input distributions.
    Formula: MI(x,y)=∑_x ∑_y P(x,y) * log( P(x,y) / [P(x)*P(y)] )
    Here we use negative mutual information to make it a loss function, lower is better.
    """
    # get the joint distribution
    hgram, _, _ = np.histogram2d(moved_img.ravel(), fixed_img.ravel(), bins=32)
    pxy = hgram / float(hgram.sum())
    
    # get two marginal distributions
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # broadcast to multiply marginals

    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return -np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def kl(moved_img: np.ndarray, fixed_img: np.ndarray, bins=128):
    """K-L divergence of the two images' distributions

    K-L divergence measure the distance of two distributions.
    It is non-negative by Gibbs' inequality, and it is not symmetric.
    Here we measure KL(fixed_img || moved_img).
    Formula: ∑ p_i * log(p_i/q_i)
    """
    # get distributions
    hist_m, _ = np.histogram(moved_img, bins)
    hist_f, _ = np.histogram(fixed_img, bins)
    p = hist_f / hist_f.sum()
    q = hist_m / hist_m.sum()
    # plus 1e-10 to avoid divide by zero or log(0)
    return np.sum( p * np.log((p+1e-10) / (q+1e-10)) )
