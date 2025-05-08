import torch
import numpy as np
import torch.nn.functional as F

from scipy.stats import multivariate_normal
from util.img_utils import Blurkernel, fft2_m
from motionblur.motionblur import Kernel

def get_operator(problem_config, data_config, device):
    accepted_operators = ['sr_bicubic4', 'sr_bicubic8', 'blur_uni', 'blur_motion', 'blur_gauss', 'blur_aniso', 'color',
                          'sr4', 'sr8', 'inp_box', 'denoising']
    if problem_config["deg"] not in accepted_operators:
        raise RuntimeError('Unknown degradation.')

    if problem_config["deg"] == 'inp_box':
        mask = torch.ones(1, 1, data_config["image_size"], data_config["image_size"])
        mask[:, :, 64:192, 64:192] = 0
        mask = mask.to(device)
        missing_r = torch.nonzero(mask[0, 0].reshape(-1) == 0).long().reshape(-1) * 3
        missing_g = missing_r + 1
        missing_b = missing_g + 1
        missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
        H = Inpainting(3, 256, missing, mask, device)
    elif problem_config["deg"][:10] == 'sr_bicubic':
        factor = int(problem_config["deg"][10:])

        def bicubic_kernel(x, a=-0.5):
            if abs(x) <= 1:
                return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
            elif 1 < abs(x) and abs(x) < 2:
                return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
            else:
                return 0

        k = np.zeros((factor * 4))
        for q in range(factor * 4):
            x = (1 / factor) * (q - np.floor(factor * 4 / 2) + 0.5)
            k[q] = bicubic_kernel(x)
        k = k / np.sum(k)
        kernel = torch.from_numpy(k).float().to(device)
        H = SRConv(kernel / kernel.sum(), 3, 256, device, stride=factor)
    elif problem_config["deg"] == 'blur_uni':
        H = Deblurring(torch.Tensor([1 / 9] * 9).to(device), 3, 256, device)
    elif problem_config["deg"] == 'blur_gauss':
        sigma = np.sqrt(3) # DDRM blur operator weird; to match dps w/ sigma=3 need sigma=sqrt(3)
        pdf = lambda x: torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = pdf(torch.arange(61) - 30).to(device)
        kernel = kernel / kernel.sum()
        H = Deblurring(kernel, 3, 256, device)
    elif problem_config["deg"] == 'blur_motion':
        H = MotionBlurOperator(61, 0.5, 3, 256, device)
    elif problem_config["deg"] == 'blur_aniso':
        sigma = 20
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
        kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
            device)
        sigma = 1
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
        kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
            device)
        H = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), 3,
                         256, device)
    elif problem_config["deg"] == 'color':
        coloring = True
        H = Colorization(256, device)
    elif problem_config["deg"][:2] == 'sr':
        sr = True
        blur_by = int(problem_config["deg"][2:])
        H = SuperResolution(3, 256, blur_by, device)
    else:
        H = Denoising(3, 256, device)

    # compute s_max
    with torch.no_grad():
        b_k = torch.randn(1, 3, 256, 256).to(device)
        for _ in range(50):
            # calculate the matrix-by-vector product Ab
            b_k1 = H.Ht(H.H(b_k))

            # calculate the norm
            b_k1_norm = torch.linalg.norm(b_k1)

            # re normalize the vector
            b_k = b_k1 / b_k1_norm

        s_max = torch.sqrt(torch.linalg.norm(H.Ht(H.H(b_k))))

        H.s_max = s_max.cpu().numpy()

    return H

class H_functions:
    """
    A class replacing the SVD of a matrix H, perhaps efficiently.
    All input vectors are of shape (Batch, ...).
    All output vectors are of shape (Batch, DataDimension).
    """

    def V(self, vec):
        """
        Multiplies the input vector by V
        """
        raise NotImplementedError()

    def Vt(self, vec):
        """
        Multiplies the input vector by V transposed
        """
        raise NotImplementedError()

    def U(self, vec):
        """
        Multiplies the input vector by U
        """
        raise NotImplementedError()

    def Ut(self, vec):
        """
        Multiplies the input vector by U transposed
        """
        raise NotImplementedError()

    def singulars(self):
        """
        Returns a vector containing the singular values. The shape of the vector should be the same as the smaller dimension (like U)
        """
        raise NotImplementedError()

    def add_zeros(self, vec):
        """
        Adds trailing zeros to turn a vector from the small dimension (U) to the big dimension (V)
        """
        raise NotImplementedError()

    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, :singulars.shape[0]])

    def Ht(self, vec):
        """
        Multiplies the input vector by H transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, :singulars.shape[0]]))

    def vamp_mu_1(self, vec, sig_y, sig_ddpm, gamma_1, evals=None):
        temp = self.Vt(vec)

        if evals is None:
            evals = ((self.singulars() / sig_y) ** 2).unsqueeze(0).repeat(vec.shape[0], 1)

        temp[:, :evals.shape[1]] = (evals + sig_ddpm ** 2 + gamma_1[:, 0, None]) ** -1 * temp[:, :evals.shape[1]]
        temp[:, evals.shape[1]:] = (sig_ddpm ** 2 + gamma_1[:, 0, None]) ** -1 * temp[:, evals.shape[1]:]

        return self.V(temp)

    def H_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        singulars_inv = singulars
        singulars_inv[singulars > 0] = 1 / singulars[singulars > 0]
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] * singulars_inv
        return self.V(self.add_zeros(temp))

    def error(self, x, y):
        return ((self.H(x) - y) ** 2).flatten(1).sum(-1)


# a memory inefficient implementation for any general degradation H
class GeneralH(H_functions):
    def mat_by_vec(self, M, v):
        vshape = v.shape[1]
        if len(v.shape) > 2: vshape = vshape * v.shape[2]
        if len(v.shape) > 3: vshape = vshape * v.shape[3]
        return torch.matmul(M, v.view(v.shape[0], vshape,
                                      1)).view(v.shape[0], M.shape[0])

    def __init__(self, H):
        self._U, self._singulars, self._V = torch.svd(H, some=False)
        self._Vt = self._V.transpose(0, 1)
        self._Ut = self._U.transpose(0, 1)

        ZERO = 1e-3
        self._singulars[self._singulars < ZERO] = 0
        print(len([x.item() for x in self._singulars if x == 0]))

    def V(self, vec):
        return self.mat_by_vec(self._V, vec.clone())

    def Vt(self, vec):
        return self.mat_by_vec(self._Vt, vec.clone())

    def U(self, vec):
        return self.mat_by_vec(self._U, vec.clone())

    def Ut(self, vec):
        return self.mat_by_vec(self._Ut, vec.clone())

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        out = torch.zeros(vec.shape[0], self._V.shape[0], device=vec.device)
        out[:, :self._U.shape[0]] = vec.clone().reshape(vec.shape[0], -1)
        return out


# Inpainting
class Inpainting(H_functions):
    def __init__(self, channels, img_dim, missing_indices, inpaint_mask, device):
        self.channels = channels
        self.img_dim = img_dim
        self.s_max = 1.0
        self._singulars = torch.ones(channels * img_dim ** 2 - missing_indices.shape[0]).to(device)
        self.missing_indices = missing_indices
        self.kept_indices = torch.Tensor([i for i in range(channels * img_dim ** 2) if i not in missing_indices]).to(
            device).long()
        self.mask = torch.zeros(2, channels, img_dim, img_dim)
        self.mask[0] = inpaint_mask[0].repeat(3, 1, 1)
        self.mask[1] = 1 - inpaint_mask[0].repeat(3, 1, 1)
        self.mask = torch.ones(1, channels, img_dim, img_dim)

    def V(self, vec):
        temp = vec.clone().reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        out[:, self.kept_indices] = temp[:, :self.kept_indices.shape[0]]
        out[:, self.missing_indices] = temp[:, self.kept_indices.shape[0]:]
        return out.reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1).reshape(vec.shape[0], -1)

    def Vt(self, vec):
        temp = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1).reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        out[:, :self.kept_indices.shape[0]] = temp[:, self.kept_indices]
        out[:, self.kept_indices.shape[0]:] = temp[:, self.missing_indices]
        return out

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        temp = torch.zeros((vec.shape[0], self.channels * self.img_dim ** 2), device=vec.device)
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp


# Denoising
class Denoising(H_functions):
    def __init__(self, channels, img_dim, device):
        self.mask = torch.ones(1, channels, img_dim, img_dim)
        self._singulars = torch.ones(channels * img_dim ** 2, device=device)

    def V(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Vt(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)


# Super Resolution
class SuperResolution(H_functions):
    def __init__(self, channels, img_dim, ratio, device):  # ratio = 2 or 4
        assert img_dim % ratio == 0
        self.img_dim = img_dim
        self.channels = channels
        self.y_dim = img_dim // ratio
        self.ratio = ratio
        H = torch.Tensor([[1 / ratio ** 2] * ratio ** 2]).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(H, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)
        self.mask = torch.ones(1, channels, img_dim, img_dim)

    def V(self, vec):
        # reorder the vector back into patches (because singulars are ordered descendingly)
        temp = vec.clone().reshape(vec.shape[0], -1)
        patches = torch.zeros(vec.shape[0], self.channels, self.y_dim ** 2, self.ratio ** 2, device=vec.device)
        patches[:, :, :, 0] = temp[:, :self.channels * self.y_dim ** 2].view(vec.shape[0], self.channels, -1)
        for idx in range(self.ratio ** 2 - 1):
            patches[:, :, :, idx + 1] = temp[:, (self.channels * self.y_dim ** 2 + idx)::self.ratio ** 2 - 1].view(
                vec.shape[0], self.channels, -1)
        # multiply each patch by the small V
        patches = torch.matmul(self.V_small, patches.reshape(-1, self.ratio ** 2, 1)).reshape(vec.shape[0],
                                                                                              self.channels, -1,
                                                                                              self.ratio ** 2)
        # repatch the patches into an image
        patches_orig = patches.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        recon = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        recon = recon.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        return recon

    def Vt(self, vec):
        # extract flattened patches
        patches = vec.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        unfold_shape = patches.shape
        patches = patches.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio ** 2)
        # multiply each by the small V transposed
        patches = torch.matmul(self.Vt_small, patches.reshape(-1, self.ratio ** 2, 1)).reshape(vec.shape[0],
                                                                                               self.channels, -1,
                                                                                               self.ratio ** 2)
        # reorder the vector to have the first entry first (because singulars are ordered descendingly)
        recon = torch.zeros(vec.shape[0], self.channels * self.img_dim ** 2, device=vec.device)
        recon[:, :self.channels * self.y_dim ** 2] = patches[:, :, :, 0].view(vec.shape[0],
                                                                              self.channels * self.y_dim ** 2)
        for idx in range(self.ratio ** 2 - 1):
            recon[:, (self.channels * self.y_dim ** 2 + idx)::self.ratio ** 2 - 1] = patches[:, :, :, idx + 1].view(
                vec.shape[0], self.channels * self.y_dim ** 2)
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):  # U is 1x1, so U^T = U
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.channels * self.y_dim ** 2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], reshaped.shape[1] * self.ratio ** 2), device=vec.device)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp


# Colorization
class Colorization(H_functions):
    def __init__(self, img_dim, device):
        self.channels = 3
        self.img_dim = img_dim
        # Do the SVD for the per-pixel matrix
        H = torch.Tensor([[0.3333, 0.3334, 0.3333]]).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(H, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)
        self.mask = torch.zeros(self.channels, self.channels, img_dim, img_dim)
        self.mask[0, 0, :, :] = 1.
        self.mask[1, 1, :, :] = 1.
        self.mask[2, 2, :, :] = 1.
        self.mask = torch.ones(1, self.channels, img_dim, img_dim)

    def V(self, vec):
        # get the needles
        needles = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1)  # shape: B, WH, C'
        # multiply each needle by the small V
        needles = torch.matmul(self.V_small, needles.reshape(-1, self.channels, 1)).reshape(vec.shape[0], -1,
                                                                                            self.channels)  # shape: B, WH, C
        # permute back to vector representation
        recon = needles.permute(0, 2, 1)  # shape: B, C, WH
        return recon.reshape(vec.shape[0], -1)

    def Vt(self, vec):
        # get the needles
        needles = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1)  # shape: B, WH, C
        # multiply each needle by the small V transposed
        needles = torch.matmul(self.Vt_small, needles.reshape(-1, self.channels, 1)).reshape(vec.shape[0], -1,
                                                                                             self.channels)  # shape: B, WH, C'
        # reorder the vector so that the first entry of each needle is at the top
        recon = needles.permute(0, 2, 1).reshape(vec.shape[0], -1)
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):  # U is 1x1, so U^T = U
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.img_dim ** 2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], self.channels * self.img_dim ** 2), device=vec.device)
        temp[:, :self.img_dim ** 2] = reshaped
        return temp


# Walsh-Hadamard Compressive Sensing
class WalshHadamardCS(H_functions):
    def fwht(self, vec):  # the Fast Walsh Hadamard Transform is the same as its inverse
        a = vec.reshape(vec.shape[0], self.channels, self.img_dim ** 2)
        h = 1
        while h < self.img_dim ** 2:
            a = a.reshape(vec.shape[0], self.channels, -1, h * 2)
            b = a.clone()
            a[:, :, :, :h] = b[:, :, :, :h] + b[:, :, :, h:2 * h]
            a[:, :, :, h:2 * h] = b[:, :, :, :h] - b[:, :, :, h:2 * h]
            h *= 2
        a = a.reshape(vec.shape[0], self.channels, self.img_dim ** 2) / self.img_dim
        return a

    def __init__(self, channels, img_dim, ratio, perm, device):
        self.channels = channels
        self.img_dim = img_dim
        self.ratio = ratio
        self.perm = perm
        self._singulars = torch.ones(channels * img_dim ** 2 // ratio, device=device)
        self.mask = torch.ones(1, channels, img_dim, img_dim)

    def V(self, vec):
        temp = torch.zeros(vec.shape[0], self.channels, self.img_dim ** 2, device=vec.device)
        temp[:, :, self.perm] = vec.clone().reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1)
        return self.fwht(temp).reshape(vec.shape[0], -1)

    def Vt(self, vec):
        return self.fwht(vec.clone())[:, :, self.perm].permute(0, 2, 1).reshape(vec.shape[0], -1)

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        out = torch.zeros(vec.shape[0], self.channels * self.img_dim ** 2, device=vec.device)
        out[:, :self.channels * self.img_dim ** 2 // self.ratio] = vec.clone().reshape(vec.shape[0], -1)
        return out


# Convolution-based super-resolution
class SRConv(H_functions):
    def mat_by_img(self, M, v, dim):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, dim,
                                         dim)).reshape(v.shape[0], self.channels, M.shape[0], dim)

    def img_by_mat(self, v, M, dim):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, dim,
                                      dim), M).reshape(v.shape[0], self.channels, dim, M.shape[1])

    def __init__(self, kernel, channels, img_dim, device, stride=1):
        self.img_dim = img_dim
        self.channels = channels
        self.s_max = 0.25
        self.ratio = stride
        self.mask = torch.ones(1, channels, img_dim, img_dim)
        small_dim = img_dim // stride
        self.small_dim = small_dim
        # build 1D conv matrix
        H_small = torch.zeros(small_dim, img_dim, device=device)
        for i in range(stride // 2, img_dim + stride // 2, stride):
            for j in range(i - kernel.shape[0] // 2, i + kernel.shape[0] // 2):
                j_effective = j
                # reflective padding
                if j_effective < 0: j_effective = -j_effective - 1
                if j_effective >= img_dim: j_effective = (img_dim - 1) - (j_effective - img_dim)
                # matrix building
                H_small[i // stride, j_effective] += kernel[j - i + kernel.shape[0] // 2]
        # get the svd of the 1D conv
        self.U_small, self.singulars_small, self.V_small = torch.svd(H_small, some=False)
        ZERO = 3e-2
        self.singulars_small[self.singulars_small < ZERO] = 0
        # calculate the singular values of the big matrix
        self._singulars = torch.matmul(self.singulars_small.reshape(small_dim, 1),
                                       self.singulars_small.reshape(1, small_dim)).reshape(small_dim ** 2)
        # permutation for matching the singular values. See P_1 in Appendix D.5.
        self._perm = torch.Tensor([self.img_dim * i + j for i in range(self.small_dim) for j in range(self.small_dim)] + \
                                  [self.img_dim * i + j for i in range(self.small_dim) for j in
                                   range(self.small_dim, self.img_dim)]).to(device).long()

    def V(self, vec):
        # invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim ** 2, self.channels)[:,
                                 :self._perm.shape[0], :]
        temp[:, self._perm.shape[0]:, :] = vec.clone().reshape(vec.shape[0], self.img_dim ** 2, self.channels)[:,
                                           self._perm.shape[0]:, :]
        temp = temp.permute(0, 2, 1)
        # multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small, temp, self.img_dim)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1), self.img_dim).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        # multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone(), self.img_dim)
        temp = self.img_by_mat(temp, self.V_small, self.img_dim).reshape(vec.shape[0], self.channels, -1)
        # permute the entries
        temp[:, :, :self._perm.shape[0]] = temp[:, :, self._perm]
        temp = temp.permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        # invert the permutation
        temp = torch.zeros(vec.shape[0], self.small_dim ** 2, self.channels, device=vec.device)
        temp[:, :self.small_dim ** 2, :] = vec.clone().reshape(vec.shape[0], self.small_dim ** 2, self.channels)
        temp = temp.permute(0, 2, 1)
        # multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small, temp, self.small_dim)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1), self.small_dim).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        # multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone(), self.small_dim)
        temp = self.img_by_mat(temp, self.U_small, self.small_dim).reshape(vec.shape[0], self.channels, -1)
        # permute the entries
        temp = temp.permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat_interleave(3).reshape(-1)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], reshaped.shape[1] * self.ratio ** 2), device=vec.device)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp


class SRBicubic(H_functions):
    def __init__(self, kk, up, channels, img_dim, device, kernel_type='gauss'):
        d = img_dim ** 2  # number of pixels

        # vectorization functions
        d_sqrt = int(np.sqrt(d))

        self.img_dim = img_dim
        self.up = up
        self.channels = channels
        self.kernel_size = kk
        self.pad_size = 64

        W = self.get_bicubic_kernel_2d(kk, up)
        dzp = d_sqrt + self.pad_size  # padded image is dzp-by-dzp
        kh = int((kk - 1) / 2)
        off = int(dzp / 2) - kh
        Wzp = np.zeros((dzp, dzp))  # zero-pad to size dzp-by-dzp
        Wzp[off:off + kk, off:off + kk] = W  # peak at center
        Wzp = np.fft.ifftshift(Wzp)  # peak at (0,0)
        # plt.imshow(Wzp)
        self.fftWzp = torch.tensor(np.real(np.fft.fft2(Wzp))).to(device).unsqueeze(0).repeat(self.channels, 1, 1)  # complex with zero-valued imaginary part
        # print('fftWzp =',fftWzp)

    def im2vec(self, im):
        return im.reshape(im.shape[0], -1).clone()

    def vec2im(self, vec):
        dd = vec.shape[1] // self.channels
        d_sqrt = int(np.sqrt(dd))
        return vec.reshape(vec.shape[0], self.channels, d_sqrt, d_sqrt)

    def pad(self, im, p):
        return F.pad(im, (p, p, p, p), mode='constant', value=0.)

    def padT(self, im, p):
        new_im = im.clone()

        if p != 0:
            new_im = new_im[:, :, p:-p, p:-p]

        return new_im

    def unpad(self, im, p):
        return self.padT(im, p)

    def unpadT(self, im, p):
        return self.pad(im, p)

    def get_bicubic_kernel_2d(self, kk, up, a=None):
        kh = int((kk - 1) / 2)
        x = np.arange(-kh / up, (kh + 1) / up, 1 / up)
        w1 = self.get_bicubic_kernel_1d(x, a)  # 1D kernel
        W = w1[:, None] * w1[None, :]  # 2D kernel

        return W / W.sum()

    def get_bicubic_kernel_1d(self, x, a=None):
        if a is None:
            a = -0.5  # default value
        W = np.zeros((x.shape[0], 2))
        W[:, 0] = (a + 2) * np.abs(x) ** 3 - (a + 3) * np.abs(x) ** 2 + 1
        W[:, 1] = a * np.abs(x) ** 3 - 5 * a * np.abs(x) ** 2 + 8 * a * np.abs(x) - 4 * a
        X = np.zeros((x.shape[0], 2))
        X[:, 0] = (np.abs(x) <= 1)
        X[:, 1] = (np.abs(x) < 2) * (np.abs(x) > 1)
        w = np.sum(W * X, axis=1)

        return w

    def downsample(self, im, factor):
        return im[:, :, 0:-1:factor, 0:-1:factor]

    def upsample(self, im, factor):
        d1 = im.shape[2]
        d2 = im.shape[3]
        Xhi = torch.zeros((im.shape[0], im.shape[1], d1 * factor, d2 * factor)).to(im.device)
        Xhi[:, :, 0:-1:factor, 0:-1:factor] = im
        return Xhi

    def H(self, vec):
        pd = lambda X: self.pad(X, self.pad_size // 2)
        upd = lambda X: self.unpad(X, self.pad_size // 2)
        imupT = lambda X: self.downsample(X, self.up)

        return self.im2vec(imupT(upd(torch.real(torch.fft.ifft2(self.fftWzp[None, :, :, :] * torch.fft.fft2(pd(vec)))))))

    def Ht(self, vec):
        pdT = lambda X: self.padT(X, self.pad_size // 2)
        updT = lambda X: self.unpadT(X, self.pad_size // 2)
        imup = lambda X: self.upsample(X, self.up)

        return self.im2vec(pdT(torch.real(torch.fft.ifft2(self.fftWzp[None, :, :, :] * torch.fft.fft2(updT(imup(self.vec2im(vec))))))))


class SRDebug(H_functions):
    def __init__(self, kk, up, channels, img_dim, device, kernel_type='gauss'):
        d = img_dim ** 2  # number of pixels

        # vectorization functions
        d_sqrt = int(np.sqrt(d))

        self.img_dim = img_dim
        self.channels = channels
        self.s_max = 1.0
        self.kernel_size = kk
        self.pad_size = 64
        self.d_sqrt = d_sqrt
        self.up = up

        kernel = self.get_bicubic_kernel_2d(kk, up)

        k = int((kk - 1) / 2)
        off = int(np.sqrt(d) / 2) - k
        G = np.zeros((d_sqrt, d_sqrt))
        G[off:off + kk, off:off + kk] = kernel  # peak at center
        G = np.fft.ifftshift(G)  # peak at (0,0)
        # plt.imshow(G)
        fftG = np.real(np.fft.fft2(G))

        self.fftG = torch.tensor(fftG).to(device).unsqueeze(0).repeat(self.channels, 1, 1)  # complex with zero-valued imaginary part

        self.sing = self.fftG

    def im2vec(self, im):
        return im.reshape(im.shape[0], -1).clone()

    def vec2im(self, vec):
        if len(vec.shape) == 4:
            return vec.reshape(vec.shape[0], self.channels, vec.shape[-1], vec.shape[-1])
        else:
            dd = vec.shape[1] // self.channels
            d_sqrt = int(np.sqrt(dd))
            return vec.reshape(vec.shape[0], self.channels, d_sqrt, d_sqrt)

    def get_bicubic_kernel_2d(self, kk, up, a=None):
        kh = int((kk - 1) / 2)
        x = np.arange(-kh / up, (kh + 1) / up, 1 / up)
        w1 = self.get_bicubic_kernel_1d(x, a)  # 1D kernel
        W = w1[:, None] * w1[None, :]  # 2D kernel

        return W.sum()

    def get_bicubic_kernel_1d(self, x, a=None):
        if a is None:
            a = -0.5  # default value
        W = np.zeros((x.shape[0], 2))
        W[:, 0] = (a + 2) * np.abs(x) ** 3 - (a + 3) * np.abs(x) ** 2 + 1
        W[:, 1] = a * np.abs(x) ** 3 - 5 * a * np.abs(x) ** 2 + 8 * a * np.abs(x) - 4 * a
        X = np.zeros((x.shape[0], 2))
        X[:, 0] = (np.abs(x) <= 1)
        X[:, 1] = (np.abs(x) < 2) * (np.abs(x) > 1)
        w = np.sum(W * X, axis=1)

        return w

    def downsample(self, im, factor):
        return im[:, :, 0:-1:factor, 0:-1:factor]

    def upsample(self, im, factor):
        d1 = im.shape[2]
        d2 = im.shape[3]
        Xhi = torch.zeros((im.shape[0], im.shape[1], d1 * factor, d2 * factor)).to(im.device)
        Xhi[:, :, 0:-1:factor, 0:-1:factor] = im
        return Xhi

    def H(self, vec):
        imupT = lambda X: self.downsample(X, self.up)

        return self.im2vec(imupT(torch.real(torch.fft.ifft2(self.fftG[None, :, :, :] * torch.fft.fft2(self.vec2im(vec))))))

    def Ht(self, vec):
        imup = lambda X: self.upsample(X, self.up)

        return self.im2vec(torch.real(torch.fft.ifft2(self.fftG[None, :, :, :] * torch.fft.fft2(imup(self.vec2im(vec))))))

    def V(self, vec):
        return torch.real(torch.fft.ifft2(self.vec2im(vec)) * self.d_sqrt)

    def Vt(self, vec):
        return torch.fft.fft2(self.vec2im(vec)) / self.d_sqrt

    def U(self, vec):
        imupT = lambda X: self.downsample(X, self.up)

        return imupT(torch.real(torch.fft.ifft2(self.vec2im(vec))))

    def Ut(self, vec):
        imup = lambda X: self.upsample(X, self.up)

        return torch.fft.fft2(imup(self.vec2im(vec)))

    def singulars(self):
        return self.sing

    def add_zeros(self, vec):
        return vec.clone()


# Deblurring
class Deblurring(H_functions):
    def mat_by_img(self, M, v):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, self.img_dim,
                                         self.img_dim)).reshape(v.shape[0], self.channels, M.shape[0], self.img_dim)

    def img_by_mat(self, v, M):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, self.img_dim,
                                      self.img_dim), M).reshape(v.shape[0], self.channels, self.img_dim, M.shape[1])

    def __init__(self, kernel, channels, img_dim, device, ZERO=0.):
        kernel = kernel.double()
        self.img_dim = img_dim
        self.channels = channels
        self.mask = torch.ones(1, channels, img_dim, img_dim)
        # build 1D conv matrix
        H_small = torch.zeros(img_dim, img_dim, device=device).double()
        for i in range(img_dim):
            for j in range(i - kernel.shape[0] // 2, i + kernel.shape[0] // 2):
                if j < 0 or j >= img_dim: continue
                H_small[i, j] = kernel[j - i + kernel.shape[0] // 2]
        # get the svd of the 1D conv
        self.U_small, self.singulars_small, self.V_small = torch.svd(H_small, some=False)
        # ZERO = 3e-2
        self.singulars_small[self.singulars_small < ZERO] = 0
        # calculate the singular values of the big matrix
        self._singulars = torch.matmul(self.singulars_small.reshape(img_dim, 1),
                                       self.singulars_small.reshape(1, img_dim)).reshape(img_dim ** 2)
        # sort the big matrix singulars and save the permutation
        self._singulars, self._perm = self._singulars.sort(descending=True)  # , stable=True)

    def V(self, vec):
        # invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device).double()
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim ** 2, self.channels).double()
        temp = temp.permute(0, 2, 1)
        # multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small, temp)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out.float()

    def Vt(self, vec):
        # multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone().double())
        temp = self.img_by_mat(temp, self.V_small).reshape(vec.shape[0], self.channels, -1)
        # permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1).float()

    def U(self, vec):
        # invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device).double()
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim ** 2, self.channels).double()
        temp = temp.permute(0, 2, 1)
        # multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small, temp)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out.float()

    def Ut(self, vec):
        # multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone().double())
        temp = self.img_by_mat(temp, self.U_small).reshape(vec.shape[0], self.channels, -1)
        # permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1).float()

    def singulars(self):
        return self._singulars.repeat(1, 3).reshape(-1)

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)


class DeblurringGeneral(H_functions):
    def __init__(self, kk, sigma, channels, img_dim, device, kernel_type='gauss'):
        d = img_dim ** 2  # number of pixels

        # vectorization functions
        d_sqrt = int(np.sqrt(d))

        self.img_dim = img_dim
        self.channels = channels
        self.s_max = 1.0
        self.kernel_size = kk
        self.pad_size = 64

        if kernel_type == 'gauss':
            kernel = self.get_gauss_kernel(kk, sigma)
        else:
            raise NotImplementedError

        dzp = d_sqrt + self.pad_size  # padded image is dzp-by-dzp
        kh = int((kk - 1) / 2)
        off = int(dzp / 2) - kh
        Gzp = np.zeros((dzp, dzp))  # zero-pad to size dzp-by-dzp
        Gzp[off:off + kk, off:off + kk] = kernel  # peak at center
        Gzp = np.fft.ifftshift(Gzp)  # peak at (0,0)
        self.fftGzp = torch.tensor(np.real(np.fft.fft2(Gzp))).to(device).unsqueeze(0).repeat(self.channels, 1, 1)  # complex with zero-valued imaginary part

        self.sing = self.fftGzp.reshape(-1)

    def im2vec(self, im):
        return im.reshape(im.shape[0], -1).clone()

    def vec2im(self, vec):
        return vec.reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)

    def pad(self, im, p):
        return F.pad(im, (p, p, p, p), mode='constant', value=0.)

    def padT(self, im, p):
        new_im = im.clone()

        if p != 0:
            new_im = new_im[:, :, p:-p, p:-p]

        return new_im

    def unpad(self, im, p):
        return self.padT(im, p)

    def unpadT(self, im, p):
        return self.pad(im, p)

    def get_gauss_kernel(self, kk, sig):
        gauss2 = multivariate_normal(mean=[0, 0], cov=[[sig ** 2, 0], [0, sig ** 2]])
        k = int((kk - 1) / 2)
        p1, p2 = np.mgrid[-k:k + 1:1, -k:k + 1:1]
        pp = np.dstack((p1, p2))  # kk-by-kk pixel grid
        G = gauss2.pdf(pp)  # peaks at center or (k,k)
        # G = np.fft.ifftshift(G) # peaks at (0,0)
        # plt.imshow(G)
        return G / np.sum(G)

    def H(self, vec):
        pd = lambda X: self.pad(X, self.pad_size // 2)
        upd = lambda X: self.unpad(X, self.pad_size // 2)

        return self.im2vec(upd(torch.real(torch.fft.ifft2(self.fftGzp[None, :, :, :] * torch.fft.fft2(pd(self.vec2im(vec)))))))

    def Ht(self, vec):
        pdT = lambda X: self.padT(X, self.pad_size // 2)
        updT = lambda X: self.unpadT(X, self.pad_size // 2)

        return self.im2vec(pdT(torch.real(torch.fft.ifft2(self.fftGzp[None, :, :, :] * torch.fft.fft2(updT(self.vec2im(vec)))))))

    def H_no_pad(self, vec):
        return self.im2vec(torch.real(torch.fft.ifft2(self.fftG[None, :, :, :] * torch.fft.fft2(vec))))

    def Ht_no_pad(self, vec):
        return self.im2vec(torch.real(torch.fft.ifft2(self.fftG[None, :, :, :] * torch.fft.fft2(self.vec2im(vec)))))

    def V(self, vec):
        return self.im2vec(torch.fft.ifft2(self.vec2im(vec)))

    def Vt(self, vec):
        return self.im2vec(torch.fft.fft2(self.vec2im(vec)))

    def U(self, vec):
        return self.img_dim * self.im2vec(torch.fft.ifft2(self.vec2im(vec)))

    def Ut(self, vec):
        return (1 / self.img_dim) * self.im2vec(torch.fft.fft2(self.vec2im(vec)))

    def singulars(self):
        return self.sing

    def add_zeros(self, vec):
        return vec.clone()

class DeblurringDebug(H_functions):
    def __init__(self, kk, sigma, channels, img_dim, device, kernel_type='gauss'):
        d = img_dim ** 2  # number of pixels

        # vectorization functions
        d_sqrt = int(np.sqrt(d))

        self.img_dim = img_dim
        self.channels = channels
        self.s_max = 1.0
        self.kernel_size = kk
        self.pad_size = 64
        self.d_sqrt = d_sqrt

        if kernel_type == 'gauss':
            kernel = self.get_gauss_kernel(kk, sigma)
        else:
            raise NotImplementedError

        # dzp = d_sqrt + self.pad_size  # padded image is dzp-by-dzp
        # kh = int((kk - 1) / 2)
        # off = int(dzp / 2) - kh
        # Gzp = np.zeros((dzp, dzp))  # zero-pad to size dzp-by-dzp
        # Gzp[off:off + kk, off:off + kk] = kernel  # peak at center
        # Gzp = np.fft.ifftshift(Gzp)  # peak at (0,0)
        # fftGzp = np.real(np.fft.fft2(Gzp))

        kk = 61  # kernel is kk-by-kk pixels
        sig = 3  # gaussian blur std in pixels
        k = int((kk - 1) / 2)
        off = int(np.sqrt(d) / 2) - k
        G = np.zeros((d_sqrt, d_sqrt))
        G[off:off + kk, off:off + kk] = kernel  # peak at center
        G = np.fft.ifftshift(G)  # peak at (0,0)
        # plt.imshow(G)
        fftG = np.real(np.fft.fft2(G))

        self.fftG = torch.tensor(fftG).to(device).unsqueeze(0).repeat(self.channels, 1, 1)  # complex with zero-valued imaginary part

        self.sing = self.fftG

    def im2vec(self, im):
        return im.reshape(im.shape[0], -1).clone()

    def vec2im(self, vec):
        return vec.reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)

    def get_gauss_kernel(self, kk, sig):
        gauss2 = multivariate_normal(mean=[0, 0], cov=[[sig ** 2, 0], [0, sig ** 2]])
        k = int((kk - 1) / 2)
        p1, p2 = np.mgrid[-k:k + 1:1, -k:k + 1:1]
        pp = np.dstack((p1, p2))  # kk-by-kk pixel grid
        G = gauss2.pdf(pp)  # peaks at center or (k,k)
        # G = np.fft.ifftshift(G) # peaks at (0,0)
        # plt.imshow(G)
        return G / np.sum(G)

    def H(self, vec):
        return self.im2vec(torch.real(torch.fft.ifft2(self.fftG[None, :, :, :] * torch.fft.fft2(self.vec2im(vec)))))

    def Ht(self, vec):
        return self.im2vec(torch.real(torch.fft.ifft2(self.fftG[None, :, :, :] * torch.fft.fft2(self.vec2im(vec)))))

    def V(self, vec):
        return torch.fft.ifft2(self.vec2im(vec)) * self.d_sqrt

    def Vt(self, vec):
        return torch.fft.fft2(self.vec2im(vec)) / self.d_sqrt

    def U(self, vec):
        return self.V(vec)

    def Ut(self, vec):
        return self.Vt(vec)

    def singulars(self):
        return self.sing

    def add_zeros(self, vec):
        return vec.clone()


class MotionBlurOperator(H_functions):
    def __init__(self, kk, intensity, channels, img_dim, device, ready_kernel=None):
        d = img_dim ** 2  # number of pixels

        # vectorization functions
        d_sqrt = int(np.sqrt(d))

        self.img_dim = img_dim
        self.channels = channels
        self.s_max = 1.0
        self.kernel_size = kk
        self.pad_size = 64
        # self.pad_size = 0

        if ready_kernel is not None:
            kernel = ready_kernel
        else:
            kernel = Kernel(size=(kk, kk), intensity=intensity)
            kernel = torch.tensor(kernel.kernelMatrix).numpy()

        dzp = d_sqrt + self.pad_size  # padded image is dzp-by-dzp
        kh = int((kk - 1) / 2)
        off = int(dzp / 2) - kh
        Gzp = np.zeros((dzp, dzp))  # zero-pad to size dzp-by-dzp
        Gzp[off:off + kk, off:off + kk] = kernel  # peak at center
        Gzp = np.fft.ifftshift(Gzp)  # peak at (0,0)
        self.fftGzp = torch.tensor(np.real(np.fft.fft2(Gzp))).to(device).unsqueeze(0).repeat(self.channels, 1, 1)  # complex with zero-valued imaginary part

        self.sing = self.fftGzp.reshape(-1)

    def im2vec(self, im):
        return im.reshape(im.shape[0], -1).clone()

    def vec2im(self, vec):
        return vec.reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)

    def pad(self, im, p):
        # replicate-padding function and its adjoint
        return F.pad(im, (p, p, p, p), mode='replicate')

    def padT(self, im, p):
        new_im = im.clone()

        if p != 0:
            new_im[:, :, p, :] = torch.sum(new_im[:, :, :p + 1, :], dim=2)
            new_im[:, :, -p - 1, :] = torch.sum(new_im[:, :,  -p - 1:, :], dim=2)
            new_im[:, :, :, p] = torch.sum(new_im[:, :, :, :p + 1], dim=3)
            new_im[:, :, :, -p - 1] = torch.sum(new_im[:, :, :, -p - 1:], dim=3)
            new_im = new_im[:, :, p:-p, p:-p]

        return new_im

    def unpad(self, im, p):
        return im[:, :, p:-p, p:-p]

    def unpadT(self, im, p):
        return F.pad(im, (p, p, p, p), mode='constant', value=0.)

    def H(self, vec):
        pd = lambda X: self.pad(X, self.pad_size // 2)
        upd = lambda X: self.unpad(X, self.pad_size // 2)

        return self.im2vec(upd(torch.real(torch.fft.ifft2(self.fftGzp[None, :, :, :] * torch.fft.fft2(pd(self.vec2im(vec))))))).float()

    def Ht(self, vec):
        pdT = lambda X: self.padT(X, self.pad_size // 2)
        updT = lambda X: self.unpadT(X, self.pad_size // 2)

        return self.im2vec(pdT(torch.real(torch.fft.ifft2(self.fftGzp[None, :, :, :] * torch.fft.fft2(updT(self.vec2im(vec))))))).float()

    # def H(self, vec):
    #     return self.im2vec(torch.real(torch.fft.ifft2(self.fftGzp[None, :, :, :] * torch.fft.fft2(self.vec2im(vec))))).float()
    #
    # def Ht(self, vec):
    #     return self.im2vec(torch.real(torch.fft.ifft2(self.fftGzp[None, :, :, :] * torch.fft.fft2(self.vec2im(vec))))).float()

    def V(self, vec):
        return self.im2vec(torch.fft.ifft2(self.vec2im(vec)))

    def Vt(self, vec):
        return self.im2vec(torch.fft.fft2(self.vec2im(vec)))

    def U(self, vec):
        return self.img_dim * self.im2vec(torch.fft.ifft2(self.vec2im(vec)))

    def Ut(self, vec):
        return (1 / self.img_dim) * self.im2vec(torch.fft.fft2(self.vec2im(vec)))

    def singulars(self):
        return self.sing

    def add_zeros(self, vec):
        return vec.clone()

class MotionBlurOperatorDPS(H_functions):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  # should we keep this device term?

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        self.kernel_matrix = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(self.kernel_matrix)

    def im2vec(self, im):
        return im.reshape(im.shape[0], -1).clone()

    def vec2im(self, vec):
        return vec.reshape(vec.shape[0], 3, 256, 256).clone()

    def H(self, vec):
        return self.im2vec(self.conv(self.vec2im(vec)))

    def Ht(self, vec):
        return self.im2vec(self.conv(self.vec2im(vec)))

    def get_kernel(self):
        kernel = self.kernel.kernelMatrix.type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)

class DeblurDPS(H_functions):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def H(self, vec):
        return self.conv(vec)

    def Ht(self, vec):
        return self.conv(vec)

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)


# Anisotropic Deblurring
class Deblurring2D(H_functions):
    def mat_by_img(self, M, v):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, self.img_dim,
                                         self.img_dim)).reshape(v.shape[0], self.channels, M.shape[0], self.img_dim)

    def img_by_mat(self, v, M):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, self.img_dim,
                                      self.img_dim), M).reshape(v.shape[0], self.channels, self.img_dim, M.shape[1])

    def __init__(self, kernel1, kernel2, channels, img_dim, device):
        self.img_dim = img_dim
        self.channels = channels
        self.mask = torch.ones(1, channels, img_dim, img_dim)

        # build 1D conv matrix - kernel1
        H_small1 = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel1.shape[0] // 2, i + kernel1.shape[0] // 2):
                if j < 0 or j >= img_dim: continue
                H_small1[i, j] = kernel1[j - i + kernel1.shape[0] // 2]
        # build 1D conv matrix - kernel2
        H_small2 = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel2.shape[0] // 2, i + kernel2.shape[0] // 2):
                if j < 0 or j >= img_dim: continue
                H_small2[i, j] = kernel2[j - i + kernel2.shape[0] // 2]
        # get the svd of the 1D conv
        self.U_small1, self.singulars_small1, self.V_small1 = torch.svd(H_small1, some=False)
        self.U_small2, self.singulars_small2, self.V_small2 = torch.svd(H_small2, some=False)
        ZERO = 3e-2
        self.singulars_small1[self.singulars_small1 < ZERO] = 0
        self.singulars_small2[self.singulars_small2 < ZERO] = 0
        # calculate the singular values of the big matrix
        self._singulars = torch.matmul(self.singulars_small1.reshape(img_dim, 1),
                                       self.singulars_small2.reshape(1, img_dim)).reshape(img_dim ** 2)
        # sort the big matrix singulars and save the permutation
        self._singulars, self._perm = self._singulars.sort(descending=True)  # , stable=True)

    def V(self, vec):
        # invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim ** 2, self.channels)
        temp = temp.permute(0, 2, 1)
        # multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small1, temp)
        out = self.img_by_mat(out, self.V_small2.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        # multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small1.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.V_small2).reshape(vec.shape[0], self.channels, -1)
        # permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        # invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim ** 2, self.channels)
        temp = temp.permute(0, 2, 1)
        # multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small1, temp)
        out = self.img_by_mat(out, self.U_small2.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        # multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small1.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.U_small2).reshape(vec.shape[0], self.channels, -1)
        # permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat(1, 3).reshape(-1)

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)