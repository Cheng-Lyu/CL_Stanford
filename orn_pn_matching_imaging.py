import plotting_help_py37 as ph
import tifffile as tiff

import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from scipy.ndimage import map_coordinates
from skimage import measure
import os, random, math, copy, glob, nrrd
from matplotlib import animation

from IPython import display
from scipy import ndimage
from PIL import Image
from PIL.TiffTags import TAGS
mpl.rcParams['animation.ffmpeg_path'] = '../ffmpeg'


# helper functions

N = 100

def fit_ellipse(x, y):
    """

    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.


    """

    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()

def cart_to_pol(coeffs):
    """

    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.

    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a*c
    if den > 0:
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    phi = phi % np.pi

    return x0, y0, ap, bp, e, phi

def get_ellipse_pts(params, npts=100, tmin=0, tmax=2*np.pi):
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.

    """

    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y

def interp3(x, y, z, v, xi, yi, zi, **kwargs):
    """Sample a 3D array "v" with pixel corner locations at "x","y","z" at the
    points in "xi", "yi", "zi" using linear interpolation. Additional kwargs
    are passed on to ``scipy.ndimage.map_coordinates``."""
    def index_coords(corner_locs, interp_locs):
        index = np.arange(len(corner_locs))
        if np.all(np.diff(corner_locs) < 0):
            corner_locs, index = corner_locs[::-1], index[::-1]
        return np.interp(interp_locs, corner_locs, index)

    orig_shape = np.asarray(xi).shape
#     xi, yi, zi = np.atleast_1d(xi, yi, zi)
    xi_ = copy.copy(xi)
    yi_ = copy.copy(yi)
    zi_ = copy.copy(zi)
    for arr in [xi_, yi_, zi_]:
        arr.shape = -1
    output = np.empty(xi_.shape, dtype=float)
    coords = [index_coords(*item) for item in zip([x, y, z], [xi_, yi_, zi_])]
    map_coordinates(v, coords, order=1, output=output, mode='constant', cval=0.0, **kwargs)
    return output.reshape(orig_shape)

def gaussian_filter_withnan(ys, sigma=1.0, truncate=4.0):
    """
    :param sigma:  # standard deviation for Gaussian kernel
    :param truncate: # truncate filter at this many sigmas
    :return:
    """
    V = ys.copy()
    V[np.isnan(ys)] = 0
    VV = ndimage.gaussian_filter(V, sigma=sigma, truncate=truncate)

    W = 0 * ys.copy() + 1
    W[np.isnan(ys)] = 0
    WW = ndimage.gaussian_filter(W, sigma=sigma, truncate=truncate)

    return VV / WW


# objects

def get_recs(genotype, parent_folder='./', **kwargs):
    recs = []
    fns = glob.glob(parent_folder + os.path.sep + genotype + '*.tif')
    for fn in fns:
        rec = AntennaLobe_vertical(folder='./', filename=fn[:-4], lobe_side=1, **kwargs)
        recs.append(rec)
        rec = AntennaLobe_vertical(folder='./', filename=fn[:-4], lobe_side=0, **kwargs)
        recs.append(rec)
    return recs

def get_recs_AntennaLobe(genotype, parent_folder='./', ch_pn=0, **kwargs):
    recs = []
    fns = glob.glob(parent_folder + os.path.sep + genotype + '*.tif')
    for fn in fns:
        rec = AntennaLobe(folder='./', filename=fn[:-4], ch_pn=ch_pn, **kwargs)
        recs.append(rec)
    return recs

class Ellipse():
    def __init__(self, xb, yb):
        coeffs = fit_ellipse(xb, yb)
        xc, yc, a, b, _, phi = cart_to_pol(coeffs)
        self.xb, self.yb, self.xc, self.yc, self.a, self.b, self.ab, self.phi = xb, yb, xc, yc, a, b, a+b, phi

class PN():
    def __init__(self, x_pnc, y_pnc, z_pnc, d):
        self.x_pnc, self.y_pnc, self.z_pnc, self.d_toncad = x_pnc, y_pnc, z_pnc, d

class AntennaLobe():

    def __init__(self, folder='./', filename='', ch_pn=1):
        # Initialize antenna lobe mask file
        almask_fns = (glob.glob(folder + os.path.sep + filename + '*almask.nrrd'))
        if len(almask_fns):
            mask_, _ = nrrd.read(almask_fns[0])
            almask = np.swapaxes(np.swapaxes(mask_.T, 0, 1), 1, 2)
            self.ncad = []
            self.Y, self.X, self.Z = almask.shape
        else:
            almask = None

        # Initialize pn mask file
        pnmask_fns = (glob.glob(folder + os.path.sep + filename + '*pnmask.nrrd'))
        if len(pnmask_fns):
            mask_, _ = nrrd.read(pnmask_fns[0])
            pnmask = np.swapaxes(np.swapaxes(mask_.T, 0, 1), 1, 2)
        else:
            pnmask = None

        # Initialize tif file
        if ch_pn is not None:
            fn = (glob.glob(folder + os.path.sep + filename + '*.tif'))[0]
            tif_ = tiff.imread(fn)
            tif = np.swapaxes(np.swapaxes(tif_.T, 0, 1), 2, 3)[:, :, :, ch_pn]
            self.Y, self.X, self.Z = tif.shape
            self.pn = [0, 0]

            img = Image.open(fn)
            exifdata = img.getexif()
            self.xf = 1. / exifdata.get(282)    # 282 is a specific number for getting the XResolution value
            self.yf = self.xf
            self.zf = 1
            # Annotation: In case 282 gives the wrong info, check with the code below
            # for tag_id in exifdata:
            #     # get the tag name, instead of human unreadable tag id
            #     tag = TAGS.get(tag_id, tag_id)
            #     if tag == 'XResolution':
            #         data = exifdata.get(tag_id)
            #         break

        else:
            tif = None
        self.open(tif, almask, pnmask)

    def open(self, tif, almask, pnmask):
        if almask is not None:
            self.add_ncad_to_all_layers(almask)
        if (pnmask is not None) & (tif is not None):
            self.calculate_pn_info(tif,pnmask)

    def add_ncad_to_all_layers(self, almask):
        ncad_l, ncad_r = [], []
        for z in range(self.Z):
            pic = almask[:,:,z]
            contours = measure.find_contours(pic, level=0)
            n = len(contours)
            if n == 0:      # no NCad circle at this layer
                ncad_l.append(None)
                ncad_r.append(None)
            elif n == 1:    # only one NCad circle at this layer, figure out left or right
                ct = contours[0]
                xb = ct[:, 1]
                yb = ct[:, 0]
                side = 0 if (np.mean(xb) < self.X / 2.) else 1      # 0 means left, 1 means right
                ncad_ = Ellipse(xb*self.xf, yb*self.yf)
                if side:
                    ncad_l.append(None)
                    ncad_r.append(ncad_)
                else:
                    ncad_l.append(ncad_)
                    ncad_r.append(None)
            else:           # two NCad circles at this layer
                for i in range(2):
                    ct = contours[i]
                    xb = ct[:, 1]
                    yb = ct[:, 0]
                    ncad_ = Ellipse(xb*self.xf, yb*self.yf)
                    if np.mean(xb) < self.X / 2.:
                        ncad_l.append(ncad_)
                    else:
                        ncad_r.append(ncad_)
        self.ncad = [ncad_l, ncad_r]
        # calculate the top and bottom layer of the left and right NCad
        self.ncadz = np.zeros((2,2)).astype(int)
        for i in np.arange(self.Z):
            if ncad_l[i] is not None:
                self.ncadz[0][0] = i
                break
        for i in np.arange(self.Z)[::-1]:
            if ncad_l[i] is not None:
                self.ncadz[0][1] = i
                break
        for i in np.arange(self.Z):
            if ncad_r[i] is not None:
                self.ncadz[1][0] = i
                break
        for i in np.arange(self.Z)[::-1]:
            if ncad_r[i] is not None:
                self.ncadz[1][1] = i
                break

    def calculate_pn_info(self, tif, pnmask):
        self.pn[0] = None
        self.pn[1] = None
        for i in range(2):
            mask = (pnmask == i+1)
            if mask.sum():
                sig = tif * mask
                sig_sum = np.sum(sig)
                x_pnc = np.sum(np.sum(np.sum(sig, axis=0), axis=1) * np.arange(self.X)) / sig_sum
                y_pnc = np.sum(np.sum(np.sum(sig, axis=1), axis=1) * np.arange(self.Y)) / sig_sum
                z_pnc = np.sum(np.sum(np.sum(sig, axis=0), axis=0) * np.arange(self.Z)) / sig_sum

                side = 0 if (x_pnc < self.X/2.) else 1
                z0_ = np.floor(z_pnc).astype(int)
                z1_ = np.ceil(z_pnc).astype(int)
                x_pnc = x_pnc * self.xf
                y_pnc = y_pnc * self.yf
                z_pnc = z_pnc * self.zf
                # ncad in z0 is larger than the one in z1
                z0, z1 = (z0_, z1_) if (self.ncad[side][z0_].ab > self.ncad[side][z1_].ab) else (z1_, z0_)
                d0 =  np.min(np.sqrt((self.ncad[side][z0].xb-x_pnc)**2+(self.ncad[side][z0].yb-y_pnc)**2))
                theta = np.arctan(self.zf / ((self.ncad[side][z0].ab - self.ncad[side][z1].ab)/2))
                d = np.min([d0 * np.sin(theta), (z0-self.ncadz[side][1])*self.zf])
                self.pn[side] = PN(x_pnc, y_pnc, z_pnc, d)

class AntennaLobe_vertical_BACKUP():

    def __init__(self, folder='./', filename='', lobe_side=0, ch_G=0, ch_R=1, N=100, invert_z=False, ):
        ## In tif and vertical plane, x and y axes corresponds to VERTICAL and HORIZONTAL axes.
        ## In X Y Z and ellipse related x y, it's the traditional x-horizontal, y-inverted vertical definition
        ## lobe_side = 0 left, 1 right

        self.lobe_side = lobe_side
        self.N = N          #step number of the movement along the ellipse long axis
        self.ch_G, self.ch_R = ch_G, ch_R
        # self.z_extra = z_extra

        # signal
        fn = (glob.glob(folder + os.path.sep + filename + '*.tif'))[0]
        tif_ = tiff.imread(fn)
        self.tifs = np.swapaxes(np.swapaxes(tif_.T, 0, 1), 2, 3)
        self.Y, self.X, self.Z = self.tifs[:, :, :, 0].shape
        img = Image.open(fn)
        exifdata = img.getexif()
        self.xf = 1. / exifdata.get(282)  # 282 is a specific number for getting the XResolution value
        self.yf = self.xf
        self.zf = 1
        if invert_z:
            self.tifs = self.tifs[:,:,::-1,:]
        # calculate Gmax using the top 1 percent green signal
        tif = self.tifs[:, :, :,ch_G]
        sig = tif.flatten()
        sig_sorted = np.sort(sig, axis=0)
        self.Gmax = np.nanmean(sig_sorted[int(len(sig) * .99):], axis=0)

        # ORN mask
        almask_fn = (glob.glob(folder + os.path.sep + filename + '*almask.nrrd'))[0]
        self.almask, _ = nrrd.read(almask_fn)
        if invert_z:
            self.almask = self.almask[:,:,::-1]

        # ORN masks to exclude from signal
        gmask_out_fns = glob.glob(folder + os.path.sep + filename + '*gmask_out*')
        if len(gmask_out_fns):
            gmask_out, _ = nrrd.read(gmask_out_fns[0])
            self.gmask_out = np.swapaxes(np.swapaxes(gmask_out.T, 0, 1), 1, 2)
            # in case there are multiple almask_out files
            for gmask_out_fn in gmask_out_fns[1:]:
                gmask_out, _ = nrrd.read(gmask_out_fn)
                self.gmask_out = self.gmask_out + np.swapaxes(np.swapaxes(gmask_out.T, 0, 1), 1, 2)
            if invert_z:
                self.gmask_out = self.gmask_out[:, :, ::-1]
        else:
            self.gmask_out = None

        self.open()

    def open(self):
        self.open_mask()
        if self.gmask_out is not None:
            self.open_mask_out()

    def open_mask(self):
        # search bottom up, only look at the first positive mask as ncad mask
        zs = np.unique(np.where(self.almask > 0)[2])
        for z in zs:
            almask = self.almask[:, :, z].T
            contours = measure.find_contours(almask, level=0)
            for i in range(len(contours)):
                ct = contours[i]
                xb = ct[:, 1]
                yb = ct[:, 0]
                if (self.lobe_side & (np.nanmean(xb) > self.X / 2)) or ((not self.lobe_side) & (np.nanmean(xb) < self.X / 2)):
                    coeffs = fit_ellipse(xb, yb)
                    self.xb, self.yb = xb, yb
                    self.x0, self.y0, self.ap, self.bp, self.e, self.phi = cart_to_pol(coeffs)
                    # self.almask, self.z_mask = almask, (z-self.z_extra)
                    self.almask_2d, self.z_mask = almask, z
                    break
        self.xas = np.linspace(self.x0 + self.ap * np.cos(self.phi), self.x0 - self.ap * np.cos(self.phi), self.N)
        self.yas = np.linspace(self.y0 + self.ap * np.sin(self.phi), self.y0 - self.ap * np.sin(self.phi), self.N)

    def open_mask_out(self):
        inds_out = self.gmask_out > 0
        self.tifs[inds_out,self.ch_G] = 0

    def get_vertical_img(self, istep=50, window_halfwidth_ratio=1.5):
        x, y, z = np.arange(self.X), np.arange(self.Y), np.arange(self.Z)
        window_halfwidth = window_halfwidth_ratio * self.bp
        zx_ratio = self.Z * self.zf / (2 * window_halfwidth * self.xf)

        for i, ch in enumerate([self.ch_G, self.ch_R]):
            x0_, y0_ = self.xas[istep], self.yas[istep]
            xs_ = np.linspace(x0_ - window_halfwidth * np.sin(self.phi), x0_ + window_halfwidth * np.sin(self.phi), self.N)
            ys_ = np.linspace(y0_ + window_halfwidth * np.cos(self.phi), y0_ - window_halfwidth * np.cos(self.phi), self.N)
            zs_ = z
            xs, zs = np.meshgrid(xs_, zs_)
            ys, _ = np.meshgrid(ys_, zs_)
            if i == 0:
                inds_out = (xs[0] < 0) | (xs[0] > self.X) | (ys[0] < 0) | (ys[0] > self.Y)
            img_ = interp3(y, x, z, self.tifs[:, :, :, ch], ys, xs, zs)
            img_[:,inds_out] = 0
            if ch == self.ch_G:
                # imgG = img_ / np.max(img_)
                imgG = img_ / self.Gmax
            if ch == self.ch_R:
                imgR = img_ / np.max(img_)
        return imgG, imgR, zx_ratio

    def get_vertical_Matrix(self, istep=50, window_halfwidth_ratio=1.1, ntheta=27, nr=20, rmax=70,):
        imgG, imgR, _ = self.get_vertical_img(istep, window_halfwidth_ratio)
        window_halfwidth = window_halfwidth_ratio * self.bp
        f = self.zf / (2 * window_halfwidth * self.xf) * self.N  # ratio of the vertical vs radius dimensions
        xv_c = self.z_mask * f       # new axis #1 in the vertical plane: the original z axis
        yv_c = self.N / 2            # new axis #2 in the vertical plane: the original (x, y) axis
        xv_ = np.arange(self.Z) * f
        yv_ = np.arange(self.N)
        # Mxv, Myv: 1st dim vertical, 2nd dim horizontal
        Mxv = np.meshgrid(xv_, yv_, indexing='ij')[0]
        Myv = np.meshgrid(xv_, yv_, indexing='ij')[1]

        # Mr, Mtheta, MG, MR: 1st dim theta, 2nd dim radius
        Mr = np.sqrt((Mxv - xv_c) ** 2 + (Myv - yv_c) ** 2)
        Mtheta = np.arctan2(Myv-yv_c,Mxv-xv_c)
        MG = np.full((ntheta, nr), np.nan)
        MR = np.full((ntheta, nr), np.nan)
        theta_bins = np.linspace(0, 1.5*np.pi, ntheta + 1) - np.pi*0.75
        r_bins = np.linspace(0, rmax, nr + 1)
        for i in range(ntheta):
            inds_theta = (Mtheta > theta_bins[i]) & (Mtheta < theta_bins[i + 1])
            for j in range(nr):
                inds = inds_theta & (Mr > r_bins[j]) & (Mr < r_bins[j + 1])
                if np.sum(inds):
                    MG[i, j] = np.nanmean(imgG[inds])
                    MR[i, j] = np.nanmean(imgR[inds])

        ts = (theta_bins[1:] + theta_bins[:-1]) / 2  # t, short for theta
        rs = (r_bins[1:] + r_bins[:-1]) / 2

        return Mr, Mtheta, MG, MR, ts, rs

    def get_Mflat(self, window_halfwidth_ratio=1.5, ntheta=27, nr=20, rmax=70,):
        Mflat = np.full((self.N, ntheta), np.nan)
        for istep in range(self.N):
            _, _, MG, _, ts, _ = self.get_vertical_Matrix(istep, window_halfwidth_ratio, ntheta, nr, rmax, )
            Mflat[istep] = np.nanmax(MG, axis=1)
            # Mflat[istep] = np.nansum(MG, axis=1)
        ns = np.linspace(0,1,self.N)
        return Mflat, ts, ns

    def get_d_axon2ncad(self, thresh_G=.2, sigma=1, window_halfwidth_ratio=1.1, ntheta=27, nr=20, rmax=70, ):
        ys = []
        for i in range(self.N):
            Mr, Mtheta, MG, MR, ts, rs = self.get_vertical_Matrix(i, window_halfwidth_ratio, ntheta, nr, rmax)
            idxs = np.where(np.nanmax(MG, axis=1) > thresh_G)[0]
            if len(idxs):
                Gs_ = np.nanmean(MG[idxs, :], axis=0)
                Rs_ = np.nanmean(MR[idxs, :], axis=0)
                ys.append(np.nanargmax(Gs_) - np.nanargmax(Rs_))
            else:
                ys.append(np.nan)
        ys = np.array(ys) * rmax / nr
        ys_sm = gaussian_filter_withnan(ys, sigma)
        return ys, ys_sm


    def get_zs_mask_oldversion(self):
        tif_ = self.tifs[:, :, :, self.ch_G]
        marker = (tif_ == 0)
        Mlabeled, nr_objects = ndimage.label(marker)
        ns = []
        for i in np.arange(1, nr_objects + 1):
            ns.append(np.sum(Mlabeled == i))
        sorted_idxs = np.argsort(ns)[::-1]
        self.zs_mask = [0, 0]
        for i in range(2):
            idx = sorted_idxs[i]
            idxss = np.where(Mlabeled == idx + 1)
            lobe_side = (np.nanmean(idxss[1]) > (tif_.shape[1] / 2)).astype(int)
            self.zs_mask[lobe_side] = idxss[2][0]

class AntennaLobe_vertical():

    def __init__(self, folder='./', filename='', lobe_side=0, ch_ORN=0, ch_NCad=1, ch_PN=2, N=100, invert_z=False, ):
        ## In tif and vertical plane, x and y axes corresponds to VERTICAL and HORIZONTAL axes.
        ## In X Y Z and ellipse related x y, it's the traditional x-horizontal, y-inverted vertical definition
        ## lobe_side = 0 left, 1 right

        self.lobe_side = lobe_side
        self.N = N          #step number of the movement along the ellipse long axis
        self.ch_O, self.ch_N, self.ch_P = ch_ORN, ch_NCad, ch_PN
        self.invert_z = invert_z

        # all-channel signals
        fn = (glob.glob(folder + os.path.sep + filename + '*.tif'))[0]
        tif_ = tiff.imread(fn)
        self.tifs = np.swapaxes(np.swapaxes(tif_.T, 0, 1), 2, 3)
        self.Y, self.X, self.Z = self.tifs[:, :, :, 0].shape
        img = Image.open(fn)
        exifdata = img.getexif()
        self.xf = 1. / exifdata.get(282)  # 282 is a specific number for getting the XResolution value
        self.yf = self.xf
        self.zf = 1
        if invert_z:
            self.tifs = self.tifs[:,:,::-1,:]

        # NCad mask
        almask_fn = (glob.glob(folder + os.path.sep + filename + '*almask.nrrd'))[0]
        self.almask, _ = nrrd.read(almask_fn)
        if invert_z:
            self.almask = self.almask[:,:,::-1]

        # ORN and PN masks to exclude from signal
        ORNmask_out_fns = glob.glob(folder + os.path.sep + filename + '*ORNmask_out*')
        self.ORNmask_out = self.mask_out(ORNmask_out_fns)
        PNmask_out_fns = glob.glob(folder + os.path.sep + filename + '*PNmask_out*')
        self.PNmask_out = self.mask_out(PNmask_out_fns)

        self.open()

    def open(self):
        self.open_almask()

        # mask out the unwanted noise signal from ORN and PN channels
        if self.ORNmask_out is not None:
            self.open_mask_out(self.ORNmask_out, self.ch_O)
        if self.PNmask_out is not None:
            self.open_mask_out(self.PNmask_out, self.ch_P)

        # calculate max using the top 1 percent signal
        if self.ch_O < self.tifs.shape[-1]:
            sig_sorted = np.sort(self.tifs[:, :, :, self.ch_O].flatten(), axis=0)
            self.sig_Omax = np.nanmean(sig_sorted[int(len(sig_sorted) * .99):], axis=0)
        else:
            self.sig_Omax = None
        if self.ch_P < self.tifs.shape[-1]:
            sig_sorted = np.sort(self.tifs[:, :, :, self.ch_P].flatten(), axis=0)
            self.sig_Pmax = np.nanmean(sig_sorted[int(len(sig_sorted) * .99):], axis=0)
        else:
            self.sig_Pmax = None

    def open_almask(self):
        # search bottom up, only look at the first positive mask as ncad mask
        zs = np.unique(np.where(self.almask > 0)[2])
        for z in zs:
            almask = self.almask[:, :, z].T
            contours = measure.find_contours(almask, level=0)
            for i in range(len(contours)):
                ct = contours[i]
                xb = ct[:, 1]
                yb = ct[:, 0]
                if (self.lobe_side & (np.nanmean(xb) > self.X / 2)) or ((not self.lobe_side) & (np.nanmean(xb) < self.X / 2)):
                    coeffs = fit_ellipse(xb, yb)
                    self.xb, self.yb = xb, yb
                    self.x0, self.y0, self.ap, self.bp, self.e, self.phi = cart_to_pol(coeffs)
                    # self.almask, self.z_mask = almask, (z-self.z_extra)
                    self.almask_2d, self.z_mask = almask, z
                    break
        self.xas = np.linspace(self.x0 + self.ap * np.cos(self.phi), self.x0 - self.ap * np.cos(self.phi), self.N)
        self.yas = np.linspace(self.y0 + self.ap * np.sin(self.phi), self.y0 - self.ap * np.sin(self.phi), self.N)

    def open_mask_out(self, mask_out, ch):
        inds_out = mask_out > 0
        self.tifs[inds_out, ch] = 0

    def mask_out(self, fns):
        if len(fns):
            mask_out_temp, _ = nrrd.read(fns[0])
            mask_out = np.swapaxes(np.swapaxes(mask_out_temp.T, 0, 1), 1, 2)
            # in case there are multiple almask_out files
            for fn in fns[1:]:
                mask_out_temp, _ = nrrd.read(fn)
                mask_out = mask_out + np.swapaxes(np.swapaxes(mask_out_temp.T, 0, 1), 1, 2)
            if self.invert_z:
                mask_out = mask_out[:, :, ::-1]
        else:
            mask_out = None
        return mask_out

    def get_vertical_img(self, chs, istep=50, window_halfwidth_ratio=1.5):
        x, y, z = np.arange(self.X), np.arange(self.Y), np.arange(self.Z)
        window_halfwidth = window_halfwidth_ratio * self.bp
        zx_ratio = self.Z * self.zf / (2 * window_halfwidth * self.xf)
        x0_, y0_ = self.xas[istep], self.yas[istep]
        xs_ = np.linspace(x0_ - window_halfwidth * np.sin(self.phi), x0_ + window_halfwidth * np.sin(self.phi), self.N)
        ys_ = np.linspace(y0_ + window_halfwidth * np.cos(self.phi), y0_ - window_halfwidth * np.cos(self.phi), self.N)
        zs_ = z
        xs, zs = np.meshgrid(xs_, zs_)
        ys, _ = np.meshgrid(ys_, zs_)
        inds_out = (xs[0] < 0) | (xs[0] > self.X) | (ys[0] < 0) | (ys[0] > self.Y)
        imgs = []

        for i, ch in enumerate(chs):
            img_ = interp3(y, x, z, self.tifs[:, :, :, ch], ys, xs, zs)
            img_[:,inds_out] = 0
            if ch == self.ch_O:
                img = img_ / self.sig_Omax
            elif ch == self.ch_P:
                img = img_ / self.sig_Pmax
            elif ch == self.ch_N:
                img = img_ / np.max(img_)
            imgs.append(img)

        return imgs, zx_ratio

    def get_vertical_Matrix(self, chs, istep=50, window_halfwidth_ratio=1.1, ntheta=27, nr=20, rmax=70,):
        imgs, _ = self.get_vertical_img(chs, istep, window_halfwidth_ratio)
        window_halfwidth = window_halfwidth_ratio * self.bp
        f = self.zf / (2 * window_halfwidth * self.xf) * self.N  # ratio of the vertical vs radius dimensions
        xv_c = self.z_mask * f       # new axis #1 in the vertical plane: the original z axis
        yv_c = self.N / 2            # new axis #2 in the vertical plane: the original (x, y) axis
        xv_ = np.arange(self.Z) * f
        yv_ = np.arange(self.N)
        # Mxv, Myv: 1st dim vertical, 2nd dim horizontal
        Mxv = np.meshgrid(xv_, yv_, indexing='ij')[0]
        Myv = np.meshgrid(xv_, yv_, indexing='ij')[1]

        # Mr, Mtheta, M in Ms: 1st dim theta, 2nd dim radius
        theta_bins = np.linspace(0, 1.5*np.pi, ntheta + 1) - np.pi*0.75
        r_bins = np.linspace(0, rmax, nr + 1)
        ts = (theta_bins[1:] + theta_bins[:-1]) / 2  # t, short for theta
        rs = (r_bins[1:] + r_bins[:-1]) / 2
        Mr = np.sqrt((Mxv - xv_c) ** 2 + (Myv - yv_c) ** 2)
        Mtheta = np.arctan2(Myv - yv_c, Mxv - xv_c)
        Ms = []
        for ch in range(len(chs)):
            Ms.append(np.full((ntheta, nr), np.nan))

        for i in range(ntheta):
            inds_theta = (Mtheta > theta_bins[i]) & (Mtheta < theta_bins[i + 1])
            for j in range(nr):
                inds = inds_theta & (Mr > r_bins[j]) & (Mr < r_bins[j + 1])
                if np.sum(inds):
                    for ch in range(len(chs)):
                        Ms[ch][i, j] = np.nanmean(imgs[ch][inds])

        return Ms, ts, rs, Mr, Mtheta

    def get_Mflat(self, ch, window_halfwidth_ratio=1.5, ntheta=27, nr=20, rmax=70,):
        Mflat = np.full((self.N, ntheta), np.nan)
        for istep in range(self.N):
            Ms, ts, _, _, _ = self.get_vertical_Matrix([ch,], istep, window_halfwidth_ratio, ntheta, nr, rmax, )
            Mflat[istep] = np.nanmax(Ms[0], axis=1)
            # Mflat[istep] = np.nansum(MG, axis=1)
        ns = np.linspace(0, 1, self.N)
        return Mflat, ts, ns

    def get_d_axon2ncad(self, chs, thresh=.2, sigma=1, window_halfwidth_ratio=1.1, ntheta=27, nr=20, rmax=70, ):
        ys = []
        for i in range(self.N):
            Ms, ts, rs, Mr, Mtheta = self.get_vertical_Matrix(chs, i, window_halfwidth_ratio, ntheta, nr, rmax)
            idxs = np.where(np.nanmax(Ms[0], axis=1) > thresh)[0]
            if len(idxs):
                Gs_ = np.nanmean(Ms[0][idxs, :], axis=0)
                Rs_ = np.nanmean(Ms[1][idxs, :], axis=0)
                ys.append(np.nanargmax(Gs_) - np.nanargmax(Rs_))
            else:
                ys.append(np.nan)
        ys = np.array(ys) * rmax / nr
        ys_sm = gaussian_filter_withnan(ys, sigma)
        return ys, ys_sm

    def get_zs_mask_oldversion(self):
        tif_ = self.tifs[:, :, :, self.ch_G]
        marker = (tif_ == 0)
        Mlabeled, nr_objects = ndimage.label(marker)
        ns = []
        for i in np.arange(1, nr_objects + 1):
            ns.append(np.sum(Mlabeled == i))
        sorted_idxs = np.argsort(ns)[::-1]
        self.zs_mask = [0, 0]
        for i in range(2):
            idx = sorted_idxs[i]
            idxss = np.where(Mlabeled == idx + 1)
            lobe_side = (np.nanmean(idxss[1]) > (tif_.shape[1] / 2)).astype(int)
            self.zs_mask[lobe_side] = idxss[2][0]



# test plots--basic
def plot_mask_ellipse(rec):
    ax = ph.large_ax([5, 5])
    ax.plot(rec.xb, rec.yb, '.')  # given points
    xf, yf = get_ellipse_pts((rec.x0, rec.y0, rec.ap, rec.bp, rec.e, rec.phi))
    ax.plot(xf, yf)
    ax.plot(rec.x0, rec.y0, 'o')
    ax.set_aspect('equal', 'box')
    ax.plot(rec.xas, rec.yas, '--', c='k')
    ax.plot(rec.xas[0], rec.yas[0], 'o', c='white')
    xbs = np.linspace(rec.x0 - rec.bp * np.sin(rec.phi), rec.x0 + rec.bp * np.sin(rec.phi), rec.N)
    ybs = np.linspace(rec.y0 + rec.bp * np.cos(rec.phi), rec.y0 - rec.bp * np.cos(rec.phi), rec.N)
    ax.plot(xbs, ybs, '--', c='b')
    ax.plot(xbs[0], ybs[0], 'o', c='white')
    ax.imshow(rec.almask_2d[:, :])

def plot_single_verticalsection(rec, chs, RGB_chs=[2,0,1], RGB_factors=[1,1,1], istep=50, window_halfwidth_ratio=1.5):
    imgs, zx_ratio = rec.get_vertical_img(chs, istep, window_halfwidth_ratio=window_halfwidth_ratio)
    fig = plt.figure(1, [5, 5 * zx_ratio])
    _ = plt.xticks([])
    _ = plt.yticks([])
    imgR = imgs[RGB_chs[0]][::-1, :] if RGB_chs[0] < len(chs) else np.zeros_like(imgs[0])
    imgG = imgs[RGB_chs[1]][::-1, :] if RGB_chs[1] < len(chs) else np.zeros_like(imgs[0])
    imgB = imgs[RGB_chs[2]][::-1, :] if RGB_chs[2] < len(chs) else np.zeros_like(imgs[0])
    # color Ncad white
    # imgR = imgs[RGB[0]][::-1, :] + imgs[RGB[2]][::-1, :] if RGB[0] < len(chs) else np.zeros_like(imgs[0])
    # imgG = imgs[RGB[1]][::-1, :] + imgs[RGB[2]][::-1, :] if RGB[1] < len(chs) else np.zeros_like(imgs[0])
    # imgB = imgs[RGB[2]][::-1, :] if RGB[2] < len(chs) else np.zeros_like(imgs[0])
    ones = np.full_like(imgG, 1)
    img_RGB = np.dstack([np.minimum(imgR*RGB_factors[0], ones), np.minimum(imgG*RGB_factors[1], ones),
                         np.minimum(imgB*RGB_factors[2], ones)])
    _ = plt.imshow(img_RGB, aspect='auto')

def plot_heatmap(packs=(), rec=None, window_halfwidth_ratio=1.5, ntheta=27, nr=20, rmax=70, thresh_G=.2, sigma=1, ):
    if len(packs):
        Mflat, ts, rs = packs
    else:
        Mflat, ts, rs = rec.get_Mflat(window_halfwidth_ratio=window_halfwidth_ratio, ntheta=ntheta, nr=nr, rmax=rmax, )

    # plot
    fig = plt.figure(1, (4, 5))
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])
    ax.pcolormesh(ts, np.linspace(0, 1, N), Mflat, cmap=ph.Greens)
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel('VL to DM')
    ax.set_xticks([-1.57, 1.57])
    _ = ax.set_yticks(np.linspace(0, 1, 6))

def plot_heatmap_distance(rec, ch_heatmap, chs_dist, window_halfwidth_ratio=1.5, ntheta=27, nr=20, rmax=70, thresh_G=.2, sigma=1, ):

    Mflat, ts, rs = rec.get_Mflat(ch_heatmap, window_halfwidth_ratio=window_halfwidth_ratio, ntheta=ntheta, nr=nr, rmax=rmax, )
    ys, ys_sm = rec.get_d_axon2ncad(chs_dist, thresh_G=thresh_G, sigma=sigma, )

    # plot
    fig = plt.figure(1, (5, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])

    ax = plt.subplot(gs[0, 0])
    ax.pcolormesh(ts, np.linspace(0, 1, rec.N), Mflat, cmap=ph.Greens)
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel('VL to DM')
    ax.set_xticks([-1.57, 1.57])
    _ = ax.set_yticks(np.linspace(0, 1, 6))

    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(ys_sm, np.linspace(0, 1, rec.N), c='k', lw=1)
    ph.adjust_spines(ax2, ['bottom'], xlim=[-30, 60], ylim=(0, 1), yticks=[0, 0.5, 1], xticks=[0, 50])
    ax2.axvline(0, c='grey', ls='--', lw=1)
    ax2.axvline(12, c='grey', ls='--', lw=.5)
    ax2.set_xlabel('d')

def get_yss_from_recss(recss, rs_lim=[.5,.6], **kwargs):
    yss = []
    for i, recs in enumerate(recss):
        Ms = []
        for rec in recs:
            Mflat, ts, ns = rec.get_Mflat(**kwargs)
            if not rec.lobe_side:
                Mflat = Mflat[:, ::-1]
            Ms.append(Mflat)
        Ms = np.array(Ms)
        inds = (ns >= rs_lim[0]) & (ns <= rs_lim[1])
        ys = []
        for M in Ms:
            y = np.nanmean(M[inds, :], axis=0)
            ys.append(y / np.nansum(y))  # normalized: density
        yss.append(ys)

    return yss, ts, ns




# plot func

# def get
#
def makemovie_verticalAL_1mv(rec, Nframes=100, video_fps=12, figwidth=5, window_halfwidth_ratio=1.5,
                             showinline=False, moviename='moviename', factor_G=5):
    t_itv_ms = 1000 / video_fps
    window_halfwidth = window_halfwidth_ratio * rec.bp
    zx_ratio = rec.Z * rec.zf / (2 * window_halfwidth * rec.xf)
    fig = plt.figure(1, [figwidth, figwidth * zx_ratio])
    _ = plt.xticks([])
    _ = plt.yticks([])

    imgG, imgR, _ = rec.get_vertical_img(0, window_halfwidth_ratio)
    imgB = np.zeros_like(imgG)
    img_RGB = np.dstack([imgR[::-1, :], np.minimum(imgG[::-1,:]*factor_G, np.full_like(imgG, 1)), imgB[::-1, :]])
    img = plt.imshow(img_RGB, aspect='auto')

    # plot each frame
    def animate(i):
        i_frame = i
        istep = i_frame
        imgG, imgR, _ = rec.get_vertical_img(istep, window_halfwidth_ratio)
        imgB = np.zeros_like(imgG)
        img_RGB = np.dstack([imgR[::-1, :], np.minimum(imgG[::-1,:]*factor_G, np.full_like(imgG, 1)), imgB[::-1, :]])
        img.set_data(img_RGB)

        return fig,

    anim = animation.FuncAnimation(fig, animate, frames=Nframes, interval=t_itv_ms, blit=True)

    if showinline:
        video = anim.to_html5_video()
        html = display.HTML(video)
        display.display(html)
        plt.close()
    else:
        writergif = animation.FFMpegWriter(fps=video_fps)
        anim.save(moviename + '.mp4', writer=writergif)
        # writergif = animation.PillowWriter(fps=video_fps)
        # anim.save(moviename+'.gif', writer=writergif)

def makemovie_verticalAL_ncadaxon(rec, thresh_G=0.2, Nframes=100, video_fps=12, showinline=True, moviename='moviename'):
    fig = plt.figure(1, (4, 3))
    gs = gridspec.GridSpec(1, 1)
    ax0 = plt.subplot(gs[0, 0])
    ax0.set_xticks([0, 30, 60])
    ax0.set_yticks([0, 1])
    ax0.set_xlim([-1, 65])
    ax0.set_ylim([-.1, 1.1])
    line1, = ax0.plot([], [], c=ph.green)
    line2, = ax0.plot([], [], c=ph.red)
    time_text = ax0.text(64, 1, '', ha='right', va='top', clip_on=False)

    # plot each frame
    def animate(i):
        i_frame = i
        Mr, Mtheta, MG, MR, ts, rs = rec.get_vertical_Matrix(istep=i_frame, window_halfwidth_ratio=1.5,
                                                             ntheta=27, nr=25, rmax=65, )
        idxs = np.where(np.nanmax(MG, axis=1) > thresh_G)[0]
        Gs_ = np.nanmean(MG[idxs, :], axis=0)
        Rs_ = np.nanmean(MR[idxs, :], axis=0)
        line1.set_data(rs, Gs_ / np.nanmax(Gs_))
        line2.set_data(rs, Rs_ / np.nanmax(Rs_))
        time_text.set_text('%i/100' % (i_frame + 1))
        return fig,

    anim = animation.FuncAnimation(fig, animate, frames=Nframes, interval=1000 / video_fps, blit=True)

    if showinline:
        video = anim.to_html5_video()
        html = display.HTML(video)
        display.display(html)
        plt.close()
    else:
        writergif = animation.FFMpegWriter(fps=video_fps)
        anim.save(moviename + '.mp4', writer=writergif)

def makemovie_verticalAL_ncadaxon_3color(rec, thresh=0.2, Nframes=100, video_fps=12, showinline=True, moviename='moviename'):
    fig = plt.figure(1, (4, 3))
    gs = gridspec.GridSpec(1, 1)
    ax0 = plt.subplot(gs[0, 0])
    ax0.set_xticks([0, 30, 60])
    ax0.set_yticks([0, 1])
    ax0.set_xlim([-1, 65])
    ax0.set_ylim([-.1, 1.1])
    line1, = ax0.plot([], [], c=ph.green)
    line2, = ax0.plot([], [], c=ph.red)
    line3, = ax0.plot([], [], c=ph.blue)
    time_text = ax0.text(64, 1, '', ha='right', va='top', clip_on=False)

    # plot each frame
    def animate(i):
        i_frame = i
        Ms, ts, rs, Mr, Mtheta = rec.get_vertical_Matrix([0, 1, 2], istep=i_frame, window_halfwidth_ratio=1.5,
                                                         ntheta=27, nr=25, rmax=65, )
        idxs = np.where(np.nanmax(Ms[1], axis=1) > thresh)[0]
        Os_ = np.nanmean(Ms[1][idxs, :], axis=0)
        Ps_ = np.nanmean(Ms[0][idxs, :], axis=0)
        Ns_ = np.nanmean(Ms[2][idxs, :], axis=0)
        line1.set_data(rs, Os_ / np.nanmax(Os_))
        line2.set_data(rs, Ps_ / np.nanmax(Ps_))
        line3.set_data(rs, Ns_ / np.nanmax(Ns_))
        time_text.set_text('%i/100' % (i_frame + 1))
        return fig,

    anim = animation.FuncAnimation(fig, animate, frames=Nframes, interval=1000 / video_fps, blit=True)

    if showinline:
        video = anim.to_html5_video()
        html = display.HTML(video)
        display.display(html)
        plt.close()
    else:
        writergif = animation.FFMpegWriter(fps=video_fps)
        anim.save(moviename + '.mp4', writer=writergif)

def makemovie_verticalAL_2mv_BACKUP(rec, Nframes=100, video_fps=12, thresh_G=.2, window_halfwidth_ratio=1.5, factor_G=1,
                             ntheta=27, nr=20, rmax=70, showinline=False, moviename='moviename', dpi=100, fs=8):
    t_itv_ms = 1000 / video_fps
    istep = 0
    fig_x = 12
    fig_y = 3
    fG, fR = 1.8, 1.2

    # setup axes
    ph.set_fontsize(fs)
    window_halfwidth = window_halfwidth_ratio * rec.bp
    zx_ratio = rec.Z * rec.zf / (2 * window_halfwidth * rec.xf)
    fig = plt.figure(1, [fig_x, fig_y], dpi=dpi)
    gs0 = gridspec.GridSpec(1, 1)
    gs0.update(left=0.01, right=0.30, wspace=0)
    ax0 = plt.subplot(gs0[0, 0])

    x = fig_x * (0.72 - 0.36) * zx_ratio / fig_y
    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(left=0.36, right=0.72, bottom=0.5 * (1 - x), top=0.5 * (1 + x), wspace=0)
    ax1 = plt.subplot(gs1[0, 0])

    gs2 = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
    gs2.update(left=0.78, right=.99, wspace=0.06)
    ax2 = plt.subplot(gs2[0, 0])
    ax3 = plt.subplot(gs2[0, 1])

    # plot
    # ax0
    x0_, y0_ = rec.xas[0], rec.yas[0]
    xs_ = np.linspace(x0_ - window_halfwidth * np.sin(rec.phi), x0_ + window_halfwidth * np.sin(rec.phi), rec.N)
    ys_ = np.linspace(y0_ + window_halfwidth * np.cos(rec.phi), y0_ - window_halfwidth * np.cos(rec.phi), rec.N)
    imgG = np.nanmax(rec.tifs[:, :, :, rec.ch_G], axis=-1)
    imgR = np.nanmax(rec.tifs[:, :, :, rec.ch_R], axis=-1)
    imgB = np.zeros_like(imgG)
    img_RGB = np.dstack([imgR / np.max(imgR) * fR, imgG / np.max(imgG) * fG, imgB / np.max(imgB)])
    ax0.imshow(img_RGB)
    line0, = ax0.plot([], [], c='white', ls='--')
    ax0.set_xlim([0, rec.X])
    ax0.set_ylim([0, rec.Y])
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.invert_yaxis()  # must be stated after set_ylim

    # ax1
    # fig = plt.figure(1, [figwidth, figwidth * zx_ratio])
    imgG, imgR, _ = rec.get_vertical_img(istep, window_halfwidth_ratio)
    imgB = np.zeros_like(imgG)
    img_RGB = np.dstack([imgR[::-1, :], np.minimum(imgG[::-1,:]*factor_G, np.full_like(imgG, 1)), imgB[::-1, :]])
    img = ax1.imshow(img_RGB, aspect='auto')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # ax2,3
    Mflat, ts, rs = rec.get_Mflat(window_halfwidth_ratio, ntheta, nr, rmax, )
    ys, ys_sm = rec.get_d_axon2ncad(thresh_G=thresh_G)
    ax2.pcolormesh(ts, np.linspace(0, 1, rec.N), Mflat, cmap=ph.Greens)
    line2, = ax2.plot([], [], c=ph.grey7, ls='--', lw=1)
    ax2.set_xlabel(r'$\theta$')
    ax2.set_ylabel('VL to DM')
    ax2.set_xticks([-1.57, 1.57])
    ax2.set_xticklabels([r'-$\pi$/2', r'$\pi$/2'])
    _ = ax2.set_yticks(np.linspace(0, 1, 6))

    # ax3 = plt.subplot(gs(left=0.8,right=1,top=1,bottom=0.3))
    ax3.plot(ys_sm, np.linspace(0, 1, rec.N), c='k', lw=1)
    line3, = ax3.plot([], [], c=ph.grey7, ls='--', lw=1)
    ph.adjust_spines(ax3, ['bottom'], xlim=[0, 60], ylim=(0, 1), xticks=[0, 50])
    ax3.axvline(0, c='grey', ls='--', lw=1)
    ax3.axvline(12, c='grey', ls='--', lw=1)
    ax3.set_xlabel('d')

    # plot each frame
    def animate(i):
        i_frame = i
        istep = i_frame

        x0_, y0_ = rec.xas[istep], rec.yas[istep]
        xs_ = np.linspace(x0_ - window_halfwidth * np.sin(rec.phi), x0_ + window_halfwidth * np.sin(rec.phi), rec.N)
        ys_ = np.linspace(y0_ + window_halfwidth * np.cos(rec.phi), y0_ - window_halfwidth * np.cos(rec.phi), rec.N)
        line0.set_data(xs_, ys_)

        imgG, imgR, _ = rec.get_vertical_img(istep, window_halfwidth_ratio)
        imgB = np.zeros_like(imgG)
        img_RGB = np.dstack([imgR[::-1, :], np.minimum(imgG[::-1,:]*factor_G, np.full_like(imgG, 1)), imgB[::-1, :]])
        img.set_data(img_RGB)

        line2.set_data([-2.355,2.355], [istep/100., istep/100.])
        line3.set_data([0,60], [istep/100., istep/100.])

        return fig,

    anim = animation.FuncAnimation(fig, animate, frames=Nframes, interval=t_itv_ms, blit=True)

    if showinline:
        video = anim.to_html5_video()
        html = display.HTML(video)
        display.display(html)
        plt.close()
    else:
        writergif = animation.FFMpegWriter(fps=video_fps)
        anim.save(moviename + '.mp4', writer=writergif)

def makemovie_verticalAL_2mv(rec, chs, ch_heatmap, chs_dist, RGB_chs=[2,0,1], dline_cs=[ph.green, ph.red, ph.blue], RGB_factors=[1,1,1],
                             Nframes=100, video_fps=12, thresh=.2, window_halfwidth_ratio=1.5, factor_G=1,
                             ntheta=27, nr=20, rmax=70, showinline=False, moviename='moviename', dpi=100, fs=8):
    t_itv_ms = 1000 / video_fps
    istep = 0
    fig_x = 12
    fig_y = 3

    # setup axes
    ph.set_fontsize(fs)
    window_halfwidth = window_halfwidth_ratio * rec.bp
    zx_ratio = rec.Z * rec.zf / (2 * window_halfwidth * rec.xf)
    fig = plt.figure(1, [fig_x, fig_y], dpi=dpi)
    gs0 = gridspec.GridSpec(1, 1)
    gs0.update(left=0.01, right=0.25, wspace=0)
    ax0 = plt.subplot(gs0[0, 0])

    x = fig_x * (0.72 - 0.36) * zx_ratio / fig_y
    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(left=0.26, right=0.56, bottom=0.5 * (1 - x), top=0.5 * (1 + x), wspace=0)
    ax1 = plt.subplot(gs1[0, 0])

    gs2 = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
    gs2.update(left=0.61, right=.79, wspace=0.03)
    ax2 = plt.subplot(gs2[0, 0])
    ax3 = plt.subplot(gs2[0, 1])

    gs3 = gridspec.GridSpec(1, 1)
    gs3.update(left=0.82, right=.99, wspace=0)
    ax4 = plt.subplot(gs3[0, 0])

    # plot
    # ax0
    imgs = []
    for ch_ in range(3):
        if RGB_chs[ch_] < len(chs):
            img_ = np.nanmax(rec.tifs[:, :, :, RGB_chs[ch_]], axis=-1)
            img_ = img_ / np.max(img_)
        else:
            img_ = np.zeros_like(rec.tifs[:,:, 0, 0])
        imgs.append(img_)
    img_RGB = np.dstack([imgs[0] * 1.5, imgs[1] * 2, imgs[2] * 2])
    ax0.imshow(img_RGB)
    line0, = ax0.plot([], [], c='white', ls='--')
    ax0.set_xlim([0, rec.X])
    ax0.set_ylim([0, rec.Y])
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.invert_yaxis()  # must be stated after set_ylim

    # ax1
    # fig = plt.figure(1, [figwidth, figwidth * zx_ratio])
    imgs, _ = rec.get_vertical_img(chs, istep, window_halfwidth_ratio)
    imgR = imgs[RGB_chs[0]][::-1, :] if RGB_chs[0] < len(chs) else np.zeros_like(imgs[0])
    imgG = imgs[RGB_chs[1]][::-1, :] if RGB_chs[1] < len(chs) else np.zeros_like(imgs[0])
    imgB = imgs[RGB_chs[2]][::-1, :] if RGB_chs[2] < len(chs) else np.zeros_like(imgs[0])
    ones = np.full_like(imgG, 1)
    img_RGB = np.dstack([np.minimum(imgR * RGB_factors[0], ones), np.minimum(imgG * RGB_factors[1], ones),
                         np.minimum(imgB * RGB_factors[2], ones)])
    img = ax1.imshow(img_RGB, aspect='auto')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # ax2,3
    Mflat, ts, rs = rec.get_Mflat(ch_heatmap, window_halfwidth_ratio, ntheta, nr, rmax, )
    ys, ys_sm = rec.get_d_axon2ncad(chs_dist, thresh=thresh)
    ax2.pcolormesh(ts, np.linspace(0, 1, rec.N), Mflat, cmap=ph.Greens)
    line2, = ax2.plot([], [], c=ph.grey7, ls='--', lw=1)
    ax2.set_xlabel(r'$\theta$')
    ax2.set_ylabel('VL to DM')
    ax2.set_xticks([-1.57, 1.57])
    ax2.set_xticklabels([r'-$\pi$/2', r'$\pi$/2'])
    _ = ax2.set_yticks(np.linspace(0, 1, 6))

    ax3.plot(ys_sm, np.linspace(0, 1, rec.N), c='k', lw=1)
    line3, = ax3.plot([], [], c=ph.grey7, ls='--', lw=1)
    ph.adjust_spines(ax3, ['bottom'], xlim=[0, 60], ylim=(0, 1), xticks=[0, 50])
    ax3.axvline(0, c='grey', ls='--', lw=1)
    ax3.axvline(12, c='grey', ls='--', lw=1)
    ax3.set_xlabel('d')

    # ax4
    ax4.set_xticks([0, 30, 60])
    ax4.set_yticks([0, 1])
    ax4.set_xlim([-1, 65])
    ax4.set_ylim([-.05, 1.05])
    nlines = rec.tifs.shape[-1]
    dline1, = ax4.plot([], [], c=dline_cs[0])
    dline2, = ax4.plot([], [], c=dline_cs[1])
    if nlines == 3:
        dline3, = ax4.plot([], [], c=ph.blue)
    time_text = ax4.text(64, 1, '', ha='right', va='top', clip_on=False)

    # plot each frame
    def animate(i):
        i_frame = i
        istep = i_frame

        x0_, y0_ = rec.xas[istep], rec.yas[istep]
        xs_ = np.linspace(x0_ - window_halfwidth * np.sin(rec.phi), x0_ + window_halfwidth * np.sin(rec.phi), rec.N)
        ys_ = np.linspace(y0_ + window_halfwidth * np.cos(rec.phi), y0_ - window_halfwidth * np.cos(rec.phi), rec.N)
        line0.set_data(xs_, ys_)

        imgs, _ = rec.get_vertical_img(chs, istep, window_halfwidth_ratio)
        imgR = imgs[RGB_chs[0]][::-1, :] if RGB_chs[0] < len(chs) else np.zeros_like(imgs[0])
        imgG = imgs[RGB_chs[1]][::-1, :] if RGB_chs[1] < len(chs) else np.zeros_like(imgs[0])
        imgB = imgs[RGB_chs[2]][::-1, :] if RGB_chs[2] < len(chs) else np.zeros_like(imgs[0])
        ones = np.full_like(imgG, 1)
        img_RGB = np.dstack([np.minimum(imgR * RGB_factors[0], ones), np.minimum(imgG * RGB_factors[1], ones),
                             np.minimum(imgB * RGB_factors[2], ones)])
        img.set_data(img_RGB)

        line2.set_data([-2.355,2.355], [istep/100., istep/100.])
        line3.set_data([0,60], [istep/100., istep/100.])

        Ms, ts, rs, Mr, Mtheta = rec.get_vertical_Matrix(chs, istep=i_frame, window_halfwidth_ratio=1.5,
                                                         ntheta=27, nr=25, rmax=65, )
        idxs = np.where(np.nanmax(Ms[1], axis=1) > thresh)[0]
        Os_ = np.nanmean(Ms[1][idxs, :], axis=0)
        Ps_ = np.nanmean(Ms[0][idxs, :], axis=0)
        if nlines == 3:
            Ns_ = np.nanmean(Ms[2][idxs, :], axis=0)
        dline1.set_data(rs, Os_ / np.nanmax(Os_))
        dline2.set_data(rs, Ps_ / np.nanmax(Ps_))
        if nlines == 3:
            dline3.set_data(rs, Ns_ / np.nanmax(Ns_))
        time_text.set_text('%i/100' % (i_frame + 1))

        return fig,

    anim = animation.FuncAnimation(fig, animate, frames=Nframes, interval=t_itv_ms, blit=True)

    if showinline:
        video = anim.to_html5_video()
        html = display.HTML(video)
        display.display(html)
        plt.close()
    else:
        writergif = animation.FFMpegWriter(fps=video_fps)
        anim.save(moviename + '.mp4', writer=writergif)












