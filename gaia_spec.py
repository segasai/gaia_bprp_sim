import astropy.table as atpy
import scipy.sparse
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import numpy as np
import scipy.interpolate


def betw(x, x1, x2):
    return (x >= x1) & (x < x2)


def interpolator(spec, left, right):
    if left > len(spec) - 1 or right < 0:
        return 0

    left = np.maximum(left, 0)
    right = np.minimum(right, len(spec) - 1)
    l1 = int(np.ceil(left))
    r1 = int(np.floor(right))
    ret = spec[l1 + 1:r1 -
               1].sum() + (l1 - left) * spec[l1] + (right - r1) * spec[r1]
    return ret


def getGaiaInfo():
    tab = atpy.Table().read('nominalXpSamplePositions_colsSimple_PUBISHED.csv')
    wave_bp = tab['BP_ROW1_FOV1_wavelength_nm'][::-1]
    wave_rp = tab['RP_ROW1_FOV1_wavelength_nm']
    wave_disp_bp = tab['BP_ROW1_FOV1_sampling_nm/pix'][::-1]
    wave_disp_rp = tab['RP_ROW1_FOV1_sampling_nm/pix']
    pix = tab['pixelIdx']
    npix = len(pix)
    wave_bp_I = scipy.interpolate.UnivariateSpline(pix, wave_bp)
    wave_rp_I = scipy.interpolate.UnivariateSpline(pix, wave_rp)
    # wave_bp_I1 = scipy.interpolate.UnivariateSpline(wave_bp, pix)
    # wave_rp_I1 = scipy.interpolate.UnivariateSpline(wave_rp, pix)

    resol_pix_bp = np.linspace(1.3, 1.9, npix, True)
    resol_pix_rp = np.linspace(3.5, 4.1, npix, True)
    pixedges_bp = wave_bp_I(np.r_[-0.5, pix + 0.5])
    pixedges_rp = wave_rp_I(np.r_[-0.5, pix + 0.5])

    gau_bp = resol_pix_bp * wave_disp_bp / 2.35
    gau_rp = resol_pix_rp * wave_disp_rp / 2.35

    resol_bp_I = scipy.interpolate.UnivariateSpline(wave_bp,
                                                    gau_bp,
                                                    ext=3,
                                                    s=0)
    resol_rp_I = scipy.interpolate.UnivariateSpline(wave_rp,
                                                    gau_rp,
                                                    ext=3,
                                                    s=0)

    T = atpy.Table().read('GaiaDR2_RevisedPassbands.dat', format='ascii')
    xind = T['col4'] < 1
    trans_bp_wave = T['col1'][xind]
    trans_bp = T['col4'][xind]

    xind = T['col6'] < 1
    trans_rp_wave = T['col1'][xind]
    trans_rp = T['col6'][xind]

    trans_bp_I = scipy.interpolate.UnivariateSpline(trans_bp_wave,
                                                    trans_bp,
                                                    ext=1,
                                                    s=0)
    trans_rp_I = scipy.interpolate.UnivariateSpline(trans_rp_wave,
                                                    trans_rp,
                                                    ext=1,
                                                    s=0)
    info = {}
    info['npix'] = npix
    info['trans_bp_I'] = trans_bp_I
    info['trans_rp_I'] = trans_rp_I
    info['resol_bp_I'] = resol_bp_I
    info['resol_rp_I'] = resol_rp_I
    info['pixedges_bp'] = pixedges_bp
    info['pixedges_rp'] = pixedges_rp
    return info


def getmat(transI, resolI, lam):
    gau = resolI(lam)
    npix = len(lam)
    coeff = []
    xi = []
    yi = []
    for i in range(npix):
        # xind = betw(lam - lam[i], -5 * gau[i], 5 * gau[i])
        x1 = np.searchsorted(lam, lam[i] - 5 * gau[i])
        x2 = np.searchsorted(lam, lam[i] + 5 * gau[i])
        if x1 == x2:
            continue

        pos = slice(x1, x2)
        wht = np.exp(-((lam[pos] - lam[i]) / gau[i])**2)
        wht = wht / wht.sum()
        xi.append(np.zeros(len(wht), dtype=np.int64) + i)
        yi.append(np.arange(x1, x2))
        coeff.append(wht)
    coeff, xi, yi = [np.concatenate(_) for _ in [coeff, xi, yi]]
    mat = scipy.sparse.coo_matrix((coeff, (xi, yi)))
    return mat


class cache:
    info = getGaiaInfo()
    mat_bp = None
    mat_rp = None


def rebin(lam, dat, N):
    assert (np.ptp(np.diff(lam)) < 1e-6)
    # cut the end
    n0 = len(lam)
    n0x = n0 - (n0 % N)
    lam1 = lam[0:n0x].reshape(n0x // N, N)
    dat1 = dat[0:n0x].reshape(n0x // N, N)
    return lam1.mean(axis=1), dat1.mean(axis=1)


def get_bp_rp(filename, rv=0):
    '''
    Get the Bp/Rp spectra from the PHOENIX spectra
    lte05600-2.00-0.0.Alpha=-0.20.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
    '''

    info = cache.info
    dat, hdr = pyfits.getdata(filename, header=True)
    wc = pywcs.WCS(hdr)
    pix = np.arange(len(dat))
    lam = wc.all_pix2world(pix, 1)[0] / 10 * (1 + rv / 3e5)  # nm
    #rebin_factor = 10  # rebin by that much
    #lam, dat = rebin(lam, dat, rebin_factor)
    pix = np.arange(len(dat))  # reconstruct

    dat = dat.astype(
        np.float64) / 6.6260755e-27 / 2.99792458e10 * lam * 1e-7 * (
            np.diff(lam)[0] * 1e-7)  # photons/cm^2/pix

    if cache.mat_bp is None or rv != 0:
        mat_bp = getmat(info['trans_bp_I'], info['resol_bp_I'], lam)
        if rv == 0:
            cache.mat_bp = mat_bp
    else:
        mat_bp = cache.mat_bp

    if cache.mat_rp is None or rv != 0:
        mat_rp = getmat(info['trans_rp_I'], info['resol_rp_I'], lam)
        if rv == 0:
            cache.mat_rp = mat_rp
    else:
        mat_rp = cache.mat_rp

    xp_bp = mat_bp.dot(dat)
    xp_rp = mat_rp.dot(dat)

    prod_bp = xp_bp * info['trans_bp_I'](lam)
    prod_rp = xp_rp * info['trans_rp_I'](lam)
    res = []
    poss = []
    npix = info['npix']
    for i in range(npix + 1):
        poss.append(np.searchsorted(lam, info['pixedges_bp'][i]))
    for i in range(npix):
        res.append(interpolator(prod_bp, poss[i], poss[i + 1] - 1))

    res1 = []
    poss1 = []

    for i in range(npix + 1):
        poss1.append(np.searchsorted(lam, info['pixedges_rp'][i]))
    for i in range(npix):
        res1.append(interpolator(prod_rp, poss1[i], poss1[i + 1] - 1))
    res = np.array(res)
    res1 = np.array(res1)
    return res, res1
