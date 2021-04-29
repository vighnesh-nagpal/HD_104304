import pyklip.instruments.utils.radonCenter as rad
from astropy.io import fits
import numpy as np

if __name__ == "__main__":
    fname='rawtest16.fits'
    img = fits.getdata(fname,ignore_missing_end=True)
    xguess=688
    yguess=471
    centre=rad.searchCenter(img, xguess, yguess, 80, size_cost=10 ,theta = np.linspace(0,360,60))
    print(f'Guess:{xguess,yguess}')
    print(f'Result: {centre}')