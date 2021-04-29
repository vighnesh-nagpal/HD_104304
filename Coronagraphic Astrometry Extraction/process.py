import pyklip.instruments.utils.radonCenter as rad
import pyklip.fakes
from astropy.io import fits
import numpy as np
import scipy
import glob
import os
import numpy as np
import scipy.ndimage as ndi
import pdb
import scipy.optimize as optimize
from matplotlib import pyplot as plt

## UNCOMMENT THIS IF USING STUFF FROM POST 2015
distXgeoim='./distortion/post2015_distort_X.fits'
distYgeoim='./distortion/post2015_distort_Y.fits'

## UNCOMMENT THIS IF USING STUFF FROM PRE 2015

# distXgeoim='./distortion/pre2015_distort_X.fits'
# distYgeoim='./distortion/pre2015_distort_Y.fits'

def undistort(in_frame, save=False, out=''):
    """
    Non-PyRAF/IRAF version of nirc2dewarp.py -- note: this method does not conserve flux!

    Using distortion solution from Yelda et al. 2010, based on 'nirc2dewarp.py' found on NIRC2 Distortion wiki

    Files needed: Distortion x and y fits map (set path in distXgeoim and distYgeoim)
    (Download files from the NIRC2 Distortion wiki: https://github.com/jluastro/nirc2_distortion/wiki)
        nirc2_X_distortion.fits 
        nirc2_Y_distortion.fits
    Input: in_frame = (string) The name of a single FITS frame to undistort
    Output: outimage = (2D numpy array) Undistorted image

    """

    hdr = fits.getheader(in_frame,ignore_missing_end=True)
    # hdr= fits.open(in_frame+'.fits',ignore_missing_end=True)[0].header()

    imgsizeX = float(hdr['NAXIS1'])
    imgsizeY = float(hdr['NAXIS2'])
    parang=hdr['PARANG']
    if (imgsizeX >= imgsizeY): imgsize = imgsizeX
    else: imgsize = imgsizeY

    inimage = fits.getdata(in_frame,ignore_missing_end=True)
    x_dist = fits.getdata(distXgeoim)
    y_dist = fits.getdata(distYgeoim)
    gridx,gridy = np.meshgrid(np.arange(imgsize),np.arange(imgsize))
    gridx -= x_dist
    gridy -= y_dist

    outimage = ndi.map_coordinates(inimage, [gridy,gridx])
    dark=fits.getdata('mean_dark.fits',ignore_missing_end=True)
    outimage = subtract_dark(outimage,dark)

    if save: 
        hdu = fits.PrimaryHDU()
        hdu.data = outimage
        hdu.header = hdr 
        if out == '':
            out = '_dewarp.fits' #in_frame[:-5]
        hdu.writeto(out, overwrite=True)

    return(outimage,parang)

def process_pre_upgrade(path,savepath):
    '''
    Function to undistort a bunch of images. Toggle distortion solution based on the date
    of the image. 
    '''
    files=glob.glob(path)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    for i,fname in enumerate(files):
        undistort(fname,save=True,out=os.path.join(savepath,f'{i+1}.fits'))

def subtract_dark(img,dark):
    '''
    Subtract dark frame from image. Both img and dark are
    2D
    '''
    return img-dark

def get_counts():
    '''
    Get the most common bright pixels among a series of FITS files. Return these pixels
    and their frequencies
    '''
    fnums=[i+1 for i in range(268)]
    imgs=[f'03-20-2016/undistorted/{i}.fits' for i in fnums]
    brightest_overall=[]
    for i in imgs:
        img=fits.getdata(i)
        brightest=find_brightest(img,coords=False)
        brightest_overall.append(brightest)

    unique,counts=np.unique(brightest_overall,return_counts=True)
    inds=np.argsort(-1*counts)
    counts=counts[inds]
    unique=unique[inds]
    img=fits.getdata(imgs[0])
    new_unique=[]
    for i, val in enumerate(unique):
        new_unique.append(np.unravel_index(val,img.shape))
    
    return new_unique, counts


def fit_gauss(img,guess):
    '''
    Function that uses the pyklip function jason sent to fit a gaussian to the 
    secondary when given a guess for its location, which is sourced from guess_secondary
    '''
    xguess,yguess=guess[0],guess[1]
    peak,fwhm,xpos,ypos=pyklip.fakes.gaussfit2d(img,xguess,yguess,searchrad=6,refinefit=True)
    return (xpos,ypos), peak


def find_brightest(img,n=10,coords=True):
    '''
    Function that finds the n brightest pixels in an image array and returns their coordinates

    Args:
        img (2D array): Image array 
        n (int)
    Returns:
        brightest (2D array): Array consisting of the coordinates for the n brightest pixels 
                              in the image

    '''
    brightest=[]
    if coords:
        for i in range(n):
            pos=np.unravel_index(img.argmax(),img.shape)
            coords=np.array([pos[1],pos[0]])
            brightest.append(coords)
            img[coords[1],coords[0]]=0
    else:
        for i in range(n):
            pos=img.argmax()
            brightest.append(pos)

    return np.array(brightest)


def find_secondary(imgs):
    '''
    The function finds the location of the secondary by first using one of the 
    brightest pixels in the image as a guess. This is then used as the initial
    guess for pyklip.fakes.gaussfit2d, which fits a gaussian to the secondary.

    args:
        imgs (List): A list of filenames containing all the frames for which to 
                     determine the secondary's position
    
    returns:

        positions     (2D np.array): array containing the guess for each image
       
        gauss_centres (2D np.array): array of the locations as determined by 
                                     the gaussian fit.  
    '''
    positions=[]
    gauss_centres=[]
    for i,fname in enumerate(imgs):
        img=fits.getdata(fname,ignore_missing_end=True)
        # mask hot pixel
        img[1,62]=0
        
        # find the 10 brightest pixels in the image
        brightest_pixels=find_brightest(img)
        pos=brightest_pixels[0]
        if i==0:
            positions.append(pos)
            gauss_mean=pyklip.fakes.gaussfit2d(img,pos[0],pos[1],searchrad=3)[-2:]
            gauss_centres.append(gauss_mean)
        else:

            # there's a few abnormally bright pixels in the images, which can mess with the
            # brightest pixel approach for some of the frames. The next line finds the pixels
            # in brightest_pixels that are close to the position of the secondary determined
            # for the previous frame. In doing so, it avoids accidentally choosing a random
            # hot pixel as the guess for the secondary's location. 
            good_pixels=np.argwhere(np.linalg.norm(brightest_pixels-positions[i-1],axis=1)<=(np.sqrt(5))).flatten()
            try:
                
                # choose the brightest of the pixels that meet the threshold above as the guess
                candidates=brightest_pixels[good_pixels]
                # print(candidates)
                positions.append(pos)
            except IndexError:
                print(fname)
        
            # fit a 2D gaussian to the data, taking pos as the initial centre guess
            for guess in candidates:
                gauss_mean=pyklip.fakes.gaussfit2d(img,guess[0],guess[1],searchrad=3)[-2:]
                if np.linalg.norm(np.array(gauss_mean)-guess)<=2:
                    pos=gauss_mean
                else:
                    break       
            # print(pos)
            gauss_centres.append(pos)
            # print(f'{i} done!')
    return np.array(positions), np.array(gauss_centres)

def find_primary(img):
    '''
    Find location of the primary via the radon transform
    '''
    # assuming that the position of the primary doesnt move too much over the 
    # different frames, we use a constant guess for its location
    xguess=698
    yguess=471
    x_cen,y_cen,cf=rad.searchCenter(img, xguess, yguess, 80, size_cost=5 ,theta = np.linspace(0,360,60),output_cost=True)
    centre=np.array([x_cen,y_cen])
    return centre,cf


def calc():
    '''
    Driver function for the whole astrometry calculation
    '''
    ### CODE TO DETERMINE THE POSITION OF THE SECONDARY IN EACH FRAME ###
    
    fnums=[i+1 for i in range(150)]
    
    # # get rid of sus frames
    # sus_frames=[68,69,70,71,72,251,252,253]
    # for frame in sus_frames:
    #     fnums.remove(frame)
    
    # create list of the corresponding file names
    imgs=[f'03-20-2016/undistorted/{i}.fits' for i in fnums]
    
    # guess position of the secondary for each frame
    positions,gauss_centres=find_secondary(imgs)
    

    ### CODE TO DETERMINE THE POSITION OF THE PRIMARY AND SEP/PA ###
    sep_arr=[]
    ang_arr=[]
    for i,fname in enumerate(imgs):

        img=fits.getdata(fname,ignore_missing_end=True)
        
        #currently just using the positions array (ie the guesses) for the secondary position
        secondary_pos=positions[i]
        # mask in a box around the secondary
        img[ secondary_pos[0]-15: secondary_pos[0] + 15 , secondary_pos[1]-15: secondary_pos[1]+15  ]=0
        # find the primary using the radon transform
        primary_pos,cf=find_primary(img)
        # these are the nirc2 constants 
        nirc2_platescale = 9.971
        nirc2_dnorth = -0.262

        # get detector separation
        offset=secondary_pos-primary_pos

        # convert to sep/pa
        sep=np.linalg.norm(offset)
        sep*=nirc2_platescale

        hdr = fits.getheader(fname,ignore_missing_end=True)
        parang=hdr['PARANG']
        rotposn=hdr['ROTPOSN']
        instangl=hdr['INSTANGL']
        ang=(np.degrees(np.arctan2([-1*offset[0]],[offset[1]]))+parang+nirc2_dnorth)[0]+rotposn-instangl

        # add to arrays
        sep_arr.append(sep)
        ang_arr.append(ang)

        print(f'{i+1} done!')
    
    sep_avg=np.mean(sep_arr)
    ang_avg=np.mean(ang_arr)

    return (sep_avg,ang_avg), (np.array(sep_arr),np.array(ang_arr))
    
def plot_cost(cf):
    fig=plt.imshow(cf,cmap='inferno')
    plt.xticks(np.arange(12,step=2),[str(2*i+693) for i in np.arange(6)])
    plt.xlabel('Pixel X coordinate')
    plt.yticks(np.arange(12,step=2)[::-1],[str(2*i+466) for i in np.arange(6)])
    plt.ylabel('Pixel Y coordinate')
    plt.title('Intensity Map of the Radon Transform')
    plt.colorbar()
    plt.savefig('CostFunction')
    plt.close()



if __name__ == "__main__":

    # fnums=[i+1 for i in range(150)]

    # # #get rid of sus frames
    # # sus_frames=[68,69,70,71,72,251,252,253]
    # # for frame in sus_frames:
    # #     fnums.remove(frame)
    
    # # create list of the corresponding file names
    # imgs=[f'04-19-2013/undistorted/{i}.fits' for i in fnums]
    # # guess position of the secondary for each frame
    # positions,gauss_centres=find_secondary(imgs)
    # for i in range(len(fnums)):
    #     print(f'Guess: {positions[i]}')
    #     print(f'Gauss: {gauss_centres[i]}')
    #     print()
    # print(gauss_centres-positions)
    # gauss_centres[np.where(np.linalg.norm(gauss_centres-positions,axis=1)>=np.sqrt(10))]=0
    # print(gauss_centres)

    # fname='03-20-2016/undistorted/7.fits'
    # img=fits.getdata(fname,ignore_missing_end=True)
    # print(img[1,62],img[62,1])
    # print(find_brightest(img))
    # (sep_info),(ang_info)=calc()
    # print(sep_info)
    # print(ang_info)


    res=calc()
    pdb.set_trace()

    # img=fits.getdata('03-20-2016/undistorted/2.fits')
    # dark=fits.getdata('mean_dark.fits')
    # res=scipy.ndimage.median_filter(img-dark,size=5)
    # hdu = fits.PrimaryHDU()
    # hdu.data = res
    # savename='filtered.fits'
    # hdu.writeto(savename, overwrite=True)



