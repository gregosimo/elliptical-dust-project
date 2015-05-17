import argparse
import os
import shutil
import cmd

from astropy.io import fits
from ds9 import ds9
import numpy as np

import masks
import photometry as phot

class MaskCMD(cmd.Cmd):
    def __init__(self, inputimage, outputimage, origimage, regionfile):
        # Set up the environment.
        self.target = outputimage
        self.ds9 = ds9()
        # Set up things needed to display the mask
        self.maskhdulist = fits.open(inputimage) 
        regionhandle = open(regionfile)
        self.region = ''.join(regionhandle.readlines())
        regionhandle.close()
        # Set up things needed to display the original image.
        self.imagehdulist = fits.open(origimage)
        # Set up quantities for masking routines
        self.masknum = np.amax(self.maskhdulist[0].data)+1
        cmd.Cmd.__init__(self)
        self.load_image()
        self.load_mask()

    def do_cut(self, num):
        '''Removes the segment with the specified number.'''
        try:
            num = int(num)
        except ValueError:
            print "Could not parse number that was returned. Try again."
            return
        self.maskhdulist[0].data = masks.remove_segment(
            self.maskhdulist[0].data, num)
        return self.load_mask()

    def do_c(self, num):
        '''alias for do_cut'''
        return self.do_cut(num)
    
    def do_mask(self, arg):
        '''Masks a point source at the given coordinate'''

        coords = arg.split()
        try:
            center = int(coords[0]), int(coords[1])
        except IndexError:
            print "Did not understand masking command. Try again."
            return
        masks.mask_point_source("W1", self.maskhdulist[0].data, center, 
                                self.masknum)
        self.masknum = self.masknum + 1
        return self.load_mask()

    def do_reload(self, arg):
        '''Reloads the argument given.

        If 'mask' is given, the mask is reloaded. If 'int' is given, the 
        intensity image is reloaded.
        '''
        target = arg
        if target.lower() is "mask":
            return self.load_mask()
        elif target.lower() is "int":
            return self.load_image()
        else:
            print "Either type 'int' or 'mask' to reload."

    def do_m(self, arg):
        '''Alias for do_mask'''
        return self.do_mask(arg)

    def load_mask(self, frame=1):
        '''Refreshes the view of the mask in the given frame.'''
        self.ds9.set("frame {0:d}".format(frame))
        self.ds9.set_pyfits(self.maskhdulist)
        self.ds9.set('regions', self.region)

    def load_image(self, frame=2, maskframe=1):
        '''Loads the image into frame 2.'''
        self.ds9.set("frame {0:d}".format(frame))
        self.ds9.set_pyfits(self.imagehdulist)
        self.ds9.set("frame {0:d}".format(maskframe))

    def do_save(self, arg):
        '''Saves the image and quits.'''
        close_fits(self.maskhdulist, self.target)
        return True

    def default(self, arg):
        num = arg.count(" ")
        if num == 0:
            return self.do_cut(arg)
        elif num == 1:
            return self.do_mask(arg)
        else:
            print "Did not understand command that was given. Try again."
            return

    def do_quit(self, arg):
        '''Quits without saving.'''
        return True
    def do_q(self, arg):
        return True

def parseregion(regionstring):
    '''Takes a region file and parses information from it.'''
    # I'm assuming there's only ellipses right now. I can make changes (or use
    # pyregion), when things get more complicated.
    coord, shape = regionstring.split("\n")

def close_fits(header, filename):
    header.writeto(filename, clobber=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("segment", help="The segment to be trimmed.")
    parser.add_argument("mask", help="The destination of the trimmed mask.")
    args = parser.parse_args()
    inputmask = args.segment
    outputmask = args.mask

    print """The status of the segmentation image will be displayed in a DS9
    instance. It will be continually updated by your edits."""
    mycmd = MaskCMD(inputmask, outputmask)
    mycmd.cmdloop()
    



