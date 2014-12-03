import argparse
import os
import shutil
import cmd

from astropy.io import fits
from ds9 import ds9

import masks

# I should probably have a more object-oriented method of loading up this CMD
# instance, such as passing it inputmask and outputmask, but I'm taking a
# short-cut and will exploit the fact that they are global variables for now. If
# somebody in the future looks at this, understand that this is REALLY bad, and
# that I'm actually pretty ashamed of it.
class MaskCMD(cmd.Cmd):
    def __init__(self, inputimage, outputimage, regionfile):
        self.hdulist = fits.open(inputimage) 
        self.target = outputimage
        self.ds9 = ds9()
        regionhandle = open(regionfile)
        self.region = ''.join(regionhandle.readlines())
        regionhandle.close()
        cmd.Cmd.__init__(self)
        self.load_image()

    def do_cut(self, num):
        '''Removes the segment with the specified number.'''
        self.hdulist[0].data = masks.remove_segment(self.hdulist[0].data, 
                int(num))
        self.load_image()

    def load_image(self):
        '''Reloads the ds9 window.'''
        self.ds9.set_pyfits(self.hdulist)
        self.ds9.set('regions', self.region)

    def do_save(self, arg):
        '''Saves the image and quits.'''
        close_fits(self.hdulist, self.target)
        return True

    def default(self, arg):
        self.do_cut(arg)

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
    



