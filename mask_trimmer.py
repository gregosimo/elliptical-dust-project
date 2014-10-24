import argparse
import os
import shutil
import cmd

from astropy.io import fits

import masks

# I should probably have a more object-oriented method of loading up this CMD
# instance, such as passing it inputmask and outputmask, but I'm taking a
# short-cut and will exploit the fact that they are global variables for now. If
# somebody in the future looks at this, understand that this is REALLY bad, and
# that I'm actually pretty ashamed of it.
class MaskCMD(cmd.Cmd):
    def do_cut(self, num):
        self.segment = masks.remove_segment(self.segment, int(num))

    def do_refresh(self, args):
        close_fits(self.segment, constructedmask)
    def do_save(self, arg):
        close_fits(self.segment, constructedmask)
        masks.normalize_segmentation_map(constructedmask, outputmask)
        return True

    def default(self, arg):
        self.do_cut(arg)

    def do_quit(self, arg):
        return True
    def do_q(self, arg):
        return True
        
def close_fits(data, filename):
    newhdu = fits.PrimaryHDU(data)
    try:
        newhdu.writeto(filename)
    except IOError:
        os.remove(filename)
        newhdu.writeto(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("segment", help="The segment to be trimmed.")
    parser.add_argument("mask", help="The destination of the trimmed mask.")
    parser.add_argument("--temp", help="Where to hold the temporary file.",
        default="segtemp.fits")
    args = parser.parse_args()
    inputmask = args.segment
    constructedmask = args.temp
    outputmask = args.mask

    shutil.copyfile(inputmask, constructedmask)
    print """Copied input file over to {0}. You may now open this file to see 
    the progress of the trims.""".format(os.path.abspath(constructedmask))
    hdulist = fits.open(inputmask) 
    segment = hdulist[0].data
    mycmd = MaskCMD()
    mycmd.segment = hdulist[0].data
    mycmd.cmdloop()
    



