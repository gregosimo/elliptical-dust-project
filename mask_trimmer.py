import argparse
import cmd

import masks

# I should probably have a more object-oriented method of loading up this CMD
# instance, such as passing it inputmask and outputmask, but I'm taking a
# short-cut and will exploit the fact that they are global variables for now. If
# somebody in the future looks at this, understand that this is REALLY bad, and
# that I'm actually pretty ashamed of it.
class MaskCMD(cmd.Cmd):
    def do_cut(num):
        masks.remove_segment(segment, num)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("segment", help="The segment to be trimmed.")
    parser.add_argument("mask", help="The destination of the trimmed mask.")
    args = parser.parse_args()
    inputmask = args.segment
    outputmask = args.mask

    hdulist = fits.open(inputmask) 
    segment = hdulist[0].data
    mycmd = MaskCMD()
    mycmd.cmdloop()
    newhdu = fits.PrimaryHDU(segment)
    try:
        newhdu.writeto(outputmask)
    except IOError:
        os.remove(outputmask)
        newhdu.writeto(outputmask)
    



    print "Load the image and 
