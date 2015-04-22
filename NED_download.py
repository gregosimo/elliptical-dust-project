import os

from astroquery.ned import Ned
from requests.exceptions import ContentDecodingError

import queries
import photometry as phot

gil_de_paz_bibcode = "2007ApJS..173..185G"

def filter_by_bibcode(urllist, bibcode):
    '''Takes a list of URLs and returns those containing the bibcode.'''
    return filter(lambda url: bibcode in url, urllist)

def download_NED_GALEX_images(BASEDIR, objlist):
    '''Downloads GALEX images via NED.

    The objects to be downloaded are provided in objlist. They will then be
    organized under a naming hierarchy of BASEDIR/OBJNAME/. Both the NUV and
    FUV images should be downloaded.'''
    for name in objlist:
        galaxydir = phot.change_to_galaxy_dir(BASEDIR, name)
        fullurllist = Ned.get_image_list(name)

        gdpurls = filter_by_bibcode(fullurllist, gil_de_paz_bibcode)

        for url in gdpurls:
            if "NUV" in url:
                filename = "{0}-nd-int.fits".format(
                    phot.object_name_to_dir(name))
            elif "FUV" in url:
                filename = "{0}-fd-int.fits".format(
                    phot.object_name_to_dir(name))
            #try:
            #    queries.download_image(galaxydir, url, filename=filename)
            # For some reason, there are no content headers when downloading
            # file name, and it poops out when it tries to read the header.
            # However, the file still downloads, so we'll just keep powering
            # on!
            #except ContentDecodingError:
            #    pass
            queries.download_image(galaxydir, url, filename=filename)
