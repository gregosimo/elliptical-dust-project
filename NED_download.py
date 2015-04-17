import os

from astroquery.ned import Ned

import queries

gil_de_paz_bibcode = "2007ApJS..173..185G "

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
        gdpurls = filter_by_bibcode(gil_de_paz_bibcode)

        for url in gdpurls:
            queries.download_image_test(galaxydir, url)
