

from astropy.table import Table

def create_upload_file(ids, ras, decs, output):
    '''Creates a file that can be uploaded to MAST.

    This file will contain Object names, RAs, and Decs as a comma-separated
    list. The file will be saved to output, where it can manually be uploaded to
    the GALEX MAST database.
    '''
    relevantTable = Table([ids, ras, decs], names=("ID", "RA", "DEC"))
    relevantTable.write(output, format="ascii.csv")
