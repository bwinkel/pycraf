#!/usr/bin/env python
# -*- coding: utf-8 -*-


# You have to create a temporary file, from the viewpano server:
# /bin/rm *.zip newpano.txt; for i in `cat pano.txt`; do echo "$i" >> newpano.txt; wget http://viewfinderpanoramas.org/dem3/${i}; unzip -l $i | grep '.hg' >> newpano.txt; done

import numpy as np
from astropy.utils.data import get_pkg_data_filename


with open(get_pkg_data_filename('newpano.txt'), 'r') as f:
    lines = f.readlines()


entries = []
for l in lines:

    l = l.strip()
    if '.zip' in l:
        zip_file = l
    else:
        tmp = l.split()[-1]
        super_tile, tile = tmp.split('/')
        entries.append((zip_file, super_tile, tile))

entries = np.array(entries, dtype=np.dtype([
    ('zipfile', np.str, 16),
    ('supertile', np.str, 16),
    ('tile', np.str, 16)
    ]))

np.save('viewpano', entries)
