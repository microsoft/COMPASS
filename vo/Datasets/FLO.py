from __future__ import print_function

import numpy as np

def read_flo_bytes(bio):
    """
    bio is an io.BytesIO object.
    """
    try:
      buffer = bio.getvalue() # python2
    except:
      buffer = bio.getbuffer() # python3

    magic = np.frombuffer( buffer, dtype=np.float32, count=1 )

    if ( 202021.25 != magic ):
        print('Matic number incorrect. Expect 202021.25, get {}. Invalid .flo file.'.format( \
            magic ))

        return None
    
    W = np.frombuffer( buffer, dtype=np.int32, count=1, offset=4 )
    H = np.frombuffer( buffer, dtype=np.int32, count=1, offset=8 )

    W = int(W)
    H = int(H)

    data = np.frombuffer( buffer, dtype=np.float32, \
        count=2*W*H, offset=12 )

    return np.resize( data, ( H, W, 2 ) )

def read_flo(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))