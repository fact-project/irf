import struct
import numpy as np

from eventio.iact.parse_corsika_data import (
    parse_corsika_run_header,
    parse_corsika_event_header
)


def corsika_read(f, raw):
    if raw is True:
        pos = f.tell() - 4

        if pos % 22932 == 0:
            f.read(8)

    return f.read(4*273)


struct_header = struct.Struct('273f')
struct_particle = struct.Struct('273f')  # 39 times, if empty rest is 0


def read_corsika_headers(inputfile):
    data = {}

    byte_data = corsika_read(inputfile, raw=False)

    if(byte_data[0:4] == b'RUNH'):
        fortran_raw = False
    else:
        fortran_raw = True
        inputfile.seek(4)
        byte_data = inputfile.read(4*273)

    RUNH = struct_header.unpack(byte_data)

    data['run_header'] = parse_corsika_run_header(np.array(RUNH))
    data['event_headers'] = []

    # Read Eventheader
    counter = 0
    while True:
        byte_data = corsika_read(inputfile, fortran_raw)

        # No more Shower in file
        if(byte_data[0:4] == b'RUNE'):
            break

        EVTH = struct_header.unpack(byte_data)

        data['event_headers'].append(
            parse_corsika_event_header(np.array(EVTH))
        )

        while True:
            byte_data = corsika_read(inputfile, fortran_raw)

            if(byte_data[0:4] == b'EVTE'):
                break

        counter = counter + 1
    return data
