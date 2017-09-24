from .parse_corsika import (
    parse_run_header,
    parse_event_header
)


def corsika_read(f, raw):
    if raw is True:
        pos = f.tell() - 4

        if pos % 22932 == 0:
            f.read(8)

    return f.read(4*273)


def read_corsika_headers(inputfile):
    byte_data = corsika_read(inputfile, raw=False)

    if(byte_data[0:4] == b'RUNH'):
        fortran_raw = False
    else:
        fortran_raw = True
        inputfile.seek(4)
        byte_data = inputfile.read(4*273)

    run_header = parse_run_header(byte_data)

    # Read Eventheader
    event_header_data = b''
    while True:
        byte_data = corsika_read(inputfile, fortran_raw)

        # No more Shower in file
        if byte_data[0:4] == b'RUNE':
            break

        if byte_data[0:4] != b'EVTH':
            continue

        event_header_data += byte_data

    event_headers = parse_event_header(event_header_data)

    return run_header, event_headers
