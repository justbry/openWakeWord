"""
Repairs a partially-downloaded .npy file by updating the shape header to match
the actual data present. Useful when a large download is interrupted mid-file.

Usage:
    python fix_truncated_npy.py openwakeword_features_ACAV100M_2000_hrs_16bit.npy
"""

import sys
import struct
import os
import numpy as np


def fix_truncated_npy(path: str) -> None:
    file_size = os.path.getsize(path)

    with open(path, "rb") as f:
        magic = f.read(6)
        assert magic == b"\x93NUMPY", "Not a valid .npy file"
        major = ord(f.read(1))
        minor = ord(f.read(1))
        if major == 1:
            header_len = struct.unpack("<H", f.read(2))[0]
        else:
            header_len = struct.unpack("<I", f.read(4))[0]
        header_bytes = f.read(header_len)

    header_offset = 6 + 2 + (2 if major == 1 else 4)
    header_total = header_offset + header_len

    # Parse shape and dtype from header
    header_str = header_bytes.decode("latin1")
    import ast
    header_dict = ast.literal_eval(header_str.strip().rstrip(","))
    original_shape = header_dict["shape"]
    dtype = np.dtype(header_dict["descr"])
    row_bytes = int(np.prod(original_shape[1:])) * dtype.itemsize

    data_bytes = file_size - header_total
    n_rows = data_bytes // row_bytes
    leftover = data_bytes % row_bytes

    print(f"File:          {path}")
    print(f"Original shape: {original_shape}")
    print(f"Row size:       {row_bytes} bytes")
    print(f"Available rows: {n_rows:,} of {original_shape[0]:,}")
    print(f"Leftover bytes: {leftover} (will be trimmed)")

    if n_rows == original_shape[0]:
        print("File is complete - no repair needed.")
        return

    # Build new header with corrected shape
    new_shape = (n_rows,) + original_shape[1:]
    new_header_dict = dict(header_dict)
    new_header_dict["shape"] = new_shape
    new_header_str = repr(new_header_dict) + " "
    # Pad to multiple of 64 bytes (numpy convention)
    padding = 64 - ((header_offset + len(new_header_str) + 1) % 64)
    new_header_str = new_header_str + " " * padding + "\n"
    new_header_bytes = new_header_str.encode("latin1")
    new_header_len = len(new_header_bytes)

    # Rewrite header length and header in place
    good_size = header_total + n_rows * row_bytes

    with open(path, "r+b") as f:
        f.seek(header_offset - (2 if major == 1 else 4))
        if major == 1:
            if new_header_len > 65535:
                raise ValueError("New header too large for format 1.0 - save as format 2.0")
            f.write(struct.pack("<H", new_header_len))
        else:
            f.write(struct.pack("<I", new_header_len))
        f.write(new_header_bytes)

    os.truncate(path, good_size)
    print(f"\nRepaired. New shape: {new_shape}, file size: {good_size / 1024**3:.2f} GB")

    # Verify
    d = np.load(path, mmap_mode="r")
    assert d.shape == new_shape, f"Verification failed: {d.shape} != {new_shape}"
    print("Verification passed.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <file.npy>")
        sys.exit(1)
    fix_truncated_npy(sys.argv[1])
