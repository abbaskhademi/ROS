#!/usr/bin/env python3
"""Extract measured positions, SNR, and subband gains from DICHASUS data.

The script is only a data conversion utility. The optimization experiments are
implemented in Julia and read the resulting compressed NPZ file.
"""

from __future__ import annotations

import argparse
import csv
import struct
from pathlib import Path

import numpy as np
from tfrecord.reader import tfrecord_loader


def _varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7


def _fields(data: bytes):
    offset = 0
    while offset < len(data):
        tag, offset = _varint(data, offset)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            value, offset = _varint(data, offset)
        elif wire == 1:
            value = data[offset : offset + 8]
            offset += 8
        elif wire == 2:
            length, offset = _varint(data, offset)
            value = data[offset : offset + length]
            offset += length
        elif wire == 5:
            value = data[offset : offset + 4]
            offset += 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        yield number, wire, value


def _shape(shape_message: bytes) -> tuple[int, ...]:
    dimensions = []
    for number, wire, value in _fields(shape_message):
        if number != 2 or wire != 2:
            continue
        for dim_number, dim_wire, dim_value in _fields(value):
            if dim_number == 1 and dim_wire == 0:
                dimensions.append(int(dim_value))
    return tuple(dimensions)


def _tensor(serialized: bytes) -> np.ndarray:
    dtype_code = None
    dimensions = None
    content = None
    for number, wire, value in _fields(serialized):
        if number == 1 and wire == 0:
            dtype_code = int(value)
        elif number == 2 and wire == 2:
            dimensions = _shape(value)
        elif number == 4 and wire == 2:
            content = value
    dtypes = {1: "<f4", 2: "<f8"}
    if dtype_code not in dtypes or dimensions is None or content is None:
        raise ValueError("Unsupported serialized TensorProto")
    return np.frombuffer(content, dtype=dtypes[dtype_code]).reshape(dimensions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path, help="Compressed NPZ output")
    parser.add_argument("--snr-csv", type=Path)
    parser.add_argument("--subbands", type=int, default=32)
    args = parser.parse_args()

    times = []
    positions = []
    snr_rows = []
    power_rows = []
    antenna_count = None
    for record_id, example in enumerate(tfrecord_loader(str(args.source), None), start=1):
        snr = _tensor(example["snr"]).astype(float)
        position = _tensor(example["pos-tachy"]).astype(float)
        csi = _tensor(example["csi"]).astype(np.float32)
        if antenna_count is None:
            antenna_count = len(snr)
        if (
            len(snr) != antenna_count
            or len(position) != 3
            or csi.shape != (antenna_count, 1024, 2)
            or 1024 % args.subbands != 0
        ):
            raise ValueError("Unexpected DICHASUS record dimensions")
        block = 1024 // args.subbands
        power = np.sum(csi * csi, axis=2).reshape(antenna_count, args.subbands, block)
        power = np.mean(power, axis=2)
        times.append(float(example["time"][0]))
        positions.append(position)
        snr_rows.append(snr)
        power_rows.append(power)

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.destination,
        time_s=np.asarray(times, dtype=np.float64),
        position_m=np.asarray(positions, dtype=np.float64),
        snr_db=np.asarray(snr_rows, dtype=np.float32),
        channel_power=np.asarray(power_rows, dtype=np.float32),
        subband_count=np.asarray([args.subbands], dtype=np.int32),
    )

    if args.snr_csv is not None:
        args.snr_csv.parent.mkdir(parents=True, exist_ok=True)
        header = ["record_id", "time_s", "x_m", "y_m", "z_m"]
        header.extend(f"snr_db_{index:02d}" for index in range(1, antenna_count + 1))
        with args.snr_csv.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            for record_id, (time_value, position, snr) in enumerate(
                zip(times, positions, snr_rows), start=1
            ):
                writer.writerow([record_id, time_value, *position.tolist(), *snr.tolist()])


if __name__ == "__main__":
    main()
