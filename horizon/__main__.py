'''Command line interface: python -m horizon.'''

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from horizon.benchmark import load_heightmap, run_benchmark
from horizon.codec import HEADER_SIZE, MAX_RAW_BYTES, METHODS, decode, encode


def _exists_error(path):
    return FileExistsError(f'refusing to overwrite existing file: {path}')


def _require_free(path):
    if os.path.exists(path):
        raise _exists_error(path)


def _write_new(path, write):
    '''Create path exclusively, write, and remove the partial file on failure.'''
    try:
        fh = open(path, 'xb')
    except FileExistsError:
        raise _exists_error(path) from None
    try:
        with fh:
            write(fh)
    except BaseException:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise


def _pack(args):
    _require_free(args.output)
    arr = load_heightmap(args.input)
    blob = encode(arr, method=args.method)
    back = decode(blob)
    if (
        back.shape != arr.shape
        or back.dtype != arr.dtype
        or back.tobytes() != np.ascontiguousarray(arr).tobytes()
    ):
        raise RuntimeError('round-trip verification failed')
    _write_new(args.output, lambda fh: fh.write(blob))
    print(f'packed {arr.shape[0]}x{arr.shape[1]} -> {len(blob)} bytes ({args.method})')
    return 0


def _unpack(args):
    _require_free(args.output)
    with open(args.input, 'rb') as fh:
        # Current codecs cannot expand uint16 input to twice its raw size.
        limit = 2 * MAX_RAW_BYTES + HEADER_SIZE
        data = fh.read(limit + 1)
    if len(data) > limit:
        raise ValueError('container exceeds the input size limit')
    arr = decode(data)
    _write_new(args.output, lambda fh: np.save(fh, arr, allow_pickle=False))
    print(f'unpacked {arr.shape[0]}x{arr.shape[1]} uint16')
    return 0


def _benchmark(args):
    if args.output:
        _require_free(args.output)
    report = run_benchmark(
        size=args.size, repeats=args.repeats, inputs=args.input or ()
    )
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        _write_new(args.output, lambda fh: fh.write(text.encode('utf-8')))
    else:
        sys.stdout.write(text)
    return 0


def _build_parser():
    parser = argparse.ArgumentParser(prog='python -m horizon')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('pack', help='compress a uint16 .npy heightmap')
    p.add_argument('input')
    p.add_argument('output')
    p.add_argument('--method', choices=METHODS, default='delta_zstd')
    p.set_defaults(func=_pack)

    u = sub.add_parser('unpack', help='restore a .npy heightmap from a container')
    u.add_argument('input')
    u.add_argument('output')
    u.set_defaults(func=_unpack)

    b = sub.add_parser('benchmark', help='run the codec benchmark')
    b.add_argument('--input', action='append', help='real .npy map (repeatable)')
    b.add_argument('--size', type=int, default=256)
    b.add_argument('--repeats', type=int, default=30)
    b.add_argument('--output', help='write JSON report here (must not exist)')
    b.set_defaults(func=_benchmark)
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
