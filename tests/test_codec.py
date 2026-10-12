import hashlib
import unittest
import zlib
from unittest import mock

import lz4.frame
import numpy as np
import zstandard

from horizon import codec
from horizon.codec import METHODS, decode, encode


def _random_map(seed, shape):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 65536, size=shape, dtype=np.uint16).astype("<u2")


def _smooth_map(shape):
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    return (1000 + 7 * x + 3 * y + (x * y) % 11).astype("<u2")


def _reference_undelta(delta):
    """Independent per-row prefix sum mod 65536 in int64."""
    return (np.cumsum(delta.astype(np.int64), axis=1) % 65536).astype("<u2")


def _with_crc(header):
    return header + codec._CRC.pack(zlib.crc32(header))


def _patch_header(blob, **fields):
    """Rewrite header fields and recompute the header CRC."""
    names = ("magic", "version", "method", "predictor", "reserved",
             "height", "width", "plen", "digest")
    values = dict(zip(names, codec._HEADER.unpack_from(blob)))
    values.update(fields)
    header = codec._HEADER.pack(*(values[n] for n in names))
    return _with_crc(header) + blob[codec.HEADER_SIZE:]


def _container(method, height, width, payload, raw=b""):
    header = codec._HEADER.pack(
        codec.MAGIC, codec.VERSION, METHODS.index(method),
        codec._PREDICTORS[method], 0, height, width, len(payload),
        hashlib.sha256(raw).digest(),
    )
    return _with_crc(header) + payload


class RoundTripTests(unittest.TestCase):
    def assertExact(self, original, method):
        blob = encode(original, method)
        self.assertIsInstance(blob, bytes)
        out = decode(blob)
        self.assertEqual(out.dtype.str, "<u2")
        self.assertEqual(out.shape, original.shape)
        self.assertTrue(out.flags.c_contiguous)
        self.assertEqual(out.tobytes(), np.ascontiguousarray(original).tobytes())
        return out

    def test_methods_constant(self):
        self.assertEqual(METHODS, ("zstd", "lz4", "delta_zstd"))

    def test_default_method_is_delta_zstd(self):
        arr = _smooth_map((5, 6))
        self.assertEqual(encode(arr), encode(arr, "delta_zstd"))

    def test_all_methods_various_maps(self):
        maps = [
            np.zeros((4, 4), dtype="<u2"),
            np.full((3, 5), 65535, dtype="<u2"),
            _smooth_map((17, 23)),
            _random_map(0, (31, 29)),
            _random_map(12345, (64, 64)),
            np.array([[7]], dtype="<u2"),
        ]
        for method in METHODS:
            for arr in maps:
                with self.subTest(method=method, shape=arr.shape):
                    self.assertExact(arr, method)

    def test_edge_values_and_wraparound(self):
        arr = np.array(
            [[0, 65535, 0, 1, 65535, 32768, 32767, 0],
             [65535, 0, 65535, 65534, 1, 0, 65535, 65535]],
            dtype="<u2",
        )
        for method in METHODS:
            with self.subTest(method=method):
                self.assertExact(arr, method)

    def test_row_column_noncontiguous(self):
        base = _random_map(7, (40, 30))
        cases = {
            "row": base[:1, :],
            "column": base[:, :1],
            "strided": base[::3, ::2],
            "transposed": base.T,
            "fortran": np.asfortranarray(base),
            "reversed": base[::-1, ::-1],
        }
        for method in METHODS:
            for name, arr in cases.items():
                with self.subTest(method=method, case=name):
                    self.assertExact(arr, method)

    def test_native_uint16_accepted_on_little_endian(self):
        arr = np.arange(12, dtype=np.uint16).reshape(3, 4)
        if np.dtype(np.uint16) == np.dtype("<u2"):
            self.assertExact(arr, "lz4")

    def test_input_not_mutated(self):
        arr = _random_map(3, (9, 9))
        copy = arr.copy()
        encode(arr, "delta_zstd")
        np.testing.assert_array_equal(arr, copy)

    def test_decoded_array_is_writable(self):
        for method in METHODS:
            out = decode(encode(_smooth_map((3, 3)), method))
            out[0, 0] = 1

    def test_deterministic(self):
        arr = _random_map(99, (16, 16))
        for method in METHODS:
            self.assertEqual(encode(arr, method), encode(arr, method))

    def test_delta_predictor_matches_spec(self):
        arr = _random_map(5, (6, 9))
        delta = codec._delta(arr)
        for r in range(arr.shape[0]):
            self.assertEqual(int(delta[r, 0]), int(arr[r, 0]))
            for c in range(1, arr.shape[1]):
                expected = (int(arr[r, c]) - int(arr[r, c - 1])) % 65536
                self.assertEqual(int(delta[r, c]), expected)
        np.testing.assert_array_equal(codec._undelta(delta), arr)

    def test_undelta_wraps_past_uint32_on_wide_rows(self):
        # Row prefix sums reach 131073 * 65535 > 2**32; output must still be
        # the prefix sum mod 65536.
        width = 131073
        delta = np.full((1, width), 65535, dtype="<u2")
        expected = ((-np.arange(1, width + 1, dtype=np.int64)) % 65536).astype("<u2")
        out = codec._undelta(delta)
        self.assertEqual(out.dtype.str, "<u2")
        self.assertTrue(out.flags.c_contiguous)
        np.testing.assert_array_equal(out[0], expected)
        np.testing.assert_array_equal(delta, np.full((1, width), 65535, dtype="<u2"))

    def test_wide_row_wraparound_roundtrip(self):
        # A descending ramp whose deltas are all 65535 (after the first sample).
        width = 131073
        arr = ((-np.arange(1, width + 1, dtype=np.int64)) % 65536).astype("<u2")
        arr = arr.reshape(1, width)
        self.assertTrue((codec._delta(arr) == 65535).all())
        out = self.assertExact(arr, "delta_zstd")
        self.assertTrue(out.flags.writeable)
        out[0, -1] = 0

    def assertUndelta(self, delta):
        before = delta.copy()
        out = codec._undelta(delta)
        self.assertEqual(out.dtype.str, "<u2")
        self.assertTrue(out.flags.c_contiguous)
        self.assertTrue(out.flags.writeable)
        np.testing.assert_array_equal(out, _reference_undelta(delta))
        np.testing.assert_array_equal(delta, before)

    def test_undelta_batch_boundaries_small_bound(self):
        # Bound of 12 samples: width 4 gives exact 3-row batches, width 5 gives
        # 2-row batches with 2 spare samples, width 12 is exactly one row, and
        # width 13 exceeds the bound (uint16 fallback). Rows of 65535 deltas
        # make any carry between rows visible; random rows catch skipped rows.
        with mock.patch.object(codec, "_UNDELTA_BATCH", 12):
            for height in (1, 2, 3, 4, 6, 7, 9):
                for width in (1, 4, 5, 12, 13):
                    for name, delta in (
                        ("max", np.full((height, width), 65535, dtype="<u2")),
                        ("random", _random_map(height * 100 + width, (height, width))),
                    ):
                        with self.subTest(height=height, width=width, data=name):
                            self.assertUndelta(delta)

    def test_undelta_multiple_batches_real_bound(self):
        # 256 rows per batch: 600 rows = two full batches plus a partial one.
        delta = _random_map(31, (600, 256))
        delta[255:258] = 65535  # rows straddling the first batch boundary
        self.assertUndelta(delta)
        self.assertUndelta(_random_map(32, (256, 256)))  # exactly one batch
        self.assertUndelta(_random_map(33, (257, 256)))  # one-row second batch

    def test_undelta_row_wider_than_bound_falls_back(self):
        width = codec._UNDELTA_BATCH + 1
        delta = _random_map(34, (3, width))
        delta[1] = 65535
        self.assertUndelta(delta)

    def test_undelta_batched_uint32_overflow(self):
        # Raise the bound so 131073-wide rows are batched two at a time; their
        # uint32 prefix sums pass 2**32 and must still wrap to mod 65536.
        width = 131073
        delta = np.full((3, width), 65535, dtype="<u2")
        delta[1, 0] = 1
        with mock.patch.object(codec, "_UNDELTA_BATCH", 2 * width):
            self.assertUndelta(delta)

    def test_decode_across_batches_small_bound(self):
        arr = _random_map(35, (11, 5))
        with mock.patch.object(codec, "_UNDELTA_BATCH", 12):
            out = self.assertExact(arr, "delta_zstd")
        self.assertTrue(out.flags.writeable)
        out[-1, -1] = 0

    def test_delta_zstd_bytes_match_reference_container(self):
        # Independently built container: pure-Python predictor, same zstd
        # settings. Guards the on-disk format and decoding of such blobs.
        arr = _random_map(77, (5, 11))
        rows = arr.tolist()
        delta = [[r[0]] + [(r[c] - r[c - 1]) % 65536 for c in range(1, len(r))]
                 for r in rows]
        delta_raw = np.array(delta, dtype="<u2").tobytes()
        payload = zstandard.ZstdCompressor(level=3, write_content_size=True).compress(delta_raw)
        reference = _container("delta_zstd", 5, 11, payload, arr.tobytes())
        self.assertEqual(encode(arr, "delta_zstd"), reference)
        self.assertEqual(decode(reference).tobytes(), arr.tobytes())

    def test_decode_does_not_mutate_or_alias_blob(self):
        arr = _random_map(11, (6, 7))
        for method in METHODS:
            with self.subTest(method=method):
                blob = bytearray(encode(arr, method))
                before = bytes(blob)
                out = decode(blob)
                self.assertEqual(bytes(blob), before)
                self.assertTrue(out.flags.writeable)
                self.assertTrue(out.flags.c_contiguous)
                out[...] = 0
                self.assertEqual(bytes(blob), before)
                self.assertEqual(decode(before).tobytes(), arr.tobytes())

    def test_bytearray_and_memoryview_blob(self):
        arr = _smooth_map((4, 7))
        blob = encode(arr, "zstd")
        np.testing.assert_array_equal(decode(bytearray(blob)), arr)
        np.testing.assert_array_equal(decode(memoryview(blob)), arr)


class EncodeValidationTests(unittest.TestCase):
    def test_invalid_method(self):
        arr = _smooth_map((2, 2))
        for method in ("gzip", "", "ZSTD", None, b"zstd", 0):
            with self.subTest(method=method):
                with self.assertRaises(ValueError):
                    encode(arr, method)

    def test_invalid_dtype(self):
        base = np.arange(6).reshape(2, 3)
        for dtype in ("<i2", "<u4", "<i8", "<f4", "<f8", ">u2", "u1", "?"):
            with self.subTest(dtype=dtype):
                with self.assertRaises(ValueError):
                    encode(base.astype(dtype), "zstd")

    def test_not_ndarray(self):
        for value in ([[1, 2], [3, 4]], b"\x00\x00", None, 5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    encode(value)

    def test_invalid_shape(self):
        for shape in ((), (5,), (2, 2, 2), (0, 5), (5, 0), (0, 0)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    encode(np.zeros(shape, dtype="<u2"))

    def test_oversized_input_rejected(self):
        # Broadcast view: no 64 MiB allocation happens for this test.
        arr = np.broadcast_to(np.zeros(1, dtype="<u2"), (8193, 4096))
        with self.assertRaises(ValueError):
            encode(arr, "zstd")


class DecodeValidationTests(unittest.TestCase):
    def setUp(self):
        self.arr = _random_map(42, (8, 8))
        self.blobs = {m: encode(self.arr, m) for m in METHODS}

    def test_not_bytes(self):
        for value in ("text", None, 3, [1, 2]):
            with self.assertRaises(ValueError):
                decode(value)

    def test_truncated_every_length(self):
        for method, blob in self.blobs.items():
            for cut in range(len(blob)):
                with self.subTest(method=method, cut=cut):
                    with self.assertRaises(ValueError):
                        decode(blob[:cut])

    def test_trailing_data(self):
        for method, blob in self.blobs.items():
            for extra in (b"\x00", b"junk", blob):
                with self.subTest(method=method, extra=len(extra)):
                    with self.assertRaises(ValueError):
                        decode(blob + extra)

    def test_trailing_data_inside_payload(self):
        # Payload length field updated to cover junk after the compressed frame.
        for method, blob in self.blobs.items():
            with self.subTest(method=method):
                padded = blob + b"\x00\x00\x00\x00"
                padded = _patch_header(padded, plen=len(padded) - codec.HEADER_SIZE)
                with self.assertRaises(ValueError):
                    decode(padded)

    def test_payload_bit_flips_never_return_wrong_data(self):
        for method, blob in self.blobs.items():
            for i in range(codec.HEADER_SIZE, len(blob)):
                corrupt = bytearray(blob)
                corrupt[i] ^= 0x41
                with self.subTest(method=method, index=i):
                    try:
                        out = decode(bytes(corrupt))
                    except ValueError:
                        continue
                    self.assertEqual(out.tobytes(), self.arr.tobytes())

    def test_checksum_mismatch(self):
        for method, blob in self.blobs.items():
            digest = bytearray(codec._HEADER.unpack_from(blob)[-1])
            digest[0] ^= 0xFF
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "checksum"):
                    decode(_patch_header(blob, digest=bytes(digest)))

    def test_bad_magic_version_method_predictor_reserved(self):
        blob = self.blobs["delta_zstd"]
        bad = [
            dict(magic=b"XXXX"),
            dict(version=0),
            dict(version=2),
            dict(version=255),
            dict(method=3),
            dict(method=255),
            dict(predictor=0),
            dict(predictor=2),
            dict(reserved=1),
        ]
        for fields in bad:
            with self.subTest(fields=fields):
                with self.assertRaises(ValueError):
                    decode(_patch_header(blob, **fields))

    def test_method_swap_rejected(self):
        # zstd payload relabelled as delta_zstd (and vice versa) must fail.
        zblob = self.blobs["zstd"]
        dblob = self.blobs["delta_zstd"]
        with self.assertRaises(ValueError):
            decode(_patch_header(zblob, method=2))
        with self.assertRaises(ValueError):
            decode(_patch_header(zblob, method=2, predictor=1))
        with self.assertRaises(ValueError):
            decode(_patch_header(dblob, method=0, predictor=0))
        with self.assertRaises(ValueError):
            decode(_patch_header(zblob, method=1))

    def test_header_bit_flips_rejected(self):
        for method, blob in self.blobs.items():
            for i in range(codec.HEADER_SIZE):
                corrupt = bytearray(blob)
                corrupt[i] ^= 0x01
                with self.subTest(method=method, index=i):
                    with self.assertRaises(ValueError):
                        decode(bytes(corrupt))

    def test_reshaped_header_without_crc_update_rejected(self):
        # Same sample bytes, different shape: only the header CRC catches this
        # for the raw methods.
        for method, blob in self.blobs.items():
            body = bytearray(blob)
            struct_fields = list(codec._HEADER.unpack_from(blob))
            struct_fields[5], struct_fields[6] = 64, 1
            codec._HEADER.pack_into(body, 0, *struct_fields)
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "CRC"):
                    decode(bytes(body))

    def test_invalid_dimensions(self):
        for method, blob in self.blobs.items():
            for h, w in ((0, 8), (8, 0), (0, 0), (4, 8), (8, 9),
                         (2**32 - 1, 2**32 - 1), (8193, 4096)):
                with self.subTest(method=method, dims=(h, w)):
                    with self.assertRaises(ValueError):
                        decode(_patch_header(blob, height=h, width=w))

    def test_payload_length_field_mismatch(self):
        blob = self.blobs["zstd"]
        plen = len(blob) - codec.HEADER_SIZE
        for value in (0, plen - 1, plen + 1, 2**64 - 1):
            with self.assertRaises(ValueError):
                decode(_patch_header(blob, plen=value))

    def test_zstd_frame_claiming_oversized_content(self):
        # Frame header claims more than the container dimensions allow; must be
        # rejected from the header alone.
        big = zstandard.ZstdCompressor(level=3).compress(b"\x00" * (1 << 20))
        for method in ("zstd", "delta_zstd"):
            with self.assertRaises(ValueError):
                decode(_container(method, 2, 2, big))

    def test_zstd_frame_claiming_huge_content(self):
        # Hand-built frame header advertising a 2**40 byte content size.
        fhd = bytes([0xE0])  # FCS field 8 bytes, single segment
        frame = b"\x28\xb5\x2f\xfd" + fhd + (1 << 40).to_bytes(8, "little")
        frame += b"\x01\x00\x00"  # last raw block of size 0
        with self.assertRaises(ValueError):
            decode(_container("zstd", 1, 1, frame))

    def test_zstd_frame_without_content_size(self):
        raw = self.arr.tobytes()
        payload = zstandard.ZstdCompressor(level=3, write_content_size=False).compress(raw)
        with self.assertRaises(ValueError):
            decode(_container("zstd", 8, 8, payload, raw))

    def test_zstd_frame_with_excessive_window(self):
        # Two output bytes, but a non-single-segment frame claiming a 2 GiB window.
        frame = b"\x28\xb5\x2f\xfd\x80\xa8" + (2).to_bytes(4, "little")
        frame += b"\x11\x00\x00\x00\x00"
        with self.assertRaisesRegex(ValueError, "window"):
            decode(_container("zstd", 1, 1, frame, b"\x00\x00"))

    def test_zstd_frame_content_smaller_than_dims(self):
        raw = self.arr.tobytes()
        payload = zstandard.ZstdCompressor(level=3).compress(raw[:-2])
        with self.assertRaises(ValueError):
            decode(_container("zstd", 8, 8, payload, raw))

    def test_lz4_frame_claiming_oversized_content(self):
        big = lz4.frame.compress(b"\x00" * (1 << 20), store_size=True)
        with self.assertRaises(ValueError):
            decode(_container("lz4", 2, 2, big))

    def test_lz4_frame_without_content_size(self):
        raw = self.arr.tobytes()
        payload = lz4.frame.compress(raw, store_size=False)
        with self.assertRaises(ValueError):
            decode(_container("lz4", 8, 8, payload, raw))

    def test_lz4_two_frames_rejected(self):
        raw = self.arr.tobytes()
        frame = lz4.frame.compress(raw, store_size=True)
        with self.assertRaises(ValueError):
            decode(_container("lz4", 8, 8, frame + frame, raw))

    def test_garbage_payload(self):
        rng = np.random.default_rng(2024)
        for method in METHODS:
            for n in (1, 4, 16, 200):
                junk = rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()
                with self.subTest(method=method, n=n):
                    with self.assertRaises(ValueError):
                        decode(_container(method, 8, 8, junk))


if __name__ == "__main__":
    unittest.main()
