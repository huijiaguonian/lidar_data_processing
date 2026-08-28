import struct
import unittest

import numpy as np

from hesai_xt32 import parse_pcap


def make_payload(distance_unit=4):
    payload = bytearray(
        parse_pcap.TAIL_START + parse_pcap.DATA_TAIL_LEN
    )
    payload[:2] = parse_pcap.SOP
    payload[2:4] = bytes(parse_pcap.PROTOCOL_VERSION)
    payload[6] = parse_pcap.CHANNELS_PER_BLOCK
    payload[7] = parse_pcap.BLOCKS_PER_PACKET
    payload[9] = distance_unit
    payload[10] = 1
    payload[11] = 0

    for block_index in range(parse_pcap.BLOCKS_PER_PACKET):
        block_offset = (
            parse_pcap.HEADER_LEN + block_index * parse_pcap.BLOCK_SIZE
        )
        struct.pack_into("<H", payload, block_offset, block_index * 100)
        channel_offset = block_offset + 2
        struct.pack_into("<H", payload, channel_offset, 1000)
        payload[channel_offset + 2] = 255

    return bytes(payload)


class PandarXTPayloadTest(unittest.TestCase):
    def test_distance_unit_is_read_from_header(self):
        for raw_unit, expected_unit in ((1, 0.001), (4, 0.004)):
            with self.subTest(raw_unit=raw_unit):
                payload = make_payload(distance_unit=raw_unit)
                blocks, header = parse_pcap.parse_payload_blocks(
                    payload,
                    np.zeros(32, dtype=np.float64),
                    np.zeros(32, dtype=np.float64),
                )

                self.assertEqual(len(blocks), 8)
                self.assertAlmostEqual(header["distance_unit_m"], expected_unit)
                self.assertEqual(sum(len(points) for _, points in blocks), 8)
                np.testing.assert_allclose(
                    blocks[0][1][0],
                    np.array([0.0, 1000 * expected_unit, 0.0, 1.0]),
                    atol=1e-6,
                )

    def test_rejects_wrong_channel_count(self):
        payload = bytearray(make_payload())
        payload[6] = 128
        with self.assertRaisesRegex(ValueError, "expected 32 x 8"):
            parse_pcap.parse_header(bytes(payload))


if __name__ == "__main__":
    unittest.main()
