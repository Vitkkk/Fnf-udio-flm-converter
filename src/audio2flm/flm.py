import base64
import gzip
import io
import math
import struct
from dataclasses import dataclass

FLM_TICKS_PER_BEAT = 128
EVN2_VERSION = 20

_TEMPLATE_GZIP_B64 = (
    "H4sIAK4faGoC/zM08HHzcHV0KWRkYAAiMChLrMrMZxjpIM6BBMUOD6ta7IH0AsOJdfZiJ9/wT53+Zm7fNO6iriuFijBFjIxMQJIdiE30DA30DM29XSOd/FgY"
    "GHSAuJpLAQiUkjPyi1KKlawUosF8EKiGs8AqkhKLU/3yS1KBagx0UKVSK0pS84oz8/PABsSiyeYnlySWgbQZo0kUlibmZJZUggyES9TqEGG/EbXtNyTNfpMB"
    "tt90gMPffIDttxzg8Dc0pLYDjElzAJkZ0ISYCACzoEYoFeXnl8Ds5KqFFSisQOybmJVfBOXbg0lf1xBHTh4GBiMDIzNdA3NdIwsFAwMQw8BipJXhBR/UBR0Y"
    "RsEoGAWjYBSMglEwCkbByAIoDWErA3MrQ6MR5f8QF88w0MgDC0uQo7N3HlTU0MA7KMjD1UUAyP4PBLDRHxA7KCDIVwLIDk+Lsof3K5D6GEHBvgFqYDN8g0H8"
    "AEeXYDHwEAcTMwsrGzsHJxc3Dy8fv4CgkLCIKMjWemYG8AATuq1MtLPV1z+FkxkyqgWyBTTKxcmAGOVidHENdmYByrikpiWW5pSM2Owx0v0PTkpBwSHiUDY/"
    "Ikz0obSji4shM1iyHphGgzvAzAZwojx75owtNIHaQ8QakBIuLjYyaEDWZ49bDAXAxYB5IogHLgZR7xwc4ssHzijhLr6eLp5M4Czg7OHn08cBKxI9nJ09PFxA"
    "o4++jsEhrkGj1cQoGAUjHexzf2jHAJ3GgIjYOOBTz8aAXn6EBHl7aABLFWDtmqzAgqhusQGf1LLUnNFAHwWjYMQCUKtEGdouh7VKQkDlR7hCcGJuQU5qkYLh"
    "aCiNglEwUsEH+zg+XXtiVYPaG6D+ELbyI9jVz0UCS78LNOAAwqC2y3SktgtsYAIEkM1nZBgtn0bBKBjxbRcfz4B0KNvIwMfZ2ccjRRVDlYADctkBB1H+/r4K"
    "6GobfBwg3a88UIF34HTCFHvXMD8j0ACOCAMAHuMSjawnAAA="
)


@dataclass
class NoteEvent:
    start_beat: float
    duration_beat: float
    key: int
    velocity: int = 100
    pan: int = 64
    fine_pitch: int = 120
    slide: bool = False


@dataclass
class Chunk:
    type: str
    payload: bytes


def template_bytes() -> bytes:
    return gzip.decompress(base64.b64decode(_TEMPLATE_GZIP_B64))


def _i32(data, offset):
    return struct.unpack_from("<i", data, offset)[0]


def _put_i32(buffer, offset, value):
    struct.pack_into("<i", buffer, offset, int(value))


def _put_f64(buffer, offset, value):
    struct.pack_into("<d", buffer, offset, float(value))


def _encode_chunks(chunks):
    output = io.BytesIO()
    for chunk in chunks:
        output.write(chunk.type.encode("ascii"))
        output.write(struct.pack("<i", len(chunk.payload)))
        output.write(chunk.payload)
    return output.getvalue()


def _parse_chunks(data, start, end):
    chunks = []
    offset = start
    while offset < end:
        if offset + 8 > end:
            raise ValueError("Chunk FLM incompleto")
        chunk_type = data[offset : offset + 4].decode("ascii")
        length = _i32(data, offset + 4)
        if length < 0 or offset + 8 + length > end:
            raise ValueError(f"Tamanho inválido no chunk {chunk_type}")
        chunks.append(Chunk(chunk_type, data[offset + 8 : offset + 8 + length]))
        offset += 8 + length
    if offset != end:
        raise ValueError("Estrutura FLM desalinhada")
    return chunks


def _parse_top(data):
    if data[:4] != b"10LF":
        raise ValueError("Template FLM inválido")
    return _parse_chunks(data, 4, len(data))


def _write_fixed_text(buffer, offset, size, text):
    raw = text.encode("utf-8")[: max(0, size - 1)]
    buffer[offset : offset + size] = b"\0" * size
    buffer[offset : offset + len(raw)] = raw


def _patch_head(chunk, name, bpm):
    payload = bytearray(chunk.payload)
    _write_fixed_text(payload, 8, 256, name)
    _put_f64(payload, 264, max(20, min(999, bpm)))
    _put_i32(payload, 354, 2)
    return Chunk(chunk.type, bytes(payload))


def _generator_rack(chunk, index=0):
    prefix = chunk.payload[:8]
    children = []
    for child in _parse_chunks(chunk.payload, 8, len(chunk.payload)):
        payload = bytearray(child.payload)
        if child.type == "RHED" and len(payload) >= 8:
            _put_i32(payload, 4, index + 2)
        elif child.type in ("RMOd", "RMOD") and len(payload) >= 8:
            _put_i32(payload, 0, 1)
            _put_i32(payload, 4, index + 2)
        children.append(Chunk(child.type, bytes(payload)))
    return Chunk(chunk.type, prefix + _encode_chunks(children))


def _channel_header(chunk, index, name):
    fixed = bytearray(chunk.payload[:1084])
    _write_fixed_text(fixed, 0, 1024, name)
    _put_f64(fixed, 1028, index + 1.0)
    _put_i32(fixed, 1080, index + 1)
    encoded_name = name.encode("utf-8")
    return Chunk(
        chunk.type,
        bytes(fixed)
        + struct.pack("<i", len(encoded_name))
        + encoded_name
        + struct.pack("<i", len(encoded_name))
        + encoded_name
        + struct.pack("<i", 0),
    )


def _encode_evn2(events):
    output = io.BytesIO()
    output.write(struct.pack("<H", EVN2_VERSION))
    for event in sorted(events, key=lambda e: (e.start_beat, e.key, e.slide)):
        tick = max(0, round(event.start_beat * FLM_TICKS_PER_BEAT))
        duration = max(event.duration_beat, 1.0 / FLM_TICKS_PER_BEAT)
        velocity = max(
            0,
            min(255, round(max(0, min(128, event.velocity)) * 255 / 128)),
        )
        pan = max(0, min(255, round(max(0, min(128, event.pan)) * 255 / 128)))
        fine_pitch = max(
            0,
            min(
                65535,
                32767 + (max(0, min(240, event.fine_pitch)) - 120) * 273,
            ),
        )
        output.write(
            struct.pack(
                "<idHBBHBB",
                tick,
                duration,
                max(0, min(127, event.key)),
                velocity,
                pan,
                fine_pitch,
                0,
                1 if event.slide else 0,
            )
        )
    return output.getvalue()


def _clip(chunk, events):
    prefix = bytearray(chunk.payload[:8])
    _put_i32(prefix, 0, 0)
    end = max((event.start_beat + event.duration_beat for event in events), default=4.0)
    pattern_length = max(4.0, math.ceil(end))
    children = []
    for child in _parse_chunks(chunk.payload, 8, len(chunk.payload)):
        if child.type in ("CLHD", "CLHd"):
            payload = bytearray(child.payload)
            if len(payload) >= 24:
                _put_f64(payload, 0, 0.0)
                _put_f64(payload, 8, pattern_length)
                _put_f64(payload, 16, 0.0)
            children.append(Chunk(child.type, bytes(payload)))
        elif child.type == "EVN2":
            children.append(Chunk("EVN2", _encode_evn2(events)))
        else:
            children.append(child)
    return Chunk(chunk.type, bytes(prefix) + _encode_chunks(children))


def _track_header(chunk, name, events):
    children = []
    for child in _parse_chunks(chunk.payload, 0, len(chunk.payload)):
        if child.type == "DESc":
            payload = bytearray(child.payload)
            if len(payload) >= 284:
                _write_fixed_text(payload, 28, 256, name)
            children.append(Chunk(child.type, bytes(payload)))
        elif child.type == "CLIP":
            children.append(_clip(child, events))
        else:
            children.append(child)
    return Chunk(chunk.type, _encode_chunks(children))


def _generator_channel(chunk, index, name, events):
    prefix = chunk.payload[:8]
    children = []
    for child in _parse_chunks(chunk.payload, 8, len(chunk.payload)):
        if child.type == "CHHD":
            children.append(_channel_header(child, index, name))
        elif child.type == "TRKH":
            children.append(_track_header(child, name, events))
        else:
            children.append(child)
    return Chunk(chunk.type, prefix + _encode_chunks(children))


def write_flm(events, project_name="Audio to FLM", channel_name="Vocal", bpm=120.0):
    top = _parse_top(template_bytes())
    head = next(chunk for chunk in top if chunk.type == "HEAD")
    keyb = next(chunk for chunk in top if chunk.type == "KEYB")
    meta = next(chunk for chunk in top if chunk.type == "META")
    tdiv = next(chunk for chunk in top if chunk.type == "TDIV")
    racks = [chunk for chunk in top if chunk.type == "RACK"]
    channels = [chunk for chunk in top if chunk.type == "CHNL"]

    if len(racks) < 2 or len(channels) < 2:
        raise ValueError("Template sem rack/canal DirectWave esperado")

    chunks = [
        _patch_head(head, project_name, bpm),
        keyb,
        meta,
        tdiv,
        racks[0],
        _generator_rack(racks[1], 0),
        channels[0],
        _generator_channel(channels[1], 0, channel_name, events),
    ]
    return b"10LF" + _encode_chunks(chunks)
