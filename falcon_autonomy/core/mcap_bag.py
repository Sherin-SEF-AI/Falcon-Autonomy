"""ROS 2 bags without ROS: a CDR encoder for the ROS 2 messages a drive needs and a writer of rosbag2 bags in MCAP
storage (numpy and the standard library; the zstandard module only when zstd chunks are asked for). Ported from
Vantage (vantage.core.rosbag's writer and message helpers), cut down to writing.

Messages are plain dicts keyed by the .msg field names: nested messages dicts, arrays lists, uint8 / byte / char arrays
(image and point-cloud data) bytes. Missing fields take their defaults (0, '', empty; a Quaternion's w = 1); keys a type
does not have are refused.

CDR: XCDR1 little-endian ("plain CDR", what every ROS 2 middleware writes): the encapsulation header 00 01 00 00, then
the fields in order, each primitive aligned to its own size relative to the payload after the header; a string is a
uint32 length counting the terminating NUL, the bytes and the NUL; a sequence a uint32 count, then its elements.

MCAP (https://mcap.dev/spec) as rosbag2's mcap storage writes it: profile 'ros2', schema encoding 'ros2msg' (the .msg
text, then each dependency after a line of 80 '=' and 'MSG: pkg/Type'), message encoding 'cdr', channel metadata
offered_qos_profiles; messages in chunks (uncompressed, or zstd) each followed by its message indexes; a summary
(schemas, channels, statistics, chunk indexes, summary offsets) and real CRC-32s. The bag directory holds
<name>_0.mcap and metadata.yaml (rosbag2's version-5 layout, which Humble and later read).

Definitions: the types below as ROS 2 Humble and Jazzy define them (the field layouts are the same in both; Jazzy's
extra NavSatStatus constants are carried, constants do not change the wire format). Limits: no reader (read a bag with
the mcap package, or rosbags); /tf_static gets the transient-local QoS rosbag2 writes for it, every other topic the
empty profile `ros2 bag play` replaces with its defaults.
"""
from __future__ import annotations

import os
import re
import struct
import zlib

import numpy as np

MSG_DEFS: dict[str, str] = {
    'builtin_interfaces/msg/Time': 'int32 sec\nuint32 nanosec\n',
    'std_msgs/msg/Header': 'builtin_interfaces/Time stamp\nstring frame_id\n',
    'geometry_msgs/msg/Vector3': 'float64 x\nfloat64 y\nfloat64 z\n',
    'geometry_msgs/msg/Quaternion': 'float64 x 0\nfloat64 y 0\nfloat64 z 0\nfloat64 w 1\n',
    'geometry_msgs/msg/Transform': 'geometry_msgs/Vector3 translation\ngeometry_msgs/Quaternion rotation\n',
    'geometry_msgs/msg/TransformStamped': ('std_msgs/Header header\nstring child_frame_id\n'
                                           'geometry_msgs/Transform transform\n'),
    'tf2_msgs/msg/TFMessage': 'geometry_msgs/TransformStamped[] transforms\n',
    'sensor_msgs/msg/Image': ('std_msgs/Header header\nuint32 height\nuint32 width\nstring encoding\n'
                              'uint8 is_bigendian\nuint32 step\nuint8[] data\n'),
    'sensor_msgs/msg/CompressedImage': 'std_msgs/Header header\nstring format\nuint8[] data\n',
    'sensor_msgs/msg/RegionOfInterest': ('uint32 x_offset\nuint32 y_offset\nuint32 height\nuint32 width\n'
                                         'bool do_rectify\n'),
    'sensor_msgs/msg/CameraInfo': ('std_msgs/Header header\nuint32 height\nuint32 width\nstring distortion_model\n'
                                   'float64[] d\nfloat64[9] k\nfloat64[9] r\nfloat64[12] p\nuint32 binning_x\n'
                                   'uint32 binning_y\nsensor_msgs/RegionOfInterest roi\n'),
    'sensor_msgs/msg/PointField': ('uint8 INT8=1\nuint8 UINT8=2\nuint8 INT16=3\nuint8 UINT16=4\nuint8 INT32=5\n'
                                   'uint8 UINT32=6\nuint8 FLOAT32=7\nuint8 FLOAT64=8\n'
                                   'string name\nuint32 offset\nuint8 datatype\nuint32 count\n'),
    'sensor_msgs/msg/PointCloud2': ('std_msgs/Header header\nuint32 height\nuint32 width\n'
                                    'sensor_msgs/PointField[] fields\nbool is_bigendian\nuint32 point_step\n'
                                    'uint32 row_step\nuint8[] data\nbool is_dense\n'),
    'sensor_msgs/msg/Imu': ('std_msgs/Header header\ngeometry_msgs/Quaternion orientation\n'
                            'float64[9] orientation_covariance\ngeometry_msgs/Vector3 angular_velocity\n'
                            'float64[9] angular_velocity_covariance\ngeometry_msgs/Vector3 linear_acceleration\n'
                            'float64[9] linear_acceleration_covariance\n'),
    'sensor_msgs/msg/NavSatStatus': ('int8 STATUS_UNKNOWN=-2\nint8 STATUS_NO_FIX=-1\nint8 STATUS_FIX=0\n'
                                     'int8 STATUS_SBAS_FIX=1\nint8 STATUS_GBAS_FIX=2\nint8 status\n'
                                     'uint16 SERVICE_UNKNOWN=0\nuint16 SERVICE_GPS=1\nuint16 SERVICE_GLONASS=2\n'
                                     'uint16 SERVICE_COMPASS=4\nuint16 SERVICE_GALILEO=8\nuint16 service\n'),
    'sensor_msgs/msg/NavSatFix': ('uint8 COVARIANCE_TYPE_UNKNOWN=0\nuint8 COVARIANCE_TYPE_APPROXIMATED=1\n'
                                  'uint8 COVARIANCE_TYPE_DIAGONAL_KNOWN=2\nuint8 COVARIANCE_TYPE_KNOWN=3\n'
                                  'std_msgs/Header header\nsensor_msgs/NavSatStatus status\nfloat64 latitude\n'
                                  'float64 longitude\nfloat64 altitude\nfloat64[9] position_covariance\n'
                                  'uint8 position_covariance_type\n'),
}

# rosbag2's version-5 QoS text of a latched topic: reliable, transient local, durations infinite
QOS_TRANSIENT_LOCAL = ('- history: 3\n  depth: 0\n  reliability: 1\n  durability: 1\n  deadline:\n    sec: 9223372036\n'
                       '    nsec: 854775807\n  lifespan:\n    sec: 9223372036\n    nsec: 854775807\n  liveliness: 1\n'
                       '  liveliness_lease_duration:\n    sec: 9223372036\n    nsec: 854775807\n'
                       '  avoid_ros_namespace_conventions: false')
LATCHED_TOPICS = ('/tf_static',)

_PRIM = {'bool': ('?', 1), 'byte': ('B', 1), 'char': ('B', 1), 'int8': ('b', 1), 'uint8': ('B', 1),
         'int16': ('h', 2), 'uint16': ('H', 2), 'int32': ('i', 4), 'uint32': ('I', 4),
         'int64': ('q', 8), 'uint64': ('Q', 8), 'float32': ('f', 4), 'float64': ('d', 8)}
_BYTES = frozenset(('uint8', 'byte', 'char'))
_SEP = '=' * 80
_TYPE_RE = re.compile(r'([A-Za-z_][\w/]*)(?:\[(\d*)\])?$')


def _norm(name: str, pkg: str = '') -> str:
    """'pkg/Type' or 'pkg/msg/Type' (or a bare 'Type' of pkg) -> 'pkg/msg/Type'."""
    parts = name.split('/')
    if len(parts) == 1:
        return f'{pkg}/msg/{parts[0]}' if parts[0] != 'Header' else 'std_msgs/msg/Header'
    if len(parts) == 2:
        return f'{parts[0]}/msg/{parts[1]}'
    return name


def _short(t: str) -> str:
    p = t.split('/')
    return f'{p[0]}/{p[-1]}'


_SPECS: dict = {}


def _spec(t: str) -> list:
    """[(name, base type, n: None scalar | -1 sequence | k fixed array, default)] of a type (constants skipped)."""
    if t in _SPECS:
        return _SPECS[t]
    if t not in MSG_DEFS:
        raise ValueError(f'unknown message type {t}')
    pkg = t.split('/')[0]
    out = []
    for raw in MSG_DEFS[t].splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        typ, rest = line.split(None, 1)
        name, eq, _ = rest.partition('=')
        if eq and ' ' not in name.strip():
            continue                                          # a constant
        bits = rest.split()
        m = _TYPE_RE.match(typ)
        base, num = m.group(1), m.group(2)
        if base not in _PRIM and base != 'string':
            base = _norm(base, pkg)
        n = None if num is None else (-1 if num == '' else int(num))
        default = float(bits[1]) if len(bits) > 1 else None
        out.append((bits[0], base, n, default))
    _SPECS[t] = out
    return out


def _deps(t: str, seen=None) -> list:
    """The type and its dependencies, depth first, sorted by name at each level, each once."""
    seen = [] if seen is None else seen
    if t in seen:
        return seen
    seen.append(t)
    for d in sorted({b for _, b, _, _ in _spec(t) if b not in _PRIM and b != 'string'}):
        _deps(d, seen)
    return seen


def schema_text(msg_type: str) -> str:
    """The ros2msg schema rosbag2 writes into an MCAP Schema record."""
    t = _norm(msg_type)
    parts = [MSG_DEFS[t]]
    for d in _deps(t)[1:]:
        parts.append(f'\n{_SEP}\nMSG: {_short(d)}\n{MSG_DEFS[d]}')
    return ''.join(parts)


def _align(o: bytearray, n: int):
    pad = (-(len(o) - 4)) % n
    if pad:
        o.extend(b'\x00' * pad)


def _prim(o, base, v):
    code, size = _PRIM[base]
    _align(o, size)
    o.extend(struct.pack('<' + code, v))


def _string(o, v):
    b = str(v).encode('utf-8')
    _align(o, 4)
    o.extend(struct.pack('<I', len(b) + 1))
    o.extend(b)
    o.append(0)


def _value(o, base, v):
    if base == 'string':
        _string(o, '' if v is None else v)
    elif base in _PRIM:
        _prim(o, base, 0 if v is None else v)
    else:
        _encode(o, base, v or {})


def _encode(o: bytearray, t: str, msg: dict):
    spec = _spec(t)
    bad = set(msg) - {f[0] for f in spec}
    if bad:
        raise ValueError(f'{t} has no field(s) {", ".join(sorted(bad))}')
    for name, base, n, default in spec:
        v = msg.get(name)
        if n is None:
            _value(o, base, default if v is None else v)
            continue
        if base in _BYTES:
            data = bytes(v or b'') if n < 0 else bytes(v or bytes(n))
            if n >= 0 and len(data) != n:
                raise ValueError(f'{t}.{name}: {n} bytes expected')
            if n < 0:
                _align(o, 4)
                o.extend(struct.pack('<I', len(data)))
            o.extend(data)
            continue
        items = list(v) if v is not None else ([0] * n if n > 0 else [])
        if n >= 0 and len(items) != n:
            raise ValueError(f'{t}.{name}: {n} values expected, got {len(items)}')
        if n < 0:
            _align(o, 4)
            o.extend(struct.pack('<I', len(items)))
        if base in _PRIM and items:
            code, size = _PRIM[base]
            _align(o, size)
            o.extend(struct.pack(f'<{len(items)}{code}', *items))
        else:
            for it in items:
                _value(o, base, it)


def encode(msg_type: str, msg: dict) -> bytes:
    """A message dict -> CDR bytes with the encapsulation header (little-endian)."""
    o = bytearray(b'\x00\x01\x00\x00')
    _encode(o, _norm(msg_type), msg)
    return bytes(o)


# ------------------------------------------------------------------------------------------------------- MCAP records
MCAP_MAGIC = b'\x89MCAP0\r\n'
OP_HEADER, OP_FOOTER, OP_SCHEMA, OP_CHANNEL, OP_MESSAGE, OP_CHUNK, OP_MESSAGE_INDEX, OP_CHUNK_INDEX = range(1, 9)
OP_STATISTICS, OP_SUMMARY_OFFSET, OP_DATA_END = 0x0B, 0x0E, 0x0F
_REC = struct.Struct('<BQ')
_MSG = struct.Struct('<BQHIQQ')         # opcode, length, channel, sequence, log time, publish time


def _s(text: str) -> bytes:
    b = text.encode()
    return struct.pack('<I', len(b)) + b


def _map(d: dict) -> bytes:
    body = b''.join(_s(k) + _s(v) for k, v in d.items())
    return struct.pack('<I', len(body)) + body


def _rec(op: int, body: bytes) -> bytes:
    return _REC.pack(op, len(body)) + body


class BagWriter:
    """A rosbag2 bag: the directory `path` with <name>_0.mcap (name = the directory's name) and metadata.yaml.

        bag = BagWriter('out/drive_01')
        bag.add_topic('/front/image/compressed', 'sensor_msgs/msg/CompressedImage')
        bag.write('/front/image/compressed', t_ns, compressed_image(jpeg, t_ns, 'front_optical'))
        bag.close()

    compression 'none' or 'zstd' (needs the zstandard module), per chunk of chunk_size uncompressed bytes. Messages
    may come in any time order. The directory must not hold a bag already."""

    def __init__(self, path: str, compression: str = 'none', chunk_size: int = 4 << 20):
        self.path = os.fspath(path)
        if compression == 'zstd':
            import zstandard
            self._compress, self.compression = zstandard.ZstdCompressor(level=3).compress, 'zstd'
        elif compression in ('none', '', None):
            self._compress, self.compression = None, ''
        else:
            raise ValueError(f"compression must be 'none' or 'zstd', not {compression!r}")
        self.chunk_size = max(1024, int(chunk_size))
        name = os.path.basename(os.path.normpath(self.path)) or 'bag'
        self.file_name = f'{name}_0.mcap'
        self.mcap_path = os.path.join(self.path, self.file_name)
        if os.path.exists(os.path.join(self.path, 'metadata.yaml')) or os.path.exists(self.mcap_path):
            raise FileExistsError(f'{self.path} already holds a bag')
        os.makedirs(self.path, exist_ok=True)
        self._f = open(self.mcap_path, 'wb')
        self._pos, self._crc = 0, 0
        self._topics = {}        # topic -> [channel id, type, count, qos]
        self._schemas = {}       # type -> (schema id, record body)
        self._channel_recs, self._chunk_indexes, self._chunks = [], [], 0
        self._count, self._start, self._end = 0, None, None
        self._new_chunk()
        self.closed = False
        self._out(MCAP_MAGIC)
        self._out(_rec(OP_HEADER, _s('ros2') + _s('falcon_autonomy')))

    def _out(self, b):
        self._f.write(b)
        self._crc = zlib.crc32(b, self._crc)
        self._pos += len(b)

    def _new_chunk(self):
        self._buf = bytearray()
        self._index = {}
        self._c_start = self._c_end = None

    def add_topic(self, topic: str, msg_type: str):
        t = _norm(msg_type)
        if topic in self._topics:
            if self._topics[topic][1] != t:
                raise ValueError(f'{topic} is already a {self._topics[topic][1]} topic, not {t}')
            return
        if t not in self._schemas:
            sid = len(self._schemas) + 1
            data = schema_text(t).encode()
            body = struct.pack('<H', sid) + _s(t) + _s('ros2msg') + struct.pack('<I', len(data)) + data
            self._schemas[t] = (sid, body)
            self._out(_rec(OP_SCHEMA, body))
        qos = QOS_TRANSIENT_LOCAL if topic in LATCHED_TOPICS else ''
        cid = len(self._topics) + 1
        body = (struct.pack('<HH', cid, self._schemas[t][0]) + _s(topic) + _s('cdr')
                + _map({'offered_qos_profiles': qos}))
        self._channel_recs.append(body)
        self._out(_rec(OP_CHANNEL, body))
        self._topics[topic] = [cid, t, 0, qos]

    def write(self, topic: str, t_ns: int, msg: dict):
        """One message at log time t_ns (integer nanoseconds since the Unix epoch)."""
        info = self._topics.get(topic)
        if info is None:
            raise ValueError(f'unknown topic {topic!r}: add_topic() it first')
        t = int(t_ns)
        o = bytearray(b'\x00\x01\x00\x00')
        _encode(o, info[1], msg)
        buf = self._buf
        self._index.setdefault(info[0], []).append((t, len(buf)))
        buf += _MSG.pack(OP_MESSAGE, 22 + len(o), info[0], info[2] & 0xffffffff, t, t)
        buf += o
        info[2] += 1
        self._count += 1
        self._c_start = t if self._c_start is None else min(self._c_start, t)
        self._c_end = t if self._c_end is None else max(self._c_end, t)
        if len(buf) >= self.chunk_size:
            self._flush_chunk()

    def _flush_chunk(self):
        raw = self._buf
        if not raw:
            return
        data = self._compress(bytes(raw)) if self._compress else raw
        start = self._pos
        head = struct.pack('<QQQI', self._c_start, self._c_end, len(raw), zlib.crc32(raw)) + _s(self.compression)
        head += struct.pack('<Q', len(data))
        self._out(_REC.pack(OP_CHUNK, len(head) + len(data)))
        self._out(head)
        self._out(bytes(data))
        length = self._pos - start
        offsets = {}
        for cid in sorted(self._index):
            entries = sorted(self._index[cid], key=lambda e: e[0])
            offsets[cid] = self._pos
            self._out(_rec(OP_MESSAGE_INDEX, struct.pack('<HI', cid, 16 * len(entries))
                           + np.asarray(entries, '<u8').tobytes()))
        mi_len = self._pos - start - length
        mio = b''.join(struct.pack('<HQ', c, off) for c, off in offsets.items())
        self._chunk_indexes.append(struct.pack('<QQQQ', self._c_start, self._c_end, start, length)
                                   + struct.pack('<I', len(mio)) + mio + struct.pack('<Q', mi_len)
                                   + _s(self.compression) + struct.pack('<QQ', len(data), len(raw)))
        self._chunks += 1
        self._start = self._c_start if self._start is None else min(self._start, self._c_start)
        self._end = self._c_end if self._end is None else max(self._end, self._c_end)
        self._new_chunk()

    def close(self) -> dict:
        """Finish the file (last chunk, summary, footer) and write metadata.yaml. Returns {'path', 'file', 'messages',
        'topics': {topic: count}, 'types': {topic: type}, 'bytes'}."""
        if self.closed:
            return self._result
        self._flush_chunk()
        self._out(_rec(OP_DATA_END, struct.pack('<I', self._crc)))
        summary_start = self._pos
        groups, summ = [], bytearray()

        def group(op, bodies):
            if bodies:
                g0 = summary_start + len(summ)
                for b in bodies:
                    summ.extend(_rec(op, b))
                groups.append((op, g0, summary_start + len(summ) - g0))
        group(OP_SCHEMA, [b for _, b in self._schemas.values()])
        group(OP_CHANNEL, self._channel_recs)
        counts = b''.join(struct.pack('<HQ', v[0], v[2]) for v in self._topics.values())
        group(OP_STATISTICS, [struct.pack('<QHIIIIQQ', self._count, len(self._schemas), len(self._topics), 0, 0,
                                          self._chunks, self._start or 0, self._end or 0)
                              + struct.pack('<I', len(counts)) + counts])
        group(OP_CHUNK_INDEX, self._chunk_indexes)
        offsets_start = summary_start + len(summ)
        for op, g0, gl in groups:
            summ.extend(_rec(OP_SUMMARY_OFFSET, struct.pack('<BQQ', op, g0, gl)))
        footer = _REC.pack(OP_FOOTER, 20) + struct.pack('<QQ', summary_start, offsets_start)
        crc = zlib.crc32(footer, zlib.crc32(summ))
        self._out(bytes(summ))
        self._out(footer + struct.pack('<I', crc))
        self._out(MCAP_MAGIC)
        self._f.close()
        self.closed = True
        self._write_metadata()
        self._result = {'path': self.path, 'file': self.mcap_path, 'messages': self._count,
                        'topics': {tp: v[2] for tp, v in self._topics.items()},
                        'types': {tp: v[1] for tp, v in self._topics.items()}, 'bytes': self._pos}
        return self._result

    def _write_metadata(self):
        """metadata.yaml (rosbag2 version 5), written by hand: the app has no YAML module."""
        start = self._start if self._count else 0
        dur = (self._end - self._start) if self._count else 0
        lines = ['rosbag2_bagfile_information:', '  version: 5', '  storage_identifier: mcap',
                 '  duration:', f'    nanoseconds: {dur}', '  starting_time:',
                 f'    nanoseconds_since_epoch: {start}', f'  message_count: {self._count}',
                 '  topics_with_message_count:']
        for tp, (_, t, n, qos) in self._topics.items():
            q = '|-\n' + '\n'.join(' ' * 10 + x for x in qos.splitlines()) if qos else '""'
            lines += ['    - topic_metadata:', f'        name: {tp}', f'        type: {t}',
                      '        serialization_format: cdr', f'        offered_qos_profiles: {q}',
                      f'      message_count: {n}']
        lines += ['  compression_format: ""', '  compression_mode: ""', '  relative_file_paths:',
                  f'    - {self.file_name}', '  files:', f'    - path: {self.file_name}', '      starting_time:',
                  f'        nanoseconds_since_epoch: {start}', '      duration:', f'        nanoseconds: {dur}',
                  f'      message_count: {self._count}']
        with open(os.path.join(self.path, 'metadata.yaml'), 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(lines) + '\n')


# --------------------------------------------------------------------------------------------------- message helpers
def stamp(t_ns: int) -> dict:
    sec, ns = divmod(int(t_ns), 1_000_000_000)
    return {'sec': sec, 'nanosec': ns}


def header(t_ns: int, frame_id: str) -> dict:
    return {'stamp': stamp(t_ns), 'frame_id': str(frame_id)}


_PF_TYPE = {('i', 1): 1, ('u', 1): 2, ('i', 2): 3, ('u', 2): 4, ('i', 4): 5, ('u', 4): 6, ('f', 4): 7, ('f', 8): 8}


def pointcloud2(points: np.ndarray, t_ns: int, frame_id: str) -> dict:
    """sensor_msgs/PointCloud2 (one row) of a numpy structured array; the data is its memory layout, little-endian."""
    a = np.ascontiguousarray(points)
    fields = []
    for name in a.dtype.names:
        fd, off = a.dtype.fields[name][:2]
        fields.append({'name': name, 'offset': int(off), 'datatype': _PF_TYPE[(fd.kind, fd.itemsize)], 'count': 1})
    dense = all(np.isfinite(a[n]).all() for n in ('x', 'y', 'z') if n in a.dtype.names)
    return {'header': header(t_ns, frame_id), 'height': 1, 'width': int(len(a)), 'fields': fields,
            'is_bigendian': False, 'point_step': int(a.dtype.itemsize), 'row_step': int(len(a) * a.dtype.itemsize),
            'data': a.astype(a.dtype.newbyteorder('<'), copy=False).tobytes(), 'is_dense': bool(dense)}


def image_msg(img: np.ndarray, t_ns: int, frame_id: str, encoding: str) -> dict:
    """sensor_msgs/Image of an H×W×C uint8 (bgr8, rgb8) or H×W (mono8, mono16) array."""
    a = np.ascontiguousarray(img)
    h, w = a.shape[:2]
    return {'header': header(t_ns, frame_id), 'height': int(h), 'width': int(w), 'encoding': encoding,
            'is_bigendian': 0, 'step': int(a.strides[0]), 'data': a.astype(a.dtype.newbyteorder('<')).tobytes()}


def compressed_image(data: bytes, t_ns: int, frame_id: str, fmt: str = 'jpeg') -> dict:
    return {'header': header(t_ns, frame_id), 'format': fmt, 'data': bytes(data)}


def camera_info(width: int, height: int, K, distortion_model: str = 'plumb_bob', d=(), t_ns: int = 0,
                frame_id: str = '') -> dict:
    """sensor_msgs/CameraInfo: K (OpenCV coordinates), R identity, P = [K | 0]."""
    k = np.asarray(K, float).reshape(3, 3)
    p = np.zeros((3, 4))
    p[:, :3] = k
    return {'header': header(t_ns, frame_id), 'height': int(height), 'width': int(width),
            'distortion_model': distortion_model, 'd': [float(v) for v in d], 'k': k.reshape(9).tolist(),
            'r': np.eye(3).reshape(9).tolist(), 'p': p.reshape(12).tolist()}


def quat_xyzw(R) -> dict:
    from .rig import quat_of
    w, x, y, z = quat_of(R)
    return {'x': x, 'y': y, 'z': z, 'w': w}


def transform_stamped(T_parent_child, t_ns: int, parent: str, child: str) -> dict:
    """geometry_msgs/TransformStamped of the 4×4 pose of `child` in `parent` (child coordinates -> parent's)."""
    T = np.asarray(T_parent_child, float)
    return {'header': header(t_ns, parent), 'child_frame_id': str(child),
            'transform': {'translation': {'x': float(T[0, 3]), 'y': float(T[1, 3]), 'z': float(T[2, 3])},
                          'rotation': quat_xyzw(T[:3, :3])}}
