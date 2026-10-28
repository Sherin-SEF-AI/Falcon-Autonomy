"""A schema-driven CDR decoder for ROS 2 messages (standard library and numpy; no ROS, no mcap_ros2).

What it reads. Message definitions in the ros2msg text rosbag2 stores (an MCAP Schema record's data, or rosbag2's
sqlite `message_definitions` table): the type's own .msg text, then each dependency after a line of 80 '=' and
'MSG: pkg/Type'. Field lines are `TYPE NAME [DEFAULT]`; constants (`TYPE NAME=VALUE`) and comments are skipped. Types:
the primitives bool, byte, char, int8…uint64, float32, float64, string (and bounded string<=N); nested messages by
'pkg/Type', 'pkg/msg/Type', a bare 'Type' of the same package, or 'Header' (std_msgs/Header); arrays T[] and T[<=N]
(sequences: a uint32 count, then the elements) and T[N] (fixed: no count). wstring is refused (no ROS 2 message this
module reads uses it, and its CDR width depends on the middleware).

The wire format. The 4-byte encapsulation header: 00 00 (CDR, big-endian) or 00 01 (CDR, little-endian), then two
option bytes. XCDR1 ("plain CDR"), what every ROS 2 middleware writes: each primitive aligned to its own size relative
to the first byte after the header; a string is a uint32 length that counts its terminating NUL, the bytes, the NUL; a
sequence a uint32 count, then its elements (aligned to the element's size when there is at least one). Other
encapsulations (parameter lists, XCDR2) are refused, saying which.

Values come back as plain dicts keyed by field name; uint8 / byte / char arrays as bytes; other primitive arrays as
lists (long ones as numpy arrays). Cross-checked against mcap_ros2's own decoder and encoder on every message type
Falcon imports (tests/test_logreplay.py).

BUILTIN holds the definitions used when a bag stores none (rosbag2 sqlite bags before Iron): the standard messages a
log import reads, as ROS 2 Humble and Jazzy define them (the same layouts in both), and vision_msgs 4.x (Humble).
"""
from __future__ import annotations

import re
import struct

SEP = '=' * 80
_PRIM = {'bool': ('?', 1), 'byte': ('B', 1), 'char': ('B', 1), 'int8': ('b', 1), 'uint8': ('B', 1),
         'int16': ('h', 2), 'uint16': ('H', 2), 'int32': ('i', 4), 'uint32': ('I', 4),
         'int64': ('q', 8), 'uint64': ('Q', 8), 'float32': ('f', 4), 'float64': ('d', 8)}
_NP = {'?': 'bool', 'B': 'u1', 'b': 'i1', 'h': 'i2', 'H': 'u2', 'i': 'i4', 'I': 'u4', 'q': 'i8', 'Q': 'u8', 'f': 'f4',
       'd': 'f8'}
_BYTES = frozenset(('uint8', 'byte', 'char'))
_CONST = re.compile(r'^\s*[A-Za-z_][\w/]*(?:<=\d+)?(?:\[[^\]]*\])?\s+[A-Za-z_]\w*\s*=')
_TYPE = re.compile(r'^([A-Za-z_][\w/]*)(?:<=(\d+))?(?:\[(<=)?(\d*)\])?$')
LONG_ARRAY = 64                 # primitive arrays longer than this come back as numpy arrays


class CdrError(ValueError):
    pass


BUILTIN: dict[str, str] = {
    'builtin_interfaces/msg/Time': 'int32 sec\nuint32 nanosec\n',
    'std_msgs/msg/Header': 'builtin_interfaces/Time stamp\nstring frame_id\n',
    'geometry_msgs/msg/Point': 'float64 x\nfloat64 y\nfloat64 z\n',
    'geometry_msgs/msg/Point32': 'float32 x\nfloat32 y\nfloat32 z\n',
    'geometry_msgs/msg/Vector3': 'float64 x\nfloat64 y\nfloat64 z\n',
    'geometry_msgs/msg/Quaternion': 'float64 x 0\nfloat64 y 0\nfloat64 z 0\nfloat64 w 1\n',
    'geometry_msgs/msg/Pose': 'Point position\nQuaternion orientation\n',
    'geometry_msgs/msg/PoseStamped': 'std_msgs/Header header\nPose pose\n',
    'geometry_msgs/msg/PoseWithCovariance': 'Pose pose\nfloat64[36] covariance\n',
    'geometry_msgs/msg/PoseWithCovarianceStamped': 'std_msgs/Header header\nPoseWithCovariance pose\n',
    'geometry_msgs/msg/Twist': 'Vector3 linear\nVector3 angular\n',
    'geometry_msgs/msg/TwistWithCovariance': 'Twist twist\nfloat64[36] covariance\n',
    'geometry_msgs/msg/Accel': 'Vector3 linear\nVector3 angular\n',
    'geometry_msgs/msg/AccelWithCovariance': 'Accel accel\nfloat64[36] covariance\n',
    'geometry_msgs/msg/Transform': 'Vector3 translation\nQuaternion rotation\n',
    'geometry_msgs/msg/TransformStamped': 'std_msgs/Header header\nstring child_frame_id\nTransform transform\n',
    'geometry_msgs/msg/Polygon': 'Point32[] points\n',
    'tf2_msgs/msg/TFMessage': 'geometry_msgs/TransformStamped[] transforms\n',
    'nav_msgs/msg/Odometry': ('std_msgs/Header header\nstring child_frame_id\ngeometry_msgs/PoseWithCovariance pose\n'
                              'geometry_msgs/TwistWithCovariance twist\n'),
    'sensor_msgs/msg/NavSatStatus': ('int8 STATUS_NO_FIX=-1\nint8 STATUS_FIX=0\nint8 status\n'
                                     'uint16 SERVICE_GPS=1\nuint16 service\n'),
    'sensor_msgs/msg/NavSatFix': ('uint8 COVARIANCE_TYPE_UNKNOWN=0\nstd_msgs/Header header\n'
                                  'sensor_msgs/NavSatStatus status\nfloat64 latitude\nfloat64 longitude\n'
                                  'float64 altitude\nfloat64[9] position_covariance\nuint8 position_covariance_type\n'),
    'vision_msgs/msg/ObjectHypothesis': 'string class_id\nfloat64 score\n',
    'vision_msgs/msg/ObjectHypothesisWithPose': ('ObjectHypothesis hypothesis\n'
                                                 'geometry_msgs/PoseWithCovariance pose\n'),
    'vision_msgs/msg/BoundingBox3D': 'geometry_msgs/Pose center\ngeometry_msgs/Vector3 size\n',
    'vision_msgs/msg/Detection3D': ('std_msgs/Header header\nObjectHypothesisWithPose[] results\n'
                                    'BoundingBox3D bbox\nstring id\n'),
    'vision_msgs/msg/Detection3DArray': 'std_msgs/Header header\nDetection3D[] detections\n',
}


def norm_type(name: str, pkg: str = '') -> str:
    """'pkg/Type', 'pkg/msg/Type' or a bare 'Type' of pkg (or 'Header') -> 'pkg/msg/Type'."""
    parts = name.strip().split('/')
    if len(parts) == 1:
        return 'std_msgs/msg/Header' if parts[0] == 'Header' else f'{pkg}/msg/{parts[0]}'
    if len(parts) == 2:
        return f'{parts[0]}/msg/{parts[1]}'
    return '/'.join(parts)


def split_schema(type_name: str, text: str) -> dict:
    """{'pkg/msg/Type': .msg text} of a ros2msg schema (the type and every dependency it carries)."""
    out, cur, buf = {}, norm_type(type_name), []
    for line in text.splitlines():
        if line.strip() == SEP:
            out[cur] = '\n'.join(buf) + '\n'
            cur, buf = None, []
            continue
        if cur is None:
            if line.startswith('MSG:'):
                cur = norm_type(line[4:].strip())
            continue
        buf.append(line)
    if cur is not None:
        out[cur] = '\n'.join(buf) + '\n'
    return out


def parse_fields(type_name: str, text: str) -> list:
    """[(name, base, n, bound)] of a .msg text: base a primitive, 'string' or a normalised message type; n None for
    a scalar, -1 for a sequence, k for a fixed array."""
    pkg = type_name.split('/')[0]
    fields = []
    for raw in text.splitlines():
        if _CONST.match(raw):
            continue
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        bits = line.split()
        if len(bits) < 2:
            raise CdrError(f'{type_name}: cannot read the field line {raw!r}')
        m = _TYPE.match(bits[0])
        if m is None:
            raise CdrError(f'{type_name}: cannot read the type {bits[0]!r}')
        base, arr_bound, arr_n = m.group(1), m.group(3), m.group(4)
        if base == 'wstring':
            raise CdrError(f'{type_name}.{bits[1]}: wstring is not supported')
        if base not in _PRIM and base != 'string':
            base = norm_type(base, pkg)
        if '[' not in bits[0]:
            n = None
        elif arr_bound or arr_n == '':
            n = -1
        else:
            n = int(arr_n)
        fields.append((bits[1], base, n))
    return fields


class Decoder:
    """Decodes CDR messages of the types it has definitions for.

        dec = Decoder()                                   # BUILTIN definitions
        dec.add_schema('vision_msgs/msg/Detection3DArray', schema_text)
        msg = dec.decode('vision_msgs/msg/Detection3DArray', data)
    """

    def __init__(self, defs: dict | None = None, builtin: bool = True):
        self.defs: dict[str, str] = dict(BUILTIN) if builtin else {}
        self.defs.update({norm_type(k): v for k, v in (defs or {}).items()})
        self._fields: dict[str, list] = {}

    def add_schema(self, type_name: str, text: str):
        """A schema record's definitions take precedence over the built-in ones (they are what was written)."""
        for t, body in split_schema(type_name, text).items():
            self.defs[t] = body
            self._fields.pop(t, None)

    def has(self, type_name: str) -> bool:
        try:
            self._check(norm_type(type_name), set())
            return True
        except CdrError:
            return False

    def _check(self, t, seen):
        if t in seen:
            return
        seen.add(t)
        for _, base, _ in self.fields(t):
            if base not in _PRIM and base != 'string':
                self._check(base, seen)

    def fields(self, t: str) -> list:
        f = self._fields.get(t)
        if f is None:
            if t not in self.defs:
                raise CdrError(f'no definition of {t}')
            f = self._fields[t] = parse_fields(t, self.defs[t])
        return f

    def decode(self, type_name: str, data: bytes) -> dict:
        b = bytes(data)
        if len(b) < 4:
            raise CdrError('a CDR message has a 4-byte encapsulation header')
        kind = b[1] if b[0] == 0 else 0xff
        if kind not in (0, 1):
            raise CdrError(f'encapsulation {b[0]:02x} {b[1]:02x} is not plain CDR (00 00 or 00 01): parameter lists '
                           'and XCDR2 are not supported')
        r = _Reader(b, '<' if kind == 1 else '>')
        return self._struct(r, norm_type(type_name))

    def _struct(self, r, t):
        out = {}
        for name, base, n in self.fields(t):
            if n is None:
                out[name] = self._one(r, base)
                continue
            count = r.prim('I', 4) if n < 0 else n
            if base in _BYTES:
                out[name] = r.raw(count)
            elif base in _PRIM:
                code, size = _PRIM[base]
                out[name] = r.array(code, size, count)
            else:
                out[name] = [self._one(r, base) for _ in range(count)]
        return out

    def _one(self, r, base):
        if base == 'string':
            return r.string()
        if base in _PRIM:
            code, size = _PRIM[base]
            return r.prim(code, size)
        return self._struct(r, base)


class _Reader:
    def __init__(self, b: bytes, endian: str):
        self.b, self.e, self.o = b, endian, 4

    def _align(self, n):
        self.o += (-(self.o - 4)) % n

    def prim(self, code, size):
        self._align(size)
        if self.o + size > len(self.b):
            raise CdrError('the message ends early')
        v = struct.unpack_from(self.e + code, self.b, self.o)[0]
        self.o += size
        return v

    def raw(self, n):
        if self.o + n > len(self.b):
            raise CdrError('the message ends early')
        v = self.b[self.o:self.o + n]
        self.o += n
        return v

    def string(self):
        n = self.prim('I', 4)
        s = self.raw(n)
        return s[:-1].decode('utf-8', 'replace') if n else ''

    def array(self, code, size, count):
        if not count:
            return []
        self._align(size)
        if self.o + size * count > len(self.b):
            raise CdrError('the message ends early')
        if count > LONG_ARRAY:
            import numpy as np
            a = np.frombuffer(self.b, (self.e + _NP[code]) if size > 1 else _NP[code], count, self.o).copy()
            self.o += size * count
            return a
        v = list(struct.unpack_from(f'{self.e}{count}{code}', self.b, self.o))
        self.o += size * count
        return v
