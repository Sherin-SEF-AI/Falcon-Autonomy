"""Collective Perception Messages (ETSI TS 103 324 V2.1.1) from roadside units: the message as a dict in the ASN.1's own
shape and names, and its UPER encoding (standard library only).

What is encoded. The subset of the CPM a roadside unit of Falcon's fills, with the field names, ranges and units of
the ASN.1 modules ETSI publishes (forge.etsi.org ITS/asn1/cpm_ts103324, tag v2.1.1, BSD-3-Clause; the common data
dictionary ETSI-ITS-CDD major-version-3 minor-version-1, cdd_ts102894_2 at 55ae879d4b68 — the submodule commit of that
tag): CollectivePerceptionMessage {header ItsPduHeader {protocolVersion 2, messageId 14 (cpm), stationId}, payload
CpmPayload {managementContainer {referenceTime TimestampIts, referencePosition, messageRateRange}, cpmContainers: an
originatingRsuContainer (empty), a sensorInformationContainer, a perceptionRegionContainer and a
perceivedObjectContainer, each a WrappedCpmContainer whose containerData is an open type}}. The other alternatives
and optional components of those types are part of the schema below (they take their places in the presence bitmaps
and choice indices) but Falcon never fills them; encoding one raises.

The encoder writes ITU-T X.691 unaligned PER (UPER): constrained whole numbers in the fewest bits of their range,
SEQUENCE presence bitmaps and extension bits, CHOICE indices, SIZE-constrained SEQUENCE OF lengths (an extensible
size outside its root as the extension bit and a length determinant), open types as a length determinant and the
inner value's complete encoding in whole octets, the whole message padded to an octet. Constraints that X.691 does
not make PER-visible (WITH COMPONENTS on the header, the table constraint of containerId) do not change the bits;
TrafficParticipantType (unknown|passengerCar..tram|agricultural) in ObjectClass is PER-visible and its effective
range 0..14 (4 bits). Every encoding is checked against two independent decoders (tests/test_v2x.py, docs/v2x.md):
asn1tools (Python, compiling ETSI's ASN.1 files) and the asn1c-generated decoder of the etsi_its_messages ROS 2
package (its conversion node, in the bridge image).

Units (the CDD's): positions 0.01 m in the station's East-North-Up frame (PerceivedObjects: "x-axis East, y-axis North,
z vertical") from the referencePosition; velocities 0.01 m/s; dimensions, shape ranges 0.1 m; angles 0.1° — a
CartesianAngleValue counted positive (counter-clockwise) from the x-axis (East), a Wgs84AngleValue clockwise from North;
latitude and longitude 1e-7°; altitude 0.01 m; times ms. CoordinateConfidence and SpeedConfidence are 95 % bounds.
"""
from __future__ import annotations

import base64
import datetime as _dt
import math

STANDARD = 'ETSI TS 103 324 V2.1.1 (CPM-PDU-Descriptions 1.1), ETSI-ITS-CDD 3.1 (TS 102 894-2 V2.1.1)'
ASN1_SOURCE = ('forge.etsi.org/rep/ITS/asn1/cpm_ts103324 tag v2.1.1 (BSD-3-Clause), its cdd submodule '
               'cdd_ts102894_2 at 55ae879d4b681514764aba37fe2f9856b70b09d8')
PROTOCOL_VERSION = 2
MESSAGE_ID_CPM = 14
BTP_PORT_CPM = 2009                    # ETSI TS 103 248: the BTP-B destination port of the CPM
# TimestampIts: TAI milliseconds since 2004-01-01T00:00:00.000Z. TAI − UTC grew by the leap seconds of 2005-12-31,
# 2008-12-31, 2012-06-30, 2015-06-30 and 2016-12-31: 5 s for any UTC time from 2017 on.
_ITS_EPOCH = _dt.datetime(2004, 1, 1, tzinfo=_dt.timezone.utc)
LEAP_SECONDS_SINCE_2004 = 5


def its_time_ms(utc_iso: str) -> int:
    """TimestampIts of a UTC instant (ISO 8601, from 2017 on: the leap seconds above)."""
    t = _dt.datetime.fromisoformat(utc_iso.replace('Z', '+00:00'))
    if t.tzinfo is None:
        t = t.replace(tzinfo=_dt.timezone.utc)
    if t.year < 2017:
        raise ValueError('reference times before 2017 are not supported (the leap-second count changes)')
    return int(round((t - _ITS_EPOCH).total_seconds() * 1000)) + LEAP_SECONDS_SINCE_2004 * 1000


# ============================================================================================== UPER (X.691 unaligned)
class EncodeError(ValueError):
    pass


class _W:
    """Bits, most significant first, in one Python int."""
    __slots__ = ('v', 'n')

    def __init__(self):
        self.v = 0
        self.n = 0

    def bits(self, value: int, n: int):
        if n:
            if value < 0 or value >> n:
                raise EncodeError(f'{value} does not fit in {n} bits')
            self.v = (self.v << n) | value
            self.n += n

    def octets(self, data: bytes):
        for b in data:
            self.bits(b, 8)

    def length(self, n: int):
        """An unconstrained length determinant (X.691 11.9.3.6–7, unaligned): 8 bits below 128, 16 below 16384."""
        if n < 128:
            self.bits(n, 8)
        elif n < 16384:
            self.bits(0b10, 2)
            self.bits(n, 14)
        else:
            raise EncodeError(f'a length of {n} needs fragmentation (not supported)')

    def to_bytes(self) -> bytes:
        pad = (-self.n) % 8
        n = self.n + pad
        if n == 0:
            return b'\x00'                              # an empty encoding is one zero octet (X.691 11.1.3)
        return (self.v << pad).to_bytes(n // 8, 'big')


class _R:
    __slots__ = ('v', 'n', 'pos')

    def __init__(self, data: bytes):
        self.v = int.from_bytes(data, 'big')
        self.n = 8 * len(data)
        self.pos = 0

    def bits(self, n: int) -> int:
        if n == 0:
            return 0
        if self.pos + n > self.n:
            raise ValueError('UPER data too short')
        out = (self.v >> (self.n - self.pos - n)) & ((1 << n) - 1)
        self.pos += n
        return out

    def octets(self, k: int) -> bytes:
        return bytes(self.bits(8) for _ in range(k))

    def length(self) -> int:
        if self.bits(1) == 0:
            return self.bits(7)
        if self.bits(1) == 0:
            return self.bits(14)
        raise ValueError('fragmented lengths are not supported')


def _nbits(rng: int) -> int:
    return (rng - 1).bit_length() if rng > 1 else 0


class Int:
    def __init__(self, lo: int, hi: int):
        self.lo, self.hi = int(lo), int(hi)
        self.n = _nbits(self.hi - self.lo + 1)

    def enc(self, w: _W, v, path: str):
        if isinstance(v, bool) or not isinstance(v, int) or not self.lo <= v <= self.hi:
            raise EncodeError(f'{path}: {v!r} is not an integer in {self.lo}..{self.hi}')
        w.bits(v - self.lo, self.n)

    def dec(self, r: _R):
        return self.lo + r.bits(self.n)


class Bool:
    def enc(self, w, v, path):
        if not isinstance(v, bool):
            raise EncodeError(f'{path}: {v!r} is not a boolean')
        w.bits(int(v), 1)

    def dec(self, r):
        return bool(r.bits(1))


class Enum:
    """A non-extensible ENUMERATED: the index of the value among the root's values in ascending order."""

    def __init__(self, names):
        self.names = list(names)
        self.n = _nbits(len(self.names))

    def enc(self, w, v, path):
        if v not in self.names:
            raise EncodeError(f'{path}: {v!r} is not one of {self.names}')
        w.bits(self.names.index(v), self.n)

    def dec(self, r):
        return self.names[r.bits(self.n)]


class Seq:
    """SEQUENCE: fields (name, type, optional); ext: an extension marker (no addition is ever encoded)."""

    def __init__(self, fields, ext=False):
        self.fields = fields
        self.ext = ext

    def enc(self, w, v, path):
        if not isinstance(v, dict):
            raise EncodeError(f'{path}: a SEQUENCE needs a dict, not {type(v).__name__}')
        unknown = set(v) - {f[0] for f in self.fields}
        if unknown:
            raise EncodeError(f'{path}: unknown components {sorted(unknown)}')
        if self.ext:
            w.bits(0, 1)
        for name, _t, opt in self.fields:
            if opt:
                w.bits(int(name in v), 1)
            elif name not in v:
                raise EncodeError(f'{path}.{name} is mandatory')
        for name, t, _opt in self.fields:
            if name in v:
                if t is None:
                    raise EncodeError(f'{path}.{name}: not implemented by Falcon\'s encoder')
                t.enc(w, v[name], f'{path}.{name}')

    def dec(self, r):
        if self.ext and r.bits(1):
            raise ValueError('SEQUENCE extension additions are not supported')
        present = [(not opt) or bool(r.bits(1)) for _n, _t, opt in self.fields]
        out = {}
        for (name, t, _opt), p in zip(self.fields, present):
            if p:
                if t is None:
                    raise ValueError(f'{name}: not implemented by Falcon\'s decoder')
                out[name] = t.dec(r)
        return out


class Choice:
    """CHOICE: alternatives (name, type); a value is a dict with exactly one key."""

    def __init__(self, alts, ext=False):
        self.alts = alts
        self.ext = ext
        self.n = _nbits(len(alts))

    def enc(self, w, v, path):
        if not isinstance(v, dict) or len(v) != 1:
            raise EncodeError(f'{path}: a CHOICE is a dict with one key')
        (name, val), = v.items()
        names = [a[0] for a in self.alts]
        if name not in names:
            raise EncodeError(f'{path}: {name!r} is not one of {names}')
        k = names.index(name)
        if self.alts[k][1] is None:
            raise EncodeError(f'{path}.{name}: not implemented by Falcon\'s encoder')
        if self.ext:
            w.bits(0, 1)
        w.bits(k, self.n)
        self.alts[k][1].enc(w, val, f'{path}.{name}')

    def dec(self, r):
        if self.ext and r.bits(1):
            raise ValueError('CHOICE extension alternatives are not supported')
        name, t = self.alts[r.bits(self.n)]
        if t is None:
            raise ValueError(f'{name}: not implemented by Falcon\'s decoder')
        return {name: t.dec(r)}


class SeqOf:
    def __init__(self, item, lo: int, hi: int, ext=False):
        self.item, self.lo, self.hi, self.ext = item, lo, hi, ext
        self.n = _nbits(hi - lo + 1)

    def enc(self, w, v, path):
        if not isinstance(v, list):
            raise EncodeError(f'{path}: a SEQUENCE OF needs a list')
        k = len(v)
        inside = self.lo <= k <= self.hi
        if not inside and not self.ext:
            raise EncodeError(f'{path}: {k} items, not {self.lo}..{self.hi}')
        if self.ext:
            w.bits(0 if inside else 1, 1)
        if inside:
            w.bits(k - self.lo, self.n)
        else:
            w.length(k)
        for i, x in enumerate(v):
            self.item.enc(w, x, f'{path}[{i}]')

    def dec(self, r):
        if self.ext and r.bits(1):
            k = r.length()
        else:
            k = self.lo + r.bits(self.n)
        return [self.item.dec(r) for _ in range(k)]


class Open:
    """An open type (here: a WrappedCpmContainer's containerData): the inner type chosen by the containerId."""

    def __init__(self, by_id: dict):
        self.by_id = by_id

    def enc_with(self, w, v, cid, path):
        t = self.by_id.get(cid)
        if t is None:
            raise EncodeError(f'{path}: no container type for id {cid}')
        inner = _W()
        t.enc(inner, v, path)
        data = inner.to_bytes()
        w.length(len(data))
        w.octets(data)

    def dec_with(self, r, cid):
        data = r.octets(r.length())
        t = self.by_id.get(cid)
        if t is None:
            return {'unknown_container_octets': data.hex()}
        return t.dec(_R(data))


class Wrapped(Seq):
    """WrappedCpmContainer: containerId CpmContainerId (1..16), containerData the open type of that id."""

    def __init__(self, by_id):
        super().__init__([('containerId', Int(1, 16), False), ('containerData', None, False)])
        self.open = Open(by_id)

    def enc(self, w, v, path):
        if set(v) != {'containerId', 'containerData'}:
            raise EncodeError(f'{path}: containerId and containerData expected')
        self.fields[0][1].enc(w, v['containerId'], f'{path}.containerId')
        self.open.enc_with(w, v['containerData'], v['containerId'], f'{path}.containerData')

    def dec(self, r):
        cid = self.fields[0][1].dec(r)
        return {'containerId': cid, 'containerData': self.open.dec_with(r, cid)}


# -------------------------------------------------------------------------------------------- the ASN.1 types used
StandardLength12b = Int(0, 4095)
Wgs84AngleValue = Int(0, 3601)
CartesianAngleValue = Int(0, 3601)
CartesianCoordinate = Int(-32768, 32767)
CartesianCoordinateSmall = Int(-3094, 1001)
CartesianCoordinateLarge = Int(-131072, 131071)
CoordinateConfidence = Int(1, 4096)
ConfidenceLevel = Int(1, 101)
Identifier1B = Int(0, 255)
Identifier2B = Int(0, 65535)
CardinalNumber1B = Int(0, 255)
DeltaTimeMilliSecondSigned = Int(-2048, 2047)
SequenceOfIdentifier1B = SeqOf(Identifier1B, 1, 128, ext=True)
CartesianPosition3d = Seq([('xCoordinate', CartesianCoordinate, False), ('yCoordinate', CartesianCoordinate, False),
                           ('zCoordinate', CartesianCoordinate, True)])
RadialShape = Seq([('shapeReferencePoint', CartesianPosition3d, True), ('range', StandardLength12b, False),
                   ('stationaryHorizontalOpeningAngleStart', Wgs84AngleValue, False),
                   ('stationaryHorizontalOpeningAngleEnd', Wgs84AngleValue, False),
                   ('verticalOpeningAngleStart', CartesianAngleValue, True),
                   ('verticalOpeningAngleEnd', CartesianAngleValue, True)])
RadialShapeDetails = Seq([('range', StandardLength12b, False), ('horizontalOpeningAngleStart', CartesianAngleValue, False),
                          ('horizontalOpeningAngleEnd', CartesianAngleValue, False),
                          ('verticalOpeningAngleStart', CartesianAngleValue, True),
                          ('verticalOpeningAngleEnd', CartesianAngleValue, True)])
RadialShapes = Seq([('refPointId', Identifier1B, False), ('xCoordinate', CartesianCoordinateSmall, False),
                    ('yCoordinate', CartesianCoordinateSmall, False), ('zCoordinate', CartesianCoordinateSmall, True),
                    ('radialShapesList', SeqOf(RadialShapeDetails, 1, 16, ext=True), False)])
Shape = Choice([('rectangular', None), ('circular', None), ('polygonal', None), ('elliptical', None),
                ('radial', RadialShape), ('radialShapes', RadialShapes)], ext=True)
ItsPduHeader = Seq([('protocolVersion', Int(0, 255), False), ('messageId', Int(0, 255), False),
                    ('stationId', Int(0, 4294967295), False)])
ALTITUDE_CONFIDENCE = ('alt-000-01', 'alt-000-02', 'alt-000-05', 'alt-000-10', 'alt-000-20', 'alt-000-50', 'alt-001-00',
                       'alt-002-00', 'alt-005-00', 'alt-010-00', 'alt-020-00', 'alt-050-00', 'alt-100-00', 'alt-200-00',
                       'outOfRange', 'unavailable')
ReferencePosition = Seq([
    ('latitude', Int(-900000000, 900000001), False), ('longitude', Int(-1800000000, 1800000001), False),
    ('positionConfidenceEllipse', Seq([('semiMajorConfidence', Int(0, 4095), False),
                                       ('semiMinorConfidence', Int(0, 4095), False),
                                       ('semiMajorOrientation', Int(0, 3601), False)]), False),
    ('altitude', Seq([('altitudeValue', Int(-100000, 800001), False),
                      ('altitudeConfidence', Enum(ALTITUDE_CONFIDENCE), False)]), False)])
MessageRateHz = Seq([('mantissa', Int(1, 100), False), ('exponent', Int(-5, 2), False)])
ManagementContainer = Seq([('referenceTime', Int(0, 4398046511103), False), ('referencePosition', ReferencePosition, False),
                           ('segmentationInfo', Seq([('totalMsgNo', Int(1, 8), False), ('thisMsgNo', Int(1, 8), False)]),
                            True),
                           ('messageRateRange', Seq([('messageRateMin', MessageRateHz, False),
                                                     ('messageRateMax', MessageRateHz, False)]), True)], ext=True)
OriginatingRsuContainer = Seq([('mapReference', None, True)], ext=True)
SensorInformation = Seq([('sensorId', Identifier1B, False), ('sensorType', Int(0, 31), False),
                         ('perceptionRegionShape', Shape, True), ('perceptionRegionConfidence', ConfidenceLevel, True),
                         ('shadowingApplies', Bool(), False)], ext=True)
SensorInformationContainer = SeqOf(SensorInformation, 1, 128, ext=True)
PerceptionRegion = Seq([('measurementDeltaTime', DeltaTimeMilliSecondSigned, False),
                        ('perceptionRegionConfidence', ConfidenceLevel, False), ('perceptionRegionShape', Shape, False),
                        ('shadowingApplies', Bool(), False), ('sensorIdList', SequenceOfIdentifier1B, True),
                        ('numberOfPerceivedObjects', CardinalNumber1B, True),
                        ('perceivedObjectIds', SeqOf(Identifier2B, 0, 255, ext=True), True)], ext=True)
PerceptionRegionContainer = SeqOf(PerceptionRegion, 1, 256, ext=True)
CartesianCoordinateWithConfidence = Seq([('value', CartesianCoordinateLarge, False),
                                         ('confidence', CoordinateConfidence, False)])
VelocityComponent = Seq([('value', Int(-16383, 16383), False), ('confidence', Int(1, 127), False)])
CartesianAngle = Seq([('value', CartesianAngleValue, False), ('confidence', Int(1, 127), False)])
ANGULAR_SPEED_CONFIDENCE = ('degSec-01', 'degSec-02', 'degSec-05', 'degSec-10', 'degSec-20', 'degSec-50', 'outOfRange',
                            'unavailable')
ObjectDimension = Seq([('value', Int(1, 256), False), ('confidence', Int(1, 32), False)])
VRU_PEDESTRIAN = ('unavailable', 'ordinary-pedestrian', 'road-worker', 'first-responder', 'max')
VRU_BICYCLIST = ('unavailable', 'bicyclist', 'wheelchair-user', 'horse-and-rider', 'rollerskater', 'e-scooter',
                 'personal-transporter', 'pedelec', 'speed-pedelec', 'max')
VRU_MOTORCYCLIST = ('unavailable', 'moped', 'motorcycle', 'motorcycle-and-sidecar-right', 'motorcycle-and-sidecar-left',
                    'max')
VRU_ANIMAL = ('unavailable', 'wild-animal', 'farm-animal', 'service-animal', 'max')
ObjectClass = Choice([
    # TrafficParticipantType (unknown|passengerCar..tram|agricultural): PER-visible, effective range 0..14
    ('vehicleSubClass', Int(0, 14)),
    ('vruSubClass', Choice([('pedestrian', Enum(VRU_PEDESTRIAN)), ('bicyclistAndLightVruVehicle', Enum(VRU_BICYCLIST)),
                            ('motorcyclist', Enum(VRU_MOTORCYCLIST)), ('animal', Enum(VRU_ANIMAL))], ext=True)),
    ('groupSubClass', None), ('otherSubClass', Int(0, 255))], ext=True)
PerceivedObject = Seq([
    ('objectId', Identifier2B, True), ('measurementDeltaTime', DeltaTimeMilliSecondSigned, False),
    ('position', Seq([('xCoordinate', CartesianCoordinateWithConfidence, False),
                      ('yCoordinate', CartesianCoordinateWithConfidence, False),
                      ('zCoordinate', CartesianCoordinateWithConfidence, True)]), False),
    ('velocity', Choice([('polarVelocity', None),
                         ('cartesianVelocity', Seq([('xVelocity', VelocityComponent, False),
                                                    ('yVelocity', VelocityComponent, False),
                                                    ('zVelocity', VelocityComponent, True)]))]), True),
    ('acceleration', None, True),
    ('angles', Seq([('zAngle', CartesianAngle, False), ('yAngle', CartesianAngle, True),
                    ('xAngle', CartesianAngle, True)]), True),
    ('zAngularVelocity', Seq([('value', Int(-255, 256), False), ('confidence', Enum(ANGULAR_SPEED_CONFIDENCE), False)]),
     True),
    ('lowerTriangularCorrelationMatrices', None, True),
    ('objectDimensionZ', ObjectDimension, True), ('objectDimensionY', ObjectDimension, True),
    ('objectDimensionX', ObjectDimension, True),
    ('objectAge', Int(0, 2047), True),
    ('objectPerceptionQuality', Int(0, 15), True),
    ('sensorIdList', SequenceOfIdentifier1B, True),
    ('classification', SeqOf(Seq([('objectClass', ObjectClass, False), ('confidence', ConfidenceLevel, False)]), 1, 8),
     True),
    ('mapPosition', None, True)], ext=True)
PerceivedObjectContainer = Seq([('numberOfPerceivedObjects', CardinalNumber1B, False),
                                ('perceivedObjects', SeqOf(PerceivedObject, 0, 255, ext=True), False)], ext=True)
CONTAINER_IDS = {'originatingVehicleContainer': 1, 'originatingRsuContainer': 2, 'sensorInformationContainer': 3,
                 'perceptionRegionContainer': 4, 'perceivedObjectContainer': 5}
CONTAINERS = {2: OriginatingRsuContainer, 3: SensorInformationContainer, 4: PerceptionRegionContainer,
              5: PerceivedObjectContainer}
# ConstraintWrappedCpmContainers is WrappedCpmContainers (SIZE(1..8,...)) with a second, non-extensible constraint
# applied serially (the WITH COMPONENT union): serially applied constraints take the extensibility of the last one
# (X.680), so its length carries no extension bit — as asn1c encodes it (measured: etsi_its_messages 3.5.0 refuses
# the bit); asn1tools 0.169 keeps it
CpmPayload = Seq([('managementContainer', ManagementContainer, False),
                  ('cpmContainers', SeqOf(Wrapped(CONTAINERS), 1, 8, ext=False), False)], ext=True)
CollectivePerceptionMessage = Seq([('header', ItsPduHeader, False), ('payload', CpmPayload, False)])


def encode(cpm: dict) -> bytes:
    """The UPER encoding of a CollectivePerceptionMessage dict (cpm_message's shape). Raises EncodeError naming the
    component when a value is outside its ASN.1 range."""
    w = _W()
    CollectivePerceptionMessage.enc(w, cpm, 'CollectivePerceptionMessage')
    return w.to_bytes()


def decode(data: bytes) -> dict:
    """The inverse of encode for the subset Falcon fills (other alternatives and extension additions are refused)."""
    return CollectivePerceptionMessage.dec(_R(bytes(data)))


def btp_payload(uper: bytes) -> bytes:
    """A BTP-B header (destination port 2009, destination-port info 0; ETSI EN 302 636-5-1) and the CPM: the UDP payload
    the etsi_its_messages conversion node reads with its defaults (has_btp_destination_port, payload offset 4). No
    GeoNetworking and no security header."""
    return BTP_PORT_CPM.to_bytes(2, 'big') + b'\x00\x00' + uper


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode('ascii')


# ================================================================================================ building a message
def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def cm(m: float, lo=-131072, hi=131071) -> int:
    return int(_clamp(int(round(m * 100.0)), lo, hi))


def deci_deg(a: float) -> int:
    """0.1° in 0..3599 (3600 'valueNotUsed' is never written)."""
    v = int(round((a % 360.0) * 10.0))
    return 0 if v >= 3600 else v


def wgs84_heading(yaw_carla: float) -> float:
    """CARLA's yaw (degrees, 0 = +x = East, positive towards +y = South: clockwise seen from above) as a WGS84 heading
    (clockwise from North)."""
    return (90.0 + yaw_carla) % 360.0


def enu_angle(yaw_carla: float) -> float:
    """CARLA's yaw as a CartesianAngle in the ENU frame (counter-clockwise from East)."""
    return (-yaw_carla) % 360.0


def confidence_cm(sigma_m: float) -> int:
    """CoordinateConfidence (95 %, 0.01 m) of a Gaussian error of sigma_m: 1.96 σ, at least 1 (the 1 cm quantisation)."""
    return int(_clamp(math.ceil(1.96 * sigma_m * 100.0 - 1e-9), 1, 4095))


def speed_confidence(sigma_mps: float) -> int:
    return int(_clamp(math.ceil(1.96 * sigma_mps * 100.0 - 1e-9), 1, 126))


def dimension(m: float) -> dict:
    v = int(round(m * 10.0))
    return {'value': 255 if v > 254 else max(v, 1), 'confidence': 1}


# nuScenes category (core.groundtruth.category_of) -> CPM ObjectClass
VEHICLE_CLASS = {'vehicle.car': 5, 'vehicle.bus.bendy': 6, 'vehicle.bus.rigid': 6, 'vehicle.truck': 8,
                 'vehicle.trailer': 9, 'vehicle.construction': 10, 'vehicle.emergency.ambulance': 10,
                 'vehicle.emergency.police': 10}


def object_class(category: str, type_id: str = '') -> dict:
    """The ObjectClass of a road user from its nuScenes category: cars passengerCar, buses bus, trucks heavyTruck
    (a van, CARLA base type 'van', lightTruck), emergency and construction vehicles specialVehicle; motorcycles and
    bicycles are VRUs (TrafficParticipantType's motorcycle, cyclist and pedestrian values are outside ObjectClass's
    vehicleSubClass constraint), pedestrians ordinary pedestrians; anything else unknown."""
    if category.startswith('human.pedestrian'):
        return {'vruSubClass': {'pedestrian': 'ordinary-pedestrian'}}
    if category == 'vehicle.motorcycle':
        return {'vruSubClass': {'motorcyclist': 'motorcycle'}}
    if category == 'vehicle.bicycle':
        return {'vruSubClass': {'bicyclistAndLightVruVehicle': 'bicyclist'}}
    if category in VEHICLE_CLASS:
        return {'vehicleSubClass': VEHICLE_CLASS[category]}
    if category.startswith('vehicle.'):
        return {'vehicleSubClass': 5}
    return {'vehicleSubClass': 0}


def perceived_object(o: dict) -> dict:
    """A PerceivedObject from a Falcon object row (station ENU frame, SI units): object_id, dt_ms, pos (x, y, z), vel
    (vx, vy), yaw_enu_deg, yaw_rate_dps (counter-clockwise), size (length, width, height), age_ms, quality (0..15),
    sensor_ids, category, sigma_pos_m, sigma_vel_mps."""
    cp, cv = confidence_cm(o.get('sigma_pos_m', 0.0)), speed_confidence(o.get('sigma_vel_mps', 0.0))
    x, y, z = o['pos']
    po = {'objectId': int(o['object_id']) & 0xFFFF, 'measurementDeltaTime': int(_clamp(int(o.get('dt_ms', 0)), -2048, 2047)),
          'position': {'xCoordinate': {'value': cm(x), 'confidence': cp},
                       'yCoordinate': {'value': cm(y), 'confidence': cp},
                       'zCoordinate': {'value': cm(z), 'confidence': cp}},
          'velocity': {'cartesianVelocity': {'xVelocity': {'value': cm(o['vel'][0], -16383, 16382), 'confidence': cv},
                                             'yVelocity': {'value': cm(o['vel'][1], -16383, 16382), 'confidence': cv}}},
          'angles': {'zAngle': {'value': deci_deg(o['yaw_enu_deg']), 'confidence': 1}},
          'zAngularVelocity': {'value': int(_clamp(int(round(o.get('yaw_rate_dps', 0.0))), -255, 255)),
                               'confidence': 'degSec-01'},
          'objectDimensionZ': dimension(o['size'][2]), 'objectDimensionY': dimension(o['size'][1]),
          'objectDimensionX': dimension(o['size'][0]),
          'objectAge': int(_clamp(int(o.get('age_ms', 0)), 0, 2047)),
          'objectPerceptionQuality': int(_clamp(int(o.get('quality', 15)), 0, 15)),
          'classification': [{'objectClass': object_class(o.get('category', '')), 'confidence': 100}]}
    if o.get('sensor_ids'):
        po['sensorIdList'] = sorted({int(s) for s in o['sensor_ids']})[:128]
    return po


def sensor_information(sid: int, kind: str, hdg_wgs_deg: float, hfov_deg: float, range_m: float,
                       pitch_deg: float = 0.0, vfov_deg: float | None = None, offset_enu=None) -> dict:
    """A SensorInformation: sensorType (monovideo 3, lidar 2, radar 1), its field of view as a RadialShape — the
    stationary horizontal opening clockwise from North (WGS84), the vertical opening (elevation, counter-clockwise
    from the horizontal), the range — around its mount (shapeReferencePoint: the offset from the reference position, cm)."""
    t = {'camera': 3, 'lidar': 2, 'radar': 1}[kind]
    if hfov_deg >= 359.9:
        a0, a1 = 0, 3599
    else:
        a0, a1 = deci_deg(hdg_wgs_deg - hfov_deg / 2.0), deci_deg(hdg_wgs_deg + hfov_deg / 2.0)
    shape = {'range': int(_clamp(int(round(range_m * 10.0)), 0, 4095)),
             'stationaryHorizontalOpeningAngleStart': a0, 'stationaryHorizontalOpeningAngleEnd': a1}
    if offset_enu is not None:
        shape['shapeReferencePoint'] = {'xCoordinate': cm(offset_enu[0], -32768, 32767),
                                        'yCoordinate': cm(offset_enu[1], -32768, 32767),
                                        'zCoordinate': cm(offset_enu[2], -32768, 32767)}
    if vfov_deg is not None:
        shape['verticalOpeningAngleStart'] = deci_deg(pitch_deg - vfov_deg / 2.0)
        shape['verticalOpeningAngleEnd'] = deci_deg(pitch_deg + vfov_deg / 2.0)
    return {'sensorId': int(sid), 'sensorType': t, 'perceptionRegionShape': {'radial': shape}, 'shadowingApplies': True}


def perception_region(sid: int, offset_enu, sectors: list, object_ids: list) -> dict:
    """A PerceptionRegion of one sensor: its field of view cut, sector by sector, at the nearest static occluder it saw
    (a RadialShapes list: range, horizontal opening clockwise from North in CartesianAngleValue's 0.1°) — the area it
    perceived; shadowingApplies true: the area behind each reported object is not claimed free. Confidence 101
    (unavailable: Falcon has no calibrated figure). sectors: [(range_m, start_deg_wgs, end_deg_wgs)]."""
    ox, oy, oz = offset_enu
    rs = {'refPointId': 0, 'xCoordinate': cm(ox, -3094, 1001), 'yCoordinate': cm(oy, -3094, 1001),
          'radialShapesList': [{'range': int(_clamp(int(round(r * 10.0)), 0, 4095)),
                                'horizontalOpeningAngleStart': deci_deg(a0), 'horizontalOpeningAngleEnd': deci_deg(a1)}
                               for r, a0, a1 in sectors]}
    if -30.94 <= oz <= 10.01:
        rs['zCoordinate'] = cm(oz, -3094, 1001)
    reg = {'measurementDeltaTime': 0, 'perceptionRegionConfidence': 101, 'perceptionRegionShape': {'radialShapes': rs},
           'shadowingApplies': True, 'sensorIdList': [int(sid)], 'numberOfPerceivedObjects': min(len(object_ids), 255)}
    if object_ids:
        reg['perceivedObjectIds'] = sorted({int(i) & 0xFFFF for i in object_ids})[:255]
    return reg


def rate_hz(hz: float) -> dict:
    """MessageRateHz (mantissa × 10^exponent) of a rate: the mantissa within 1..100."""
    e = 0
    m = float(hz)
    while m > 100 and e < 2:
        m /= 10.0
        e += 1
    while m < 10 and e > -5 and abs(m - round(m)) > 1e-9:
        m *= 10.0
        e -= 1
    return {'mantissa': int(_clamp(int(round(m)), 1, 100)), 'exponent': e}


def cpm_message(station_id: int, reference_time_ms: int, lat_deg: float, lon_deg: float, alt_m: float,
                sensors: list, regions: list, objects: list, rate: float | None = None) -> dict:
    """A CollectivePerceptionMessage dict: the RSU's header and management container (its surveyed position, 1 cm
    confidence), an empty originatingRsuContainer, then the sensor information, the perception regions (when any) and
    the perceived objects (at most 255: the nearest are kept by the caller)."""
    mgmt = {'referenceTime': int(reference_time_ms),
            'referencePosition': {'latitude': int(round(lat_deg * 1e7)), 'longitude': int(round(lon_deg * 1e7)),
                                  'positionConfidenceEllipse': {'semiMajorConfidence': 1, 'semiMinorConfidence': 1,
                                                                'semiMajorOrientation': 0},
                                  'altitude': {'altitudeValue': int(_clamp(int(round(alt_m * 100)), -100000, 800000)),
                                               'altitudeConfidence': 'alt-000-01'}}}
    if rate:
        r = rate_hz(rate)
        mgmt['messageRateRange'] = {'messageRateMin': r, 'messageRateMax': r}
    conts = [{'containerId': 2, 'containerData': {}}]
    if sensors:
        conts.append({'containerId': 3, 'containerData': sensors[:128]})
    if regions:
        conts.append({'containerId': 4, 'containerData': regions[:256]})
    objs = objects[:255]
    conts.append({'containerId': 5, 'containerData': {'numberOfPerceivedObjects': len(objs), 'perceivedObjects': objs}})
    return {'header': {'protocolVersion': PROTOCOL_VERSION, 'messageId': MESSAGE_ID_CPM, 'stationId': int(station_id)},
            'payload': {'managementContainer': mgmt, 'cpmContainers': conts}}


def containers(cpm: dict) -> dict:
    """{'originatingRsuContainer': data, …} of a CPM dict, by name."""
    names = {v: k for k, v in CONTAINER_IDS.items()}
    return {names.get(c['containerId'], str(c['containerId'])): c['containerData']
            for c in cpm['payload']['cpmContainers']}
