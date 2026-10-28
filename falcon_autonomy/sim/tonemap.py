"""CARLA's camera images in linear light: its RGB camera's response and Unreal Engine's vignette, so the bridge can
take the vignette out of every pixel.

Every CARLA 0.9.16 RGB camera passes its image through Unreal Engine 4's tone-mapping pass, which darkens it towards
the edges with UE's default vignette (PostProcessSettings.VignetteIntensity 0.4; not one of the camera's blueprint
attributes, and only gone with all post-processing, tone curve included). In linear light the mask is

    m(r) = 1 / (1 + (0.4 r)²)²                    (UE 4.26 PostProcessCommon.ush, ComputeVignetteMask)

with r the pixel's distance from the image centre in UE's vignette space: x from −1 to 1 across the width, y scaled by
height / width, both by √2 / √(1 + (h/w)²), so the corners lie at √2 for any aspect. It is applied before the tone
curve: the edge midpoints of a square image 26 % darker in linear light, the corners 43 %. A real lens vignettes too,
but this one is the same in every pinhole view, and the bridge renders a fisheye camera as several views: each view
darkens towards its own edges, and where two views meet the image is about 25 % brighter on one side of a line than
on the other.

Measured on a CARLA 0.9.16 server (Town10HD, 2026-10-27), with the bridge's camera attributes (manual exposure, no
motion blur, bloom, lens flare or CARLA lens effect):
- RESPONSE_LN, the camera's response: per 8-bit output level, ln of its relative exposure in linear light, one curve
  per channel in raw_data's order (B, G, R), monotone. Fitted by Debevec and Malik's method (SIGGRAPH 1997) to frames
  at exposure compensations −3 … +3 EV of three scenes, 3000 pixels of each, every frame the mean of 16. Fitted on
  two scenes, it predicts the third one's frames from its EV 0 frame within 0.25–0.45 output levels (median) at ±0.5
  and ±1 EV, 0.9–1.1 at −3 EV. The vignette correction moves a pixel by at most 0.8 EV.
- The vignette: pairs of views from one point that overlap (640 × 640 at 90°, and 960 × 540 at 100°): in linear light
  the ratio of one scene point's two values follows m(r) with intensity 0.398, UE's default 0.4. With m taken out, the
  median log ratio of the views at the edges falls from ±0.21–0.28 to below 0.008.

mask_of_remap() gives every output pixel of a camera the mask at the view pixel it reads (through the rig's remap
tables), and Devignette.apply() replaces each level z with T(T⁻¹(z) / m), T the response, through one look-up table
per bin of m (BINS bins, uniform in ln m: at most 0.2 % off in linear light). Levels 0 and 255 stay as they are: what
the view clipped cannot be recovered. Needs numpy only (it runs in CARLA's Python).
"""
from __future__ import annotations

import math

import numpy as np

VIGNETTE_INTENSITY = 0.4
BINS = 128

# ln of the relative linear exposure per output level 0 … 255 (see the module doc), B, G, R
RESPONSE_LN = np.array((
    # B
    (-2.7199, -2.6770, -2.6344, -2.5922, -2.5503, -2.5088, -2.4675, -2.4267, -2.3862, -2.3462, -2.3065, -2.2672,
     -2.2285, -2.1902, -2.1524, -2.1153, -2.0789, -2.0431, -2.0080, -1.9735, -1.9396, -1.9063, -1.8737, -1.8419,
     -1.8109, -1.7806, -1.7512, -1.7223, -1.6942, -1.6667, -1.6397, -1.6133, -1.5874, -1.5619, -1.5368, -1.5121,
     -1.4880, -1.4645, -1.4417, -1.4196, -1.3981, -1.3773, -1.3571, -1.3372, -1.3178, -1.2986, -1.2797, -1.2609,
     -1.2421, -1.2234, -1.2046, -1.1858, -1.1670, -1.1482, -1.1294, -1.1107, -1.0922, -1.0738, -1.0556, -1.0375,
     -1.0195, -1.0017, -0.9840, -0.9664, -0.9489, -0.9315, -0.9140, -0.8965, -0.8788, -0.8610, -0.8431, -0.8249,
     -0.8067, -0.7883, -0.7699, -0.7515, -0.7331, -0.7148, -0.6966, -0.6785, -0.6604, -0.6423, -0.6242, -0.6061,
     -0.5880, -0.5699, -0.5519, -0.5339, -0.5159, -0.4980, -0.4802, -0.4623, -0.4445, -0.4266, -0.4088, -0.3909,
     -0.3729, -0.3550, -0.3370, -0.3191, -0.3011, -0.2832, -0.2653, -0.2474, -0.2295, -0.2116, -0.1937, -0.1759,
     -0.1582, -0.1405, -0.1230, -0.1055, -0.0880, -0.0706, -0.0533, -0.0361, -0.0188, -0.0016, 0.0156, 0.0328,
     0.0500, 0.0672, 0.0843, 0.1013, 0.1184, 0.1355, 0.1526, 0.1698, 0.1870, 0.2042, 0.2214, 0.2385,
     0.2554, 0.2722, 0.2889, 0.3054, 0.3218, 0.3382, 0.3544, 0.3706, 0.3867, 0.4027, 0.4188, 0.4349,
     0.4510, 0.4672, 0.4834, 0.4996, 0.5157, 0.5318, 0.5478, 0.5637, 0.5796, 0.5955, 0.6114, 0.6273,
     0.6433, 0.6593, 0.6755, 0.6918, 0.7082, 0.7248, 0.7415, 0.7582, 0.7749, 0.7916, 0.8082, 0.8247,
     0.8411, 0.8573, 0.8735, 0.8896, 0.9055, 0.9215, 0.9374, 0.9533, 0.9692, 0.9852, 1.0013, 1.0176,
     1.0341, 1.0507, 1.0675, 1.0844, 1.1014, 1.1183, 1.1351, 1.1519, 1.1686, 1.1853, 1.2019, 1.2185,
     1.2352, 1.2520, 1.2689, 1.2860, 1.3033, 1.3210, 1.3389, 1.3571, 1.3755, 1.3941, 1.4128, 1.4316,
     1.4506, 1.4697, 1.4890, 1.5086, 1.5283, 1.5483, 1.5686, 1.5892, 1.6102, 1.6315, 1.6534, 1.6757,
     1.6986, 1.7220, 1.7461, 1.7708, 1.7961, 1.8220, 1.8485, 1.8756, 1.9032, 1.9315, 1.9606, 1.9907,
     2.0218, 2.0543, 2.0881, 2.1234, 2.1603, 2.1990, 2.2395, 2.2821, 2.3266, 2.3732, 2.4218, 2.4725,
     2.5256, 2.5817, 2.6412, 2.7047, 2.7731, 2.8470, 2.9270, 3.0133, 3.1065, 3.2072, 3.3163, 3.4347,
     3.5630, 3.7012, 3.8492, 4.0060),
    # G
    (-2.8683, -2.8263, -2.7846, -2.7432, -2.7023, -2.6616, -2.6214, -2.5815, -2.5419, -2.5026, -2.4637, -2.4252,
     -2.3872, -2.3498, -2.3129, -2.2768, -2.2414, -2.2068, -2.1728, -2.1395, -2.1068, -2.0746, -2.0431, -2.0121,
     -1.9817, -1.9519, -1.9227, -1.8940, -1.8659, -1.8383, -1.8112, -1.7847, -1.7586, -1.7330, -1.7079, -1.6834,
     -1.6595, -1.6363, -1.6139, -1.5920, -1.5708, -1.5502, -1.5300, -1.5101, -1.4906, -1.4713, -1.4521, -1.4331,
     -1.4141, -1.3953, -1.3764, -1.3576, -1.3389, -1.3203, -1.3019, -1.2837, -1.2657, -1.2479, -1.2302, -1.2127,
     -1.1953, -1.1779, -1.1605, -1.1432, -1.1257, -1.1082, -1.0906, -1.0728, -1.0549, -1.0369, -1.0187, -1.0004,
     -0.9819, -0.9635, -0.9451, -0.9268, -0.9086, -0.8905, -0.8725, -0.8546, -0.8368, -0.8189, -0.8011, -0.7832,
     -0.7652, -0.7472, -0.7292, -0.7113, -0.6934, -0.6756, -0.6579, -0.6403, -0.6228, -0.6054, -0.5881, -0.5708,
     -0.5535, -0.5362, -0.5189, -0.5016, -0.4842, -0.4668, -0.4493, -0.4318, -0.4143, -0.3967, -0.3792, -0.3616,
     -0.3441, -0.3266, -0.3092, -0.2919, -0.2746, -0.2575, -0.2404, -0.2234, -0.2064, -0.1894, -0.1722, -0.1550,
     -0.1376, -0.1202, -0.1027, -0.0852, -0.0676, -0.0500, -0.0325, -0.0149, 0.0026, 0.0200, 0.0374, 0.0545,
     0.0714, 0.0882, 0.1048, 0.1213, 0.1377, 0.1539, 0.1701, 0.1862, 0.2023, 0.2184, 0.2346, 0.2508,
     0.2671, 0.2835, 0.2999, 0.3163, 0.3327, 0.3490, 0.3651, 0.3811, 0.3970, 0.4128, 0.4285, 0.4443,
     0.4601, 0.4760, 0.4920, 0.5082, 0.5245, 0.5410, 0.5575, 0.5741, 0.5908, 0.6074, 0.6239, 0.6404,
     0.6567, 0.6730, 0.6893, 0.7055, 0.7217, 0.7378, 0.7539, 0.7701, 0.7862, 0.8024, 0.8187, 0.8352,
     0.8518, 0.8685, 0.8854, 0.9023, 0.9193, 0.9363, 0.9533, 0.9703, 0.9873, 1.0044, 1.0214, 1.0383,
     1.0553, 1.0722, 1.0891, 1.1061, 1.1233, 1.1407, 1.1582, 1.1760, 1.1940, 1.2122, 1.2307, 1.2494,
     1.2683, 1.2875, 1.3071, 1.3269, 1.3470, 1.3674, 1.3880, 1.4089, 1.4300, 1.4514, 1.4731, 1.4953,
     1.5180, 1.5413, 1.5652, 1.5898, 1.6150, 1.6409, 1.6674, 1.6947, 1.7227, 1.7514, 1.7809, 1.8113,
     1.8429, 1.8756, 1.9096, 1.9450, 1.9820, 2.0209, 2.0617, 2.1046, 2.1496, 2.1967, 2.2456, 2.2966,
     2.3499, 2.4060, 2.4653, 2.5286, 2.5967, 2.6704, 2.7503, 2.8368, 2.9308, 3.0333, 3.1452, 3.2678,
     3.4022, 3.5492, 3.7089, 3.8808),
    # R
    (-3.0989, -3.0581, -3.0176, -2.9773, -2.9373, -2.8975, -2.8580, -2.8188, -2.7799, -2.7413, -2.7031, -2.6654,
     -2.6281, -2.5913, -2.5551, -2.5195, -2.4846, -2.4503, -2.4166, -2.3835, -2.3510, -2.3190, -2.2876, -2.2568,
     -2.2266, -2.1969, -2.1678, -2.1393, -2.1112, -2.0838, -2.0568, -2.0304, -2.0045, -1.9790, -1.9541, -1.9297,
     -1.9058, -1.8826, -1.8599, -1.8378, -1.8161, -1.7950, -1.7742, -1.7539, -1.7340, -1.7143, -1.6948, -1.6755,
     -1.6563, -1.6371, -1.6179, -1.5987, -1.5794, -1.5602, -1.5408, -1.5215, -1.5022, -1.4829, -1.4636, -1.4443,
     -1.4251, -1.4059, -1.3868, -1.3678, -1.3489, -1.3300, -1.3112, -1.2924, -1.2736, -1.2548, -1.2359, -1.2170,
     -1.1981, -1.1792, -1.1603, -1.1415, -1.1226, -1.1039, -1.0852, -1.0665, -1.0479, -1.0293, -1.0107, -0.9921,
     -0.9735, -0.9548, -0.9362, -0.9175, -0.8988, -0.8802, -0.8617, -0.8432, -0.8249, -0.8067, -0.7885, -0.7705,
     -0.7525, -0.7345, -0.7166, -0.6987, -0.6807, -0.6627, -0.6446, -0.6265, -0.6084, -0.5902, -0.5721, -0.5540,
     -0.5360, -0.5181, -0.5003, -0.4826, -0.4651, -0.4477, -0.4303, -0.4131, -0.3959, -0.3788, -0.3617, -0.3446,
     -0.3275, -0.3105, -0.2933, -0.2762, -0.2590, -0.2417, -0.2244, -0.2070, -0.1896, -0.1723, -0.1549, -0.1376,
     -0.1204, -0.1032, -0.0862, -0.0693, -0.0526, -0.0359, -0.0193, -0.0028, 0.0137, 0.0302, 0.0467, 0.0632,
     0.0798, 0.0963, 0.1129, 0.1295, 0.1461, 0.1627, 0.1791, 0.1954, 0.2117, 0.2280, 0.2442, 0.2605,
     0.2767, 0.2929, 0.3091, 0.3253, 0.3414, 0.3577, 0.3740, 0.3903, 0.4066, 0.4229, 0.4392, 0.4555,
     0.4717, 0.4878, 0.5039, 0.5201, 0.5361, 0.5522, 0.5684, 0.5845, 0.6008, 0.6170, 0.6334, 0.6498,
     0.6664, 0.6830, 0.6997, 0.7165, 0.7334, 0.7502, 0.7671, 0.7839, 0.8006, 0.8172, 0.8339, 0.8505,
     0.8672, 0.8839, 0.9007, 0.9177, 0.9349, 0.9523, 0.9699, 0.9878, 1.0058, 1.0242, 1.0427, 1.0615,
     1.0805, 1.0998, 1.1191, 1.1386, 1.1582, 1.1779, 1.1978, 1.2179, 1.2381, 1.2587, 1.2797, 1.3011,
     1.3232, 1.3458, 1.3692, 1.3932, 1.4180, 1.4435, 1.4695, 1.4961, 1.5232, 1.5508, 1.5792, 1.6084,
     1.6388, 1.6704, 1.7034, 1.7380, 1.7742, 1.8123, 1.8523, 1.8942, 1.9380, 1.9839, 2.0317, 2.0815,
     2.1336, 2.1885, 2.2467, 2.3089, 2.3758, 2.4481, 2.5264, 2.6111, 2.7032, 2.8042, 2.9157, 3.0398,
     3.1786, 3.3334, 3.5054, 3.6952),
))


def vignette_radius(u, v, w: int, h: int) -> np.ndarray:
    """UE's vignette-space radius at image coordinates (u, v) of a w × h image (pixels, 0 = the first pixel's centre,
    as cv2.remap reads them)."""
    a = h / w
    sc = math.sqrt(2.0) / math.sqrt(1.0 + a * a)
    px = (np.asarray(u, float) + 0.5) / w * 2.0 - 1.0
    py = (np.asarray(v, float) + 0.5) / h * 2.0 - 1.0
    return np.hypot(px * sc, py * a * sc)


def vignette_mask(r, intensity: float = VIGNETTE_INTENSITY) -> np.ndarray:
    """UE's vignette mask at vignette-space radius r: the factor linear light is multiplied by."""
    return 1.0 / (1.0 + (intensity * np.asarray(r, float)) ** 2) ** 2


M_LO = float(vignette_mask(math.sqrt(2.0))) * 0.999           # the corners' mask, a hair below


def mask_of_remap(map_x, map_y, offsets, sizes) -> np.ndarray:
    """(h, w) float32: the vignette mask at the view pixel each output pixel reads, the views stacked vertically in one
    atlas (view k's rows from offsets[k], sizes[k] = (width, height)); 1 where the pixel reads nothing (map −1)."""
    mx = np.asarray(map_x, np.float32)
    my = np.asarray(map_y, np.float32)
    out = np.ones(mx.shape, np.float32)
    seen = mx >= -0.5                                   # (−1: the remap tables' 'nothing')
    for y0, (w, h) in zip(offsets, sizes):
        sel = seen & (my >= y0 - 0.5) & (my < y0 + h - 0.5)
        out[sel] = vignette_mask(vignette_radius(mx[sel], my[sel] - y0, int(w), int(h)))
    return out


def correction_lut(bins: int = BINS) -> np.ndarray:
    """(bins, 3, 256) uint8: per mask bin and channel, the level z becomes T(T⁻¹(z) / m)."""
    lm = np.linspace(math.log(M_LO), 0.0, bins)
    z = np.arange(256, dtype=float)
    out = np.empty((bins, 3, 256), np.uint8)
    for c in range(3):
        g = RESPONSE_LN[c]
        for k, l in enumerate(lm):
            out[k, c] = np.clip(np.rint(np.interp(g - l, g, z)), 0, 255)       # beyond 255's exposure: 255
    out[:, :, 0] = 0                                    # clipped black (its exposure is anything below level 1's)
    out[:, :, 255] = 255
    return out


_LUT = None


class Devignette:
    """Takes the vignette out of a camera's output images, given its per-pixel mask (mask_of_remap)."""

    def __init__(self, mask):
        global _LUT
        if _LUT is None:
            _LUT = correction_lut()
        k = np.rint((np.log(np.clip(np.asarray(mask, float), M_LO, 1.0)) - math.log(M_LO)) /
                    -math.log(M_LO) * (BINS - 1)).astype(np.int32)
        self.base = k * 256
        self.flat = [np.ascontiguousarray(_LUT[:, c, :]).ravel() for c in range(3)]

    def apply(self, img: np.ndarray) -> np.ndarray:
        """img (h, w, 3) uint8 in raw_data's channel order (B, G, R): a new image without the vignette."""
        out = np.empty_like(img)
        for c in range(3):
            out[..., c] = self.flat[c][self.base + img[..., c]]
        return out
