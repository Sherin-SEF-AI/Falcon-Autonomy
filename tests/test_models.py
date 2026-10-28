"""Model registration, decoders and runners: real ONNX graphs built here (onnx.helper), run by onnxruntime."""
import hashlib
import os

import numpy as np
import pytest

from falcon_autonomy.core import models as M

onnx = pytest.importorskip('onnx')
pytest.importorskip('onnxruntime')
from onnx import TensorProto, helper  # noqa: E402


def _const_model(path, outputs: dict, w=64, h=32):
    """An ONNX model whose image input feeds a ReduceMean (so it is used) and whose outputs are constants."""
    inp = helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, h, w])
    nodes, outs = [helper.make_node('ReduceMean', ['images'], ['m'], keepdims=0)], []
    for name, arr in outputs.items():
        arr = np.asarray(arr)
        dt = TensorProto.INT64 if arr.dtype.kind in 'iu' else TensorProto.FLOAT
        t = helper.make_tensor(name + '_c', dt, arr.shape, arr.astype(np.int64 if dt == TensorProto.INT64 else
                                                                        np.float32).ravel().tolist())
        nodes.append(helper.make_node('Constant', [], [name], value=t))
        outs.append(helper.make_tensor_value_info(name, dt, list(arr.shape)))
    g = helper.make_graph(nodes, 'g', [inp], outs + [helper.make_tensor_value_info('m', TensorProto.FLOAT, [])])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 8
    onnx.save(model, path)
    return path


def test_boxes_scores_labels_through_onnxruntime(tmp_path):
    p = _const_model(str(tmp_path / 'det.onnx'), {
        'boxes': [[10, 4, 30, 20], [11, 4, 31, 20], [40, 10, 60, 30]],      # in the model's 64 × 32 input
        'scores': [0.9, 0.8, 0.2], 'labels': [0, 0, 1]})
    m = {'kind': 'detector', 'runtime': 'onnx', 'path': p, 'input': {'width': 64, 'height': 32},
         'output': {'format': 'boxes_scores_labels', 'score_threshold': 0.3, 'nms_iou': 0.5},
         'classes': ['vehicle', 'person'], 'class_map': {'vehicle': 'car', 'person': 'pedestrian'},
         'licence': 'Apache-2.0'}
    r = M.Runner(m)
    assert 'CPUExecutionProvider' in r.providers
    dets = r.predict(np.zeros((64, 128, 3), np.uint8))                   # the image is twice the input size
    assert len(dets) == 1 and dets[0]['class'] == 'car' and dets[0]['score'] == pytest.approx(0.9)
    assert dets[0]['bbox'] == pytest.approx([20, 8, 60, 40])


def test_yolo_layout_and_nms():
    nc = 2
    rows = np.zeros((4 + nc, 3))                                           # (4 + nc, N) as YOLOv8 writes it
    rows[:, 0] = [32, 16, 20, 10, 0.95, 0.05]
    rows[:, 1] = [33, 16, 20, 10, 0.70, 0.10]                              # suppressed by NMS
    rows[:, 2] = [10, 10, 4, 4, 0.05, 0.60]
    m = {'output': {'format': 'yolo', 'score_threshold': 0.25, 'nms_iou': 0.5}, 'classes': ['car', 'person']}
    back = {'kx': 1.0, 'ky': 1.0, 'px': 0, 'py': 0, 'model_w': 64, 'model_h': 32}
    dets = M.decode_detections([rows[None]], m, back)
    assert sorted(d['class'] for d in dets) == ['car', 'person']
    car = next(d for d in dets if d['class'] == 'car')
    assert car['bbox'] == pytest.approx([22, 11, 42, 21])


def test_letterbox_round_trip():
    img = np.zeros((90, 160, 3), np.uint8)
    x, back = M.preprocess(img, {'width': 64, 'height': 64, 'letterbox': True})
    assert x.shape == (1, 3, 64, 64)
    # a box covering the whole image in model pixels maps back to the whole image
    b = M.to_image([[0, back['py'], 64, 64 - back['py']]], back)
    assert b[0] == pytest.approx([0, 0, 160, 90], abs=1.5)


def test_segmentation_logits():
    logits = np.zeros((1, 3, 4, 4), np.float32)
    logits[0, 2, :2] = 5.0
    logits[0, 1, 2:] = 5.0
    m = {'classes': ['bg', 'road', 'car'], 'class_map': {'road': 1, 'car': 14}}
    back = {'kx': 1, 'ky': 1, 'px': 0, 'py': 0, 'model_w': 4, 'model_h': 4}
    lab = M.decode_segmentation([logits], m, back, (8, 8))
    assert lab.shape == (8, 8) and (lab[:4] == 14).all() and (lab[4:] == 1).all()
    by_name = dict(m, class_map={'road': 'road', 'car': 'car'})           # names, as the harness and docs use
    assert (M.decode_segmentation([logits], by_name, back, (8, 8)) == lab).all()
    assert (M.decode_segmentation([logits], dict(m, class_map={}), back, (8, 8)) == lab).all()   # Falcon names


def test_validation_refuses_agpl_and_fetch_checks_the_hash(tmp_path):
    assert any('AGPL' in p for p in M.validate({'kind': 'detector', 'runtime': 'onnx', 'path': 'x.onnx',
                                                 'output': {'format': 'yolo'}, 'licence': 'AGPL-3.0'}))
    src = tmp_path / 'model.bin'
    src.write_bytes(b'weights')
    good = hashlib.sha256(b'weights').hexdigest()
    m = {'path': 'downloaded.bin', 'source': {'url': src.as_uri(), 'sha256': good}}
    path = M.fetch(m, str(tmp_path / 'ws'))
    assert open(path, 'rb').read() == b'weights'
    m['source']['sha256'] = '0' * 64
    m['path'] = 'other.bin'
    with pytest.raises(ValueError, match='mismatch'):
        M.fetch(m, str(tmp_path / 'ws'))
    assert not os.path.exists(os.path.join(tmp_path, 'ws', 'models', 'other.bin'))
