# Data lineage

Every dataset version and every trained model names the exact bytes it was made from: a content address for each
frame and its labels, and a Merkle root over all of them — in the version's splits file and in the model's ONNX
metadata. `falcon lineage verify` checks a model or a version against the files; `falcon lineage prove` shows that one
frame is among a model's training samples with a handful of hashes. Code: `falcon_autonomy/core/cas.py` (the store),
`falcon_autonomy/core/lineage.py` (indexes, roots, proofs, the commands; their docstrings are the reference).

```sh
falcon lineage index RUN...                       # hash a run's frames into RUN/frames.parquet (run.json untouched)
falcon lineage verify MODEL|VERSION|RUN [--quick] # re-check: ONNX metadata, data.json, images re-hashed / the store
falcon lineage prove MODEL FRAME [--split train]  # an RFC 6962 audit path: FRAME is an image, a sample id or an address
falcon lineage diff A B                           # frames only in A, only in B, relabelled
falcon lineage materialise VERSION DIR [--hardlink|--reflink|--copy|--symlink] [--split S]
falcon lineage gc WORKSPACE [--dry-run] [--grace S]
```

In the GUI: Models ▸ **Lineage** (every model and dataset version with its root; Verify, Prove a frame…, Index runs,
Store), and Data ▸ Datasets shows each version's root beside its SHA-256.

## Addresses

An address is `b3:<64 hex>` (BLAKE3, when the `blake3` package is installed — the extra `falcon-autonomy[lineage]`)
or `s256:<64 hex>` (SHA-256, always available). The prefix says which: an index, a version or a model made with one is
checked with the same one, and a BLAKE3 address cannot be checked without the package (the error says so). On the
reference machine blake3 is not installed: everything is SHA-256.

## The store

`<workspace>/cas/objects/<first two hex>/<hex>`, each object read-only (mode 0444). A file is put in by the cheapest
way the file system allows: a **reflink** (the FICLONE ioctl: a copy-on-write clone on btrfs, XFS, bcachefs — no data
copied, the original stays writable), else a **hard link** (no data copied; the store's object *is* the run's file,
which therefore becomes read-only too — Falcon never edits a recorded frame in place, and a rewrite by rename makes a
new file and leaves the object alone), else a **copy** (another file system). ext4 has no reflinks: measured on this
machine, FICLONE fails with EOPNOTSUPP and every put is a hard link. The address is computed from the bytes as stored,
after the 0444 bit is set, so an object can never be named after bytes it does not hold. Identical bytes are stored
once.

**Garbage collection** is mark and sweep: the roots are every dataset version's splits file (its frames, label files
and label sets), every run index and every training's data.json; objects none of them names are removed, except any
linked within the last hour (`--grace`: a put whose reference is not yet written). A root that cannot be parsed stops
the sweep before anything is removed. `--dry-run` lists what would go.

## The run index

`falcon lineage index RUN` writes `RUN/frames.parquet`: one row per recorded step and sensor — `frame` (the address
of the camera's rgb image, the lidar sweep or the radar return), `label` (the address of the frame's label set: the
canonical JSON `{kind: address}` of its semantic, instance, range, objects, flow, lidar labels and radar flags), each
file's path, size and modification time. run.json is never modified; the summary (algorithm, root, counts) is in the
parquet file's metadata. A second index reads only the files whose size or time changed. After a run finishes, the
index is a cheap CPU job of its own (`lineage.after_job`, queued by the service).

## Merkle roots

RFC 6962 §2.1 (Certificate Transparency): a leaf hashes as H(0x00 ‖ data), a node as H(0x01 ‖ left ‖ right); the tree
pairs a level's nodes left to right and promotes an odd last node unchanged, which is the same tree as RFC 6962's
split at the largest power of two. A dataset's leaves are its samples' `"<frame address>|<label address>"` (ASCII),
**sorted** (the root does not depend on listing order; a sample listed twice counts twice). The empty tree's root is
H(""). The tests check RFC 6962's published roots for one to eight leaves and its audit paths.

`prove` gives the leaf's index, the ⌈log₂ n⌉ sibling hashes and the root, checked with RFC 9162 §2.1.3.2's algorithm.

## Dataset versions

Making a version (Data ▸ Datasets ▸ New version, `sim.engine.new_version`, `core.datasets.make_version`) also indexes
its runs, puts every frame, label file and label set into the store and writes
`datasets/<spec>/splits/vNNN.parquet` (split, run, step, sensor, frame and label addresses, the label files). The
version's JSON gains `lineage`: the splits file, the hash algorithm, the root of the whole version and of each split,
the sample and object counts; how the files were stored (reflink / hard link) is in its provenance. Its SHA-256 now
covers the roots. `materialise` rebuilds the version's files from the store — after its runs are gone, too.

Without pyarrow (the extra `lineage`), a version is made without lineage and says so in its provenance notes.

## Models

A training (`falcon train`, Training page) hashes every sample — `frame`, the image file's bytes (read once: an
indexed run's unchanged files are taken from its index); `label`, the labels as trained (the canonical JSON of the
sample's boxes, classes and ignore flags) — and data.json keeps both per image, each split's root, `hash_alg`,
`manifest_sha` (the SHA-256 that keys the training's settings) and `dataset_version` (the version a `version:` source
names, or "none"). The exported `model.onnx` carries, in its `metadata_props`:

| key | value |
|---|---|
| `falcon.data_merkle_root` | the training split's root |
| `falcon.val_merkle_root` | the validation split's root |
| `falcon.hash_alg` | `s256` or `b3` |
| `falcon.dataset_version` | `<name> v<k> (<spec> sha256 <12 hex>)`, or `none` |
| `falcon.manifest_sha` | data.json's `manifest_sha` |

written before the file's SHA-256 is computed, so the registration (`model.json`'s `source.sha256`) is of the file that
names its data. `verify` checks that metadata against data.json, data.json's roots against its per-sample addresses,
the registration's SHA-256 against the file and (without `--quick`) re-hashes every training image still on disk.

The catalogue indexes dataset versions (`dataset:<spec>@v<k>`) and trained models (`model:<training id>`) with their
roots, and links a model to the version it was trained on (`trained_on`).

## Limits

- A frame's address is its file's bytes: re-encoding an image changes it. Sensors' jsonl streams (GNSS, IMU) and the
  run's own JSON files are not leaves.
- The labels in a model's leaves are the labels **as trained** (after the class selection and the harness's ignore
  rule); a version's leaves are the label **files**. `diff` compares frames between the two kinds but labels only
  between like kinds.
- Hard links make a run's put files read-only (they are the store's objects).
- A training started before lineage (data.json without addresses) is reported as such, not verified.
