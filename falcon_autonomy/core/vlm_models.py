"""Vision-language model runners for core.vlm, and the small open VLMs Falcon can download when asked (numpy and the
standard library at import; torch, transformers and PIL only inside the runners).

A runner answers a question about images: answer(images, prompt, meta) -> str, where images are the frame's camera
images (paths; a clip's frames oldest first), prompt the question with core.vlm.PROMPT_SUFFIX, and meta the frame's
drive, step and question type. Runners (make_runner(spec)):

    oracle                    core.language's own template answer — perfect by construction; NOT a model. It reads
                              the ground truth and exists to validate the scorer end to end.
    noisy[:p=0.3,h=0.1,seed=0]  the template answer, replaced with probability p by an answer that is wrong by
                              construction (core.vlm.corrupt) and, with probability h, with its object phrase
                              replaced by one naming an absent object (a planted hallucination). Seeded per
                              question. NOT a model: its expected scores are known exactly (1 − share corrupted),
                              which tests the pipeline.
    plugin:module:Class[,k=v] a Python class with answer(images, prompt, meta) -> str (like the models' plug-ins);
                              its meta never carries the truth.
    hf:NAME | hf:DIR[,adapter=DIR,device=…,max_new_tokens=…]   a local Hugging Face model run with torch through
                              transformers (AutoProcessor + AutoModelForImageTextToText, greedy decoding), loaded with
                              local_files_only from W/models/vlm/NAME (a candidate below, checked against its pins
                              first) or a folder; an adapter is a LoRA from core.vlm_lora. Needs the weights on
                              disk: nothing is ever downloaded here.

Candidates (CANDIDATES): small open VLMs whose licences were read from their model cards (the card's YAML licence
field and its text) and whose files, sizes and checksums were read from the Hugging Face API's metadata
(https://huggingface.co/api/models/REPO?blobs=true, 2026-10-29) — WITHOUT downloading any weights. Each is pinned to a
repository revision (commit). The publisher's checksums are what the metadata states: an LFS file's SHA-256 (the
weights) and a small file's git blob id (SHA-1 of "blob <size>\\0" + content). Falcon has not downloaded these
files, so its own SHA-256 pins are "to be pinned on first approved download": fetch() downloads every file from
https://huggingface.co/REPO/resolve/REVISION/FILE only when asked (the licence, sizes and checksums shown first and a
confirmation given), checks each file's size and the publisher's checksum (refusing and deleting on a mismatch),
computes its SHA-256 and writes them to pins.json beside the files. Every later load checks the files against
pins.json and the publisher's checksums (verify()). Only models that transformers runs natively are candidates:
REFUSED lists models refused by default and why — a non-commercial or research-only licence, terms that are not an
open licence, a licence that cannot be checked, or code that would have to be run from the repository
(trust_remote_code). An AGPL licence is refused like core.models refuses it.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os

HF = 'https://huggingface.co'
METADATA_READ = '2026-10-29'
PINS = 'pins.json'
PERMISSIVE = ('Apache-2.0', 'MIT', 'BSD-3-Clause', 'BSD-2-Clause')

_FILES = {
    'smolvlm-256m': ('7e3e67edbbed1bf9888184d9df282b700a323964', 256484928, '2025-04-08', [
        ('added_tokens.json', 4739, None, '71dc811700b6aa9274d3697ee7d4fb2d5375e766'),
        ('chat_template.json', 429, None, '214a30563adf9598b26cdd3ad3bbafca9d7af45d'),
        ('config.json', 7353, None, '1c09d62975f946e7e07f62ec4db3f16c9c701144'),
        ('generation_config.json', 136, None, '14051e31eab5938a09976bf8c3e4769491f53a80'),
        ('merges.txt', 466391, None, '69503b13f727ba3812b6803e97442a6de05ef5eb'),
        ('model.safetensors', 513028808,
         '74dea5904032e5ae99a2e0eef5179e6ac0f1dedc3ab0c7c2a5d4d387c843203e',
         '78287ef1548dfe243932903a67750122b2e7cf2c'),
        ('preprocessor_config.json', 486, None, '893236748032d4bf033823d6a4a0d62b78b6bc0e'),
        ('processor_config.json', 68, None, 'b3b1deadd0e6e18c2070fdb0c29074892ea58ffe'),
        ('special_tokens_map.json', 1069, None, '647565708c361dc42350b5c7b79c46a4ff153b18'),
        ('tokenizer.json', 3548256, None, 'a4005d1cf3170a31600a5c96f95768166cbc2b28'),
        ('tokenizer_config.json', 28249, None, 'c85c07d72e6ffca6b6ff138a8e8f09ae65740bfb'),
        ('vocab.json', 800662, None, '0ad5ecc2035b7031b88afb544ee95e2d49baa484'),
    ]),
    'smolvlm-500m': ('a7da5b986cb59b408707209984f360a5f4ad7e47', 507482304, '2025-04-08', [
        ('added_tokens.json', 4739, None, '71dc811700b6aa9274d3697ee7d4fb2d5375e766'),
        ('chat_template.json', 429, None, '214a30563adf9598b26cdd3ad3bbafca9d7af45d'),
        ('config.json', 7339, None, 'd2a7bdf9e204095994fa15a3160a28e4c1fbd505'),
        ('generation_config.json', 136, None, '14051e31eab5938a09976bf8c3e4769491f53a80'),
        ('merges.txt', 466391, None, '69503b13f727ba3812b6803e97442a6de05ef5eb'),
        ('model.safetensors', 1015025832,
         'd05b567eeaf534e83d375551f068ed57b5f52d37c657197f644af5ef9db091a2',
         'ea2388e7a1e70b506f1d55d44200fd94382402bc'),
        ('preprocessor_config.json', 486, None, '893236748032d4bf033823d6a4a0d62b78b6bc0e'),
        ('processor_config.json', 68, None, 'b3b1deadd0e6e18c2070fdb0c29074892ea58ffe'),
        ('special_tokens_map.json', 1069, None, '647565708c361dc42350b5c7b79c46a4ff153b18'),
        ('tokenizer.json', 3548256, None, 'a4005d1cf3170a31600a5c96f95768166cbc2b28'),
        ('tokenizer_config.json', 28249, None, 'c85c07d72e6ffca6b6ff138a8e8f09ae65740bfb'),
        ('vocab.json', 800662, None, '0ad5ecc2035b7031b88afb544ee95e2d49baa484'),
    ]),
    'smolvlm-2.2b': ('81cd9a775a4d644f2faf4e7becff4559b46b14c7', 2246272880, '2025-04-08', [
        ('added_tokens.json', 92, None, '3931e0759cc880bf72050e12d25a3e67179cec93'),
        ('chat_template.json', 429, None, '214a30563adf9598b26cdd3ad3bbafca9d7af45d'),
        ('config.json', 7446, None, '895dae86deefabacae15e58e72ed29c9410d8167'),
        ('generation_config.json', 136, None, 'c8c90c3d21849e1ec9431cf5ba37af2eadd73480'),
        ('merges.txt', 466391, None, '69503b13f727ba3812b6803e97442a6de05ef5eb'),
        ('model.safetensors', 4492630912,
         '8a4f76cb64f6f2e4e74716d8fc1cfc6a70bbb3eeea69d424c3ec9902655065eb',
         '1616027444f647e228108996af62079300c82262'),
        ('preprocessor_config.json', 486, None, '386f4280443ee8df012d6dd2362b6932d0202370'),
        ('processor_config.json', 68, None, 'f166fc7e3b5824ed0c6bcdfb04b0e8d159dc7113'),
        ('special_tokens_map.json', 1069, None, '647565708c361dc42350b5c7b79c46a4ff153b18'),
        ('tokenizer.json', 3523234, None, '761f4389b21a4218d0a7a3a0ff7a26a88faaeffd'),
        ('tokenizer_config.json', 4478, None, '3bdd66767b006f70e97ed5b3f692e5028b7b0cdf'),
        ('vocab.json', 800662, None, '0ad5ecc2035b7031b88afb544ee95e2d49baa484'),
    ]),
    'smolvlm2-2.2b': ('482adb537c021c86670beed01cd58990d01e72e4', 2246784880, '2025-04-08', [
        ('added_tokens.json', 4739, None, '71dc811700b6aa9274d3697ee7d4fb2d5375e766'),
        ('chat_template.json', 430, None, '1d342e4f070c962a14ba4ece4e06212bc70d8e36'),
        ('config.json', 3642, None, '6a01472ec2a0595b85ac601a4e99efdcaa1ff108'),
        ('generation_config.json', 136, None, 'eace9aa6d392fd94dbe0c90825074c53fd7ecd4a'),
        ('merges.txt', 466391, None, '69503b13f727ba3812b6803e97442a6de05ef5eb'),
        ('model-00001-of-00002.safetensors', 4959390584,
         '7c88a90e78f567460b2369a0369cc4dafbfda5607c9cb090b30fa4437b0282f2',
         '27a1b8c2f57a6a3ff466e6ba56797390a663add6'),
        ('model-00002-of-00002.safetensors', 4027833472,
         'cf3d0958a8240f71054a6812038f2a6bfbe2e5124573a3b363f6b120c9a52657',
         'f6bb3458057bd255bdd62f02d9e17a1c5446970f'),
        ('model.safetensors.index.json', 63881, None, '9401d792de4308a3844189f61673670dc8c09349'),
        ('preprocessor_config.json', 599, None, '9b5b2a5a98428bac8ba0257a59bc74da5d826e5c'),
        ('processor_config.json', 67, None, '093f7dbbd52805fca7e747f569d2c7906227499a'),
        ('special_tokens_map.json', 868, None, '2b5aee28d56c0e9d89ff6bc70818fba986a90ca9'),
        ('tokenizer.json', 3548256, None, 'a4005d1cf3170a31600a5c96f95768166cbc2b28'),
        ('tokenizer_config.json', 28627, None, 'd986ab355a4a4de5aa4485fdfba4625eb737928c'),
        ('vocab.json', 800662, None, '0ad5ecc2035b7031b88afb544ee95e2d49baa484'),
    ]),
    'qwen2-vl-2b': ('895c3a49bc3fa70a340399125c650a463535e71c', 2208985600, '2025-01-12', [
        ('LICENSE', 11343, None, '6634c8cc3133b3848ec74b9f275acaaa1ea618ab'),
        ('chat_template.json', 1050, None, '732bd68bc5427d1fb6c06a59b3bf2456b2155d24'),
        ('config.json', 1196, None, '8e922d54443e7d2ad46c01baea1b772ecbc6228e'),
        ('generation_config.json', 272, None, '896384ee102f9a2696e2f6eb02ba5a143f198fac'),
        ('merges.txt', 1671839, None, '20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0'),
        ('model-00001-of-00002.safetensors', 3988609112,
         '994ac2b03f97de8bc647d0fe5eba2e4b632b3e28dc03574c29bdfc36cf47e1b9',
         '85cc1b67c1a034c0be0f4500069cce629b1a7ca7'),
        ('model-00002-of-00002.safetensors', 429441656,
         '92540d8353c8d226a589a3b179bdb33851c970ee2cc2ac7ba035f79425e7b833',
         '11541ce8d7bb033df9dfc9eaa891dd1c80d0fd52'),
        ('model.safetensors.index.json', 56411, None, '293d54cd8408e82d1bdd2eabd07d9311eb406b52'),
        ('preprocessor_config.json', 347, None, 'c8bf52bdaee98b049e28da3ef94119f896b03a8d'),
        ('tokenizer.json', 7029741, None, 'f5195cf97e3c90290b670d48a7c46fef5f145ebc'),
        ('tokenizer_config.json', 4190, None, 'ecbc13d75ff851c4f1e57990064f81303b2494f6'),
        ('vocab.json', 2776833, None, '4783fe10ac3adce15ac8f358ef5462739852c569'),
    ]),
    'llava-onevision-0.5b': ('74dd0bf867a4cda7950c17663794267c60cf4b40', 893675552, '2025-06-18', [
        ('added_tokens.json', 122, None, 'f4d12380a35b1f8032caad7bc7a006f5d3fa759e'),
        ('chat_template.json', 826, None, '68724c68b1612e9ab2e31f84e3279099ad656ff6'),
        ('config.json', 2591, None, 'd1fcc2ccafee7f4c82166f07f0f1918c8334553b'),
        ('generation_config.json', 126, None, '9f451858a72667b8ed295a9c44c14cefb99deb8f'),
        ('merges.txt', 1671853, None, '31349551d90c7606f325fe0f11bbb8bd5fa0d7c7'),
        ('model.safetensors', 1787445680,
         '07b3362c3412de79baf2379e44e5b0b2a8f4b965ebebd11d7b5b3eb4450fe96e',
         '4376b1854899171c0875d59bbd734ad41e82d502'),
        ('preprocessor_config.json', 1732, None, '1ff68626f165d5d9e234e1d6d293e646f7d269bf'),
        ('processor_config.json', 178, None, '7779c690fef0fe2026261c1f68eac18c57448319'),
        ('special_tokens_map.json', 367, None, '2c53454aff27c6d22873cc28c884c2b2d5d07693'),
        ('tokenizer.json', 7028579, None, '532b756b16b09479d11b1b5696d96d773555a053'),
        ('tokenizer_config.json', 1800, None, '36ffab84cf909f8f7940f5a85a81e516ce0f4026'),
        ('video_preprocessor_config.json', 621, None, '54bfac33b4d586990eabc014123202e3322e1f22'),
        ('vocab.json', 2776833, None, '4783fe10ac3adce15ac8f358ef5462739852c569'),
    ]),
    'granite-vision-3.2-2b': ('ec894191f2e9e9be3d90eab9ac6949583c46f03b', 2975396928, '2026-04-02', [
        ('added_tokens.json', 116, None, '19135458075113dc86ca4c031cefb83a3d0e9f5e'),
        ('chat_template.json', 1643, None, '9418a479b1d9e2f59fa9900c125c53e5572b1cb4'),
        ('config.json', 3162, None, '3aac6010e370edecc8faa6d40fc617d9ac9ee2b6'),
        ('generation_config.json', 137, None, '2ba69c37bbd345246f10e28f971197cde10389cd'),
        ('merges.txt', 441810, None, 'f8479fb696fe07332c55300a6accf8cc191acc6a'),
        ('model-00001-of-00002.safetensors', 4999997080,
         'c4d8e90faaebb1ebb6c8ab35766d6147a48156dbee0b7018c1010ed9a35cd069',
         '3ea8fbe1fc826b245af6ee81cc3b3c6796b19a74'),
        ('model-00002-of-00002.safetensors', 950905416,
         'c9f0d38fa3e72c501c2c4a03e7afa1459f3a3ecbe7dece5925778c0d34062bb4',
         'e21a6ee28f965d9802f97d1a25c1b9edeb253b6e'),
        ('model.safetensors.index.json', 86365, None, '0019438ee4ade26327e04993819922bc85493d31'),
        ('preprocessor_config.json', 2047, None, 'c9f5303e42825af60fd41bba7b8e2a636658efa9'),
        ('processor_config.json', 115, None, '8dcec4ee9c782f8463150b48168e9f50bf64abea'),
        ('special_tokens_map.json', 701, None, '386500a5040da66c6db3d8b9c44ccd1ee202c744'),
        ('tokenizer.json', 2057967, None, 'e8777924cfb6b21454620fae0f5adc4457cb38bc'),
        ('tokenizer_config.json', 6975, None, '75ecaf5259c8ce5ba7a2eb875687e0ab464002b2'),
        ('vocab.json', 776995, None, '0a11f2016e660fd490f7bf168e6d1f9c86a8f744'),
    ]),
}

_ABOUT = {
    'smolvlm-256m': ('HuggingFaceTB/SmolVLM-256M-Instruct', 'Apache-2.0', 'Idefics3ForConditionalGeneration',
                     'SmolVLM 256M (Marafioti et al. 2025, arXiv:2504.05299); below 1B — the cheapest first try'),
    'smolvlm-500m': ('HuggingFaceTB/SmolVLM-500M-Instruct', 'Apache-2.0', 'Idefics3ForConditionalGeneration',
                     'SmolVLM 500M (arXiv:2504.05299)'),
    'smolvlm-2.2b': ('HuggingFaceTB/SmolVLM-Instruct', 'Apache-2.0', 'Idefics3ForConditionalGeneration',
                     'SmolVLM 2.2B (arXiv:2504.05299)'),
    'smolvlm2-2.2b': ('HuggingFaceTB/SmolVLM2-2.2B-Instruct', 'Apache-2.0', 'SmolVLMForConditionalGeneration',
                      'SmolVLM2 2.2B, images and video (arXiv:2504.05299); weights stored in float32 (9 GB)'),
    'qwen2-vl-2b': ('Qwen/Qwen2-VL-2B-Instruct', 'Apache-2.0', 'Qwen2VLForConditionalGeneration',
                    'Qwen2-VL 2B (Wang et al. 2024, arXiv:2409.12191)'),
    'llava-onevision-0.5b': ('llava-hf/llava-onevision-qwen2-0.5b-ov-hf', 'Apache-2.0',
                             'LlavaOnevisionForConditionalGeneration',
                             'LLaVA-OneVision 0.5B (Li et al. 2024, arXiv:2408.03326), transformers conversion'),
    'granite-vision-3.2-2b': ('ibm-granite/granite-vision-3.2-2b', 'Apache-2.0', 'LlavaNextForConditionalGeneration',
                              'Granite Vision 3.2 2B (IBM; arXiv:2502.09927), about 3B parameters with its encoder'),
}
CANDIDATES = {}
for _n, (_repo, _lic, _arch, _note) in _ABOUT.items():
    _rev, _params, _mod, _files = _FILES[_n]
    CANDIDATES[_n] = {'repo': _repo, 'revision': _rev, 'licence': _lic, 'arch': _arch, 'params': _params,
                      'note': _note, 'last_modified': _mod, 'bytes': sum(f[1] for f in _files),
                      'files': [{'name': f[0], 'size': f[1], 'lfs_sha256': f[2], 'git_oid': f[3]} for f in _files],
                      'sha256_pin': 'to be pinned on first approved download',
                      'metadata': f'{HF}/api/models/{_repo}?blobs=true (read {METADATA_READ}; no weights downloaded)'}

# Read from the model cards and repositories on 2026-10-29; refused by default.
REFUSED = {
    'qwen2.5-vl-3b': ('Qwen/Qwen2.5-VL-3B-Instruct', 'qwen-research (Qwen RESEARCH LICENSE)', 3754622976,
                      'non-commercial: its LICENSE defines "Non-Commercial" as research or evaluation only and asks '
                      'commercial users for a licence'),
    'fastvlm-1.5b': ('apple/FastVLM-1.5B', 'apple-amlr', 1909278176,
                     'research only (its licence limits use and derivatives to non-commercial research); also needs '
                     'trust_remote_code'),
    'paligemma2-3b': ('google/paligemma2-3b-pt-224', 'gemma (Gemma Terms of Use)', 3032242416,
                      'not an open licence: use restrictions, and gated behind accepting Google\'s terms'),
    'deepseek-vl-1.3b': ('deepseek-ai/deepseek-vl-1.3b-chat', 'other (deepseek)', 1975235584,
                         'the licence cannot be checked: the card links a LICENSE file the repository does not hold, '
                         'and the DeepSeek model licence carries use restrictions'),
    'minicpm-v-2': ('openbmb/MiniCPM-V-2', 'not stated in the card\'s metadata (MiniCPM model licence)', 3434965792,
                    'commercial use only after registration with the publisher; also needs trust_remote_code'),
    'phi-3.5-vision': ('microsoft/Phi-3.5-vision-instruct', 'MIT', 4146621440,
                       'needs trust_remote_code (Phi3V is not in transformers): running a repository\'s Python is '
                       'refused by default'),
    'moondream2': ('vikhyatk/moondream2', 'Apache-2.0', 1927237104,
                   'needs trust_remote_code (its model code lives in the repository)'),
    'internvl2.5-2b': ('OpenGVLab/InternVL2_5-2B', 'MIT', 2205754368,
                       'needs trust_remote_code (InternVLChatModel lives in the repository)'),
}


class VLMError(Exception):
    pass


# ================================================================================================ files and pins
def candidate_dir(workspace: str, name: str) -> str:
    return os.path.join(workspace, 'models', 'vlm', name)


def git_oid(path: str) -> str:
    """The git blob id of a file: SHA-1 of "blob <size>\\0" + its bytes (what the Hub states for a non-LFS file)."""
    h = hashlib.sha1()
    h.update(f'blob {os.path.getsize(path)}\0'.encode())
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def check_licence(c: dict) -> list:
    lic = str(c.get('licence') or '')
    out = []
    if 'AGPL' in lic.upper():
        out.append('an AGPL-licensed model is refused (its licence would bind the app)')
    elif lic not in PERMISSIVE:
        out.append(f'licence {lic or "not stated"} is not one of the permissive licences Falcon accepts by default '
                   f'({", ".join(PERMISSIVE)})')
    return out


def verify(name: str, folder: str, c: dict | None = None, full: bool = True) -> dict:
    """Check a downloaded candidate: every file present with its size; its SHA-256 equal to pins.json's; and (full)
    the publisher's checksum (LFS SHA-256 or git blob id). {'ok', 'problems', 'pins'}."""
    from .provenance import sha256_file
    c = c or CANDIDATES[name]
    problems = []
    pins_p = os.path.join(folder, PINS)
    pins = None
    if os.path.isfile(pins_p):
        with open(pins_p, encoding='utf-8') as fh:
            pins = json.load(fh)
        if pins.get('revision') != c['revision']:
            problems.append(f'pins.json is for revision {pins.get("revision")}, not {c["revision"]}')
    else:
        problems.append('no pins.json: not downloaded by falcon vlm fetch')
    for f in c['files']:
        p = os.path.join(folder, f['name'])
        if not os.path.isfile(p):
            problems.append(f'{f["name"]}: missing')
            continue
        if os.path.getsize(p) != f['size']:
            problems.append(f'{f["name"]}: {os.path.getsize(p)} bytes, not {f["size"]}')
            continue
        h = sha256_file(p)
        pin = ((pins or {}).get('files') or {}).get(f['name'], {}).get('sha256')
        if pin and h != pin:
            problems.append(f'{f["name"]}: SHA-256 {h} is not the pinned {pin}')
        if full:
            if f['lfs_sha256'] and h != f['lfs_sha256']:
                problems.append(f'{f["name"]}: SHA-256 {h} is not the publisher\'s {f["lfs_sha256"]}')
            elif not f['lfs_sha256'] and git_oid(p) != f['git_oid']:
                problems.append(f'{f["name"]}: git blob id is not the publisher\'s {f["git_oid"]}')
    return {'ok': not problems, 'problems': problems, 'pins': pins}


def fetch(name: str, workspace: str, progress=None, base_url: str = HF, candidates: dict | None = None) -> str:
    """Download a candidate's files (only when asked: the caller has shown the licence and had it confirmed), check
    each one's size and publisher checksum (refused and deleted on a mismatch), and write pins.json with the SHA-256
    computed here — the first approved download pins it. Returns the folder."""
    import urllib.request
    from .provenance import utc_now, write_json
    cands = candidates or CANDIDATES
    if name not in cands:
        raise VLMError(f'unknown candidate {name!r}' + (f': refused — {REFUSED[name][3]}' if name in REFUSED else
                                                         f'; one of {", ".join(cands)}'))
    c = cands[name]
    bad = check_licence(c)
    if bad:
        raise VLMError('; '.join(bad))
    folder = candidate_dir(workspace, name)
    os.makedirs(folder, exist_ok=True)
    total = sum(f['size'] for f in c['files'])
    done = 0
    pins = {'name': name, 'repo': c['repo'], 'revision': c['revision'], 'licence': c['licence'], 'files': {},
            'source': f'{base_url}/{c["repo"]}/resolve/{c["revision"]}/', 'downloaded': None}
    for f in c['files']:
        p = os.path.join(folder, f['name'])
        url = f'{base_url}/{c["repo"]}/resolve/{c["revision"]}/{f["name"]}'
        tmp = p + '.part'
        h256, h1 = hashlib.sha256(), hashlib.sha1()
        h1.update(f'blob {f["size"]}\0'.encode())
        got = 0
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, 'wb') as fh:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                fh.write(b)
                h256.update(b)
                h1.update(b)
                got += len(b)
                if progress and total:
                    progress((done + got) / total)
        why = None
        if got != f['size']:
            why = f'{got} bytes, not {f["size"]}'
        elif f['lfs_sha256'] and h256.hexdigest() != f['lfs_sha256']:
            why = f'SHA-256 {h256.hexdigest()} is not the publisher\'s {f["lfs_sha256"]}'
        elif not f['lfs_sha256'] and h1.hexdigest() != f['git_oid']:
            why = f'git blob id {h1.hexdigest()} is not the publisher\'s {f["git_oid"]}'
        if why:
            os.remove(tmp)
            raise VLMError(f'{f["name"]}: {why} — not installed')
        os.replace(tmp, p)
        done += f['size']
        pins['files'][f['name']] = {'size': f['size'], 'sha256': h256.hexdigest(),
                                    'checked': 'publisher LFS SHA-256' if f['lfs_sha256'] else 'publisher git blob id'}
    pins['downloaded'] = utc_now()
    write_json(os.path.join(folder, PINS), pins)
    return folder


def list_candidates(workspace: str = '') -> list:
    """Rows for a table: every candidate with its state in the workspace, then the refused ones."""
    rows = []
    for n, c in CANDIDATES.items():
        folder = candidate_dir(workspace, n) if workspace else ''
        state = 'not downloaded'
        if folder and os.path.isfile(os.path.join(folder, PINS)):
            state = 'downloaded (pinned)'
        rows.append({'name': n, 'repo': c['repo'], 'licence': c['licence'], 'params': c['params'],
                     'bytes': c['bytes'], 'state': state, 'refused': None, 'note': c['note'],
                     'revision': c['revision']})
    for n, (repo, lic, params, why) in REFUSED.items():
        rows.append({'name': n, 'repo': repo, 'licence': lic, 'params': params, 'bytes': None, 'state': 'refused',
                     'refused': why, 'note': '', 'revision': None})
    return rows


# ================================================================================================ runners
class Runner:
    """answer(images, prompt, meta) -> str. name, kind, licence, reads_truth (True only for the test baselines),
    providers (where it ran), files (model files hashed for provenance)."""
    name = 'runner'
    kind = ''
    licence = None
    reads_truth = False
    providers = ['python']
    needs_gpu = False

    def info(self) -> dict:
        return {'name': self.name, 'kind': self.kind, 'licence': self.licence, 'reads_truth': self.reads_truth,
                'providers': list(self.providers), 'not_a_model': self.reads_truth}

    def note(self, qid: str):
        return None

    def close(self):
        pass


class OracleRunner(Runner):
    """core.language's template answer — perfect by construction (reads the truth: not a model)."""
    name, kind, licence, reads_truth = 'oracle', 'oracle', 'GPL-3.0-or-later (Falcon)', True

    def answer(self, images, prompt, meta):
        return meta['question']['template']


class NoisyRunner(Runner):
    """The template answer with seeded corruption (module doc). note(qid) says what was done to each answer."""
    kind, licence, reads_truth = 'noisy', 'GPL-3.0-or-later (Falcon)', True

    def __init__(self, p: float = 0.3, h: float = 0.1, seed: int = 0):
        if not (0 <= p <= 1 and 0 <= h <= 1):
            raise VLMError('noisy: p and h are probabilities')
        self.p, self.h, self.seed = float(p), float(h), int(seed)
        self.name = f'noisy(p={self.p:g},h={self.h:g},seed={self.seed})'
        self.log = {}

    def answer(self, images, prompt, meta):
        from . import vlm as V
        q, ctx = meta['question'], meta['context']
        rng = V._rng('noisy', self.seed, q['qid'])
        text, done = q['template'], []
        if rng.random() < self.p:
            text = V.corrupt(q, ctx, rng)
            done.append('corrupted')
        if rng.random() < self.h:
            text = V.hallucinate(text, ctx)
            done.append('hallucinated')
        self.log[q['qid']] = done
        return text

    def note(self, qid):
        return self.log.get(qid)


class PluginRunner(Runner):
    """A Python class with answer(images, prompt, meta) -> str; its meta has no truth."""
    kind = 'plugin'

    def __init__(self, entry: str, params: dict | None = None):
        mod, _, cls = entry.partition(':')
        if not mod or not cls:
            raise VLMError('a plug-in is module:Class')
        self.plugin = getattr(importlib.import_module(mod), cls)(**(params or {}))
        self.name = entry
        self.licence = getattr(self.plugin, 'licence', None)
        bad = check_licence({'licence': self.licence}) if self.licence and 'AGPL' in str(self.licence).upper() else []
        if bad:
            raise VLMError(bad[0])
        self.needs_gpu = bool(getattr(self.plugin, 'needs_gpu', False))

    def answer(self, images, prompt, meta):
        clean = {k: v for k, v in meta.items() if k not in ('question', 'context')}
        clean['type'] = meta['question']['type']
        return str(self.plugin.answer(images, prompt, clean))


def load_images(images, max_side: int = 0):
    from PIL import Image
    out = []
    for im in images:
        img = Image.open(im).convert('RGB') if isinstance(im, str) else Image.fromarray(im)
        if max_side and max(img.size) > max_side:
            k = max_side / max(img.size)
            img = img.resize((max(1, round(img.size[0] * k)), max(1, round(img.size[1] * k))), Image.BILINEAR)
        out.append(img)
    return out


class HFRunner(Runner):
    """A local Hugging Face VLM (module doc). Loaded with local_files_only=True and HF_HUB_OFFLINE set: it never
    downloads. A candidate folder is checked against pins.json (and the publisher's checksums) first."""
    kind = 'hf'
    needs_gpu = True

    def __init__(self, source: str, workspace: str = '', adapter: str = '', device: str = 'auto',
                 max_new_tokens: int = 64, max_side: int = 768, dtype: str = 'auto', seed: int = 0,
                 verify_files: bool = True, gpu_memory_fraction: float = 0.6):
        os.environ['HF_HUB_OFFLINE'] = '1'                 # no network from transformers or huggingface_hub
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        if source in CANDIDATES:
            folder = candidate_dir(workspace, source)
            c = CANDIDATES[source]
            if not os.path.isdir(folder):
                raise VLMError(f'{source} is not downloaded: falcon vlm fetch {source} --workspace W (licence '
                               f'{c["licence"]}, {c["bytes"] / 2 ** 30:.1f} GiB) — asked for, never automatic')
            v = verify(source, folder, c) if verify_files else {'ok': True}
            if not v['ok']:
                raise VLMError(f'{source}: ' + '; '.join(v['problems']))
            self.licence = c['licence']
            self.name = source
        elif os.path.isdir(source):
            folder = source
            meta_p = os.path.join(folder, 'falcon_vlm.json')        # a folder's own licence note, when it has one
            info = json.load(open(meta_p, encoding='utf-8')) if os.path.isfile(meta_p) else {}
            self.licence = info.get('licence')
            if self.licence and check_licence({'licence': self.licence}) and 'AGPL' in self.licence.upper():
                raise VLMError(check_licence({'licence': self.licence})[0])
            self.name = os.path.basename(os.path.normpath(folder))
        else:
            raise VLMError(f'hf:{source}: neither a candidate ({", ".join(CANDIDATES)}) nor a folder')
        self.folder = folder
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        if self.device.type == 'cuda':        # the training lab's heuristic: leave room for the desktop (training.py)
            torch.cuda.set_per_process_memory_fraction(float(gpu_memory_fraction), self.device.index or 0)
        if dtype == 'auto':
            dt = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
        else:
            dt = getattr(torch, dtype)
        self.processor = AutoProcessor.from_pretrained(folder, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(folder, local_files_only=True, dtype=dt).to(
            self.device).eval()
        self.adapter = ''
        if adapter:
            from .vlm_lora import load_adapter
            load_adapter(self.model, adapter)
            self.adapter = os.path.abspath(adapter)
            self.name += f'+lora({os.path.basename(os.path.normpath(adapter))})'
        self.max_new_tokens, self.max_side, self.seed = int(max_new_tokens), int(max_side), int(seed)
        self.providers = [f'torch:{self.device}', f'dtype:{str(dt).replace("torch.", "")}']
        self.needs_gpu = self.device.type == 'cuda'

    def messages(self, n_images: int, prompt: str) -> list:
        return [{'role': 'user', 'content': [{'type': 'image'} for _ in range(n_images)] +
                 [{'type': 'text', 'text': prompt}]}]

    def inputs(self, images, prompt, answer: str | None = None):
        """The processor's tensors for images + prompt (and the answer appended, for training: core.vlm_lora)."""
        msgs = self.messages(len(images), prompt)
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
        if answer is not None:
            text = text + answer
        return self.processor(text=[text], images=[load_images(images, self.max_side)] if images else None,
                              return_tensors='pt'), text

    def answer(self, images, prompt, meta):
        import torch
        torch.manual_seed(self.seed)
        enc, _ = self.inputs(images, prompt)
        enc = {k: (v.to(self.device) if hasattr(v, 'to') else v) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model.generate(**enc, do_sample=False, max_new_tokens=self.max_new_tokens)
        new = out[0, enc['input_ids'].shape[1]:]
        return self.processor.decode(new, skip_special_tokens=True).strip()

    def info(self):
        d = super().info()
        d.update(folder=self.folder, adapter=self.adapter or None, max_new_tokens=self.max_new_tokens)
        return d

    def close(self):
        import gc
        self.model = None
        gc.collect()


def parse_spec(spec: str) -> tuple:
    """'kind:rest,k=v,…' -> (kind, rest, {k: v}) (values JSON when they parse)."""
    kind, _, rest = spec.partition(':')
    params = {}
    if kind in ('noisy',):
        body, rest = rest, ''
    else:
        body = ''
        if ',' in rest:
            rest, _, body = rest.partition(',')
    for kv in [x for x in body.split(',') if x]:
        k, eq, v = kv.partition('=')
        if not eq:
            raise VLMError(f'runner option {kv!r} is not key=value')
        try:
            params[k.strip()] = json.loads(v)
        except ValueError:
            params[k.strip()] = v
    return kind, rest, params


def make_runner(spec: str, workspace: str = '') -> Runner:
    kind, rest, params = parse_spec(spec)
    if kind == 'oracle':
        return OracleRunner()
    if kind == 'noisy':
        return NoisyRunner(**params)
    if kind == 'plugin':
        return PluginRunner(rest, params)
    if kind == 'hf':
        return HFRunner(rest, workspace, **params)
    raise VLMError(f'unknown runner {spec!r}: oracle, noisy[:p=…,h=…,seed=…], plugin:module:Class or hf:NAME|DIR')


def describe_candidate(name: str) -> str:
    """The text shown before a download is confirmed."""
    c = CANDIDATES[name]
    lines = [f'{name}: {c["repo"]} at revision {c["revision"]}', f'licence: {c["licence"]} (read from the model card)',
             f'parameters: {c["params"] / 1e9:.2f} B; download: {c["bytes"] / 2 ** 30:.2f} GiB in {len(c["files"])} '
             f'files from {HF}/{c["repo"]}/resolve/{c["revision"]}/', f'{c["note"]}',
             'publisher checksums (checked on download; Falcon\'s SHA-256 pins are written on this first download):']
    for f in c['files']:
        lines.append(f'  {f["name"]:36s} {f["size"]:>12d} B  ' + (f'LFS sha256 {f["lfs_sha256"]}' if f['lfs_sha256']
                                                                   else f'git blob {f["git_oid"]}'))
    return '\n'.join(lines)
