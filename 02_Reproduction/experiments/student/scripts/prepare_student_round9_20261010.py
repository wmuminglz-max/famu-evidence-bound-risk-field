# Deidentified copy. String literals containing machine locations were replaced.
"""Freeze one bounded small-language adapter training experiment."""
import json, hashlib, shutil, sys, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.student_round9_data import generate
OUT = ROOT / 'outputs/famu_student_round9_20261010'
PREV = ROOT / 'outputs/famu_student_round8_20261010'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf-8')

def main():
    if (OUT / 'package').exists():
        raise RuntimeError('Existing R9 preserved')
    package = OUT / 'package'
    package.mkdir(parents=True)
    prior = json.loads((PREV / 'package/input_manifest.json').read_text('utf-8'))
    for rel, e in prior.items():
        if rel.startswith(('src/', 'scripts/', 'encoder/')):
            p = PREV / 'package' / rel
            assert sha(p) == e['sha256']
            dest = package / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
    for rel in ['src/evidence_reasoning/adapted_language_student.py', 'scripts/student_round9_data.py', 'scripts/prepare_student_round9_20261010.py', 'scripts/run_student_round9_20261010.py']:
        shutil.copyfile(ROOT / rel, package / rel)
    for n in ['prepare_student_round2_20261010.py', 'prepare_student_round3_20261010.py', 'prepare_student_round8_20261010.py']:
        shutil.copyfile(ROOT / 'scripts' / n, package / 'scripts' / n)
    (package / 'data').mkdir()
    for n in ['fit', 'historical', 'r2_regression', 'r3_regression', 'r4_regression', 'r5_regression', 'r6_regression', 'r7_regression', 'contrast_pairs', 'language_pairs']:
        shutil.copyfile(PREV / f'package/data/{n}.json', package / f'data/{n}.json')
    prevrows = json.loads((PREV / 'package/data/new_scenes.json').read_text('utf-8'))
    for suffix, split in [('', 'test'), ('_short', 'test_short')]:
        write(package / f'data/r8_regression{suffix}.json', [r for r in prevrows if r['split'] == split])
    rows, triples = generate()
    write(package / 'data/new_scenes.json', rows)
    write(package / 'data/long_pairs.json', triples)
    held = {r['family'] for r in rows if r['split'] != 'train'}
    assert not held & {p['group'] for p in triples}
    for seed in [17, 29, 43]:
        src = PREV / f'result_bundle/outputs/LEX_SFT_s{seed}/model'
        dest = package / f'head_source_s{seed}'
        dest.mkdir()
        m = json.loads((src / 'manifest.json').read_text('utf-8'))
        for n, h in m.items():
            assert sha(src / n) == h
            shutil.copyfile(src / n, dest / n)
        shutil.copyfile(src / 'manifest.json', dest / 'manifest.json')
    protocol = {'id': 'famu_student_round9_20261010', 'seeds': [17, 29, 43], 'arms': ['FROZEN', 'LANG_SFT', 'LANG_CONTRAST'], 'updates_per_run': 960, 'max_total_updates': 8640, 'batch_size': 32, 'max_relation_anchor_exposures': 276480, 'max_CE_positions': 691200, 'wall_hours_max': 2, 'checkpoint_steps': [480, 960], 'model': 'Original4layer512 Chinese BERT with rank8 alpha16 LoRA on all24 encoder layer Linear modules; base frozen, adapters optimized through actual Transformer forwards/backwards. Same R8 LEX head4618->64->12 and query4096->4 initialized from each corresponding seed. No language generation or large teacher runtime. FROZEN identical zero-output adapters held fixed; heads train in all arms.', 'optimizer': 'FP32 eager; no dropout all arms, TF32 off; AdamW head1e-3, adapter2e-4, wd.01, clip1,96warmup linear960; identical per-seed batches/initial adapters/head. No model-based RL this round.', 'sampling': 'Per32 anchors:8original,4recombined,2R7contrast,2R8language,16R9complex; classes16irrelevant/8support/8refute. Family uniform within pool/class. CE on32anchors+32same-fact compact views+up to16opposite-fact views; actual counts/token/FLOP differences recorded, no hidden repeated exposure.', 'loss': 'Allarms CE over anchor/compact/validopposite positions + auxiliaryqueryCE; LANG_CONTRAST adds .2*(1-cos(anchor,compact)+relu(.2+cos(anchor,opposite)-cos(anchor,compact))) on R9 samefield T/F pairs only. Opposite changes one fact, not arbitrary same-label contrast. No test examples or teacher labels in fitting.', 'dataset': '1152 new fictional training scenes:24families4wordingstyles12variations;72newdev;96newlongtest/96samefactshort/96samefactreordered. Program-derived shared structures, NOT independent expert/legal gold; oldR2-R8 tests regression only. Training forms differ from new dev/test wording. Long/short/reordered not independent facts.', 'selection': 'All9 final960step thresholds .5,.6,.7,.8,.9 and querygate0,.4,.6 only historicaldev+newdev, minimizeunknownstrong thenmaxmacroF1/citation thenlowergate/threshold; lockall beforetest output. Also3frozenR8LEX references unchangedthresholds zero updates.', 'signals': 'All3seeds must strictly improve BOTH newlong and reordered correct/citationF1, keep short correct/citation and fieldF recall nondecreasing, unknownstrong nonincreasing, and every historical/R2-R8 regression correct/citation nondecreasing. Compare LANG_SFT->FROZEN and LANG_CONTRAST->LANG_SFT separately; also paired frozenR8 comparisons. No bestseed promotion.', 'limits': '512 encoder tokens; overlimit rejected no truncation. Only shadow; original evidence/unknown/conflict/explicit rules unchanged. No formal77/LAB9/realwebsite4core8GiB/concurrency claim. End this bounded round, no infinite same-recipe repeats.', 'encoder_sha256': json.loads((PREV / 'package/protocol.json').read_text('utf-8'))['encoder_sha256'], 'references': ['httpREPRO_ROOT/2106.09685', 'httpREPRO_ROOT/2104.08821', 'httpREPRO_ROOT/2209.11055']}
    write(package / 'protocol.json', protocol)
    files = {p.relative_to(package).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(package.rglob('*')) if p.is_file()}
    write(package / 'input_manifest.json', files)
    with zipfile.ZipFile(OUT / 'inputs.zip', 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in sorted(package.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(package).as_posix())
    receipt = {'input_zip_sha256': sha(OUT / 'inputs.zip'), 'input_zip_bytes': (OUT / 'inputs.zip').stat().st_size, 'protocol_sha256': sha(package / 'protocol.json'), 'files': len(files), 'new_scenes_sha256': sha(package / 'data/new_scenes.json'), 'long_pairs_sha256': sha(package / 'data/long_pairs.json'), 'split_sizes': {s: sum((r['split'] == s for r in rows)) for s in sorted({r['split'] for r in rows})}, 'long_pairs': len(triples)}
    write(OUT / 'preparation.json', receipt)
    print(json.dumps(receipt))
if __name__ == '__main__':
    main()
