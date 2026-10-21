# GitHub upload notes

Uploaded from `Famu_Evidence_Bound_Risk_Field_Materials_20261021.zip`.

- Archive SHA256: `7225ddf960572dc7251b9948a889117408105ef8674154620b2f8ed22ae5e9ca`.
- Original archive: 1,141 files, 524,315,519 uncompressed bytes.
- All original files are preserved byte for byte, with their original paths.
- This note is additional upload documentation and is not listed in the original checksum manifests.

## Original archive discrepancy

The root `MANIFEST_SHA256.json` and both root READMEs reference `整理与校验记录.json`, but that file is absent from the supplied ZIP archive. It has not been reconstructed, and the original manifest has not been changed. Every other manifest-listed file present in the archive matches its recorded size and SHA256.

The `01_Dataset/` manifest verifies all 44 listed files. The `02_Reproduction/` manifest verifies all 1,092 listed files. The manifests themselves are excluded from their own file lists.

Running `python -B tools/verify_release.py` inside `02_Reproduction/` passed: 1,092 file hashes, 1,085 syntax checks, 28 real-material question references, 357 student evidence anchors, and 1,076 source fingerprints.

## Reproduction scope

This repository contains the de-identified dataset, frozen code, protocols, and archived predictions supplied in the archive. Model weights and controlled original materials are not included. The upload does not add a new license or change the materials' existing terms.
