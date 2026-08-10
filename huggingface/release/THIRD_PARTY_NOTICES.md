# Third-party notices

This release combines ApexOracle-owned model weights and wrapper code with the
following attributed runtime and tokenizer files.

## MDLM runtime

`models/dit.py` and `noise_schedule.py` are derived from the upstream
[`kuleshov-group/mdlm`](https://github.com/kuleshov-group/mdlm) runtime. They
remain available under the Apache License 2.0. A copy is included at
`LICENSES/Apache-2.0.txt`.

## IBM SELFIES tokenizer

`tokenizer.json`, `tokenizer_config.json`, and `special_tokens_map.json` are
from `ibm-research/materials.selfies-ted`, audited at revision
`55e83392264cb998f7aa5014847df29868aefeb8`. The tokenizer repository declares
the Apache License 2.0.

The MIT license in the root of this model repository applies to the
ApexOracle-owned wrapper and frozen model release. It does not replace the
licenses of the third-party files listed above.
