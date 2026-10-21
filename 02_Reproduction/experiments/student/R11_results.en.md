# R11: Contiguous evidence spans and training-data replay

[中文](R11_中文结果_连续跨度与训练回放.md) · [Reproduction guide](../../README.en.md)

R11 compares token-level selection with contiguous-span selection and examines the effect of replaying earlier training material. Four training configurations use seeds 17, 29, and 43, giving 12 training runs. Archived checks cover 24 complete checkpoints, 360 prediction sets, and CPU loading checks for 12 models.

**No comparison passed the predefined requirement of improvement across all three seeds without regression on earlier tasks.** Interpret new scenarios, three wording variants, historical regressions, and evidence metrics together.

## Configurations and evaluation

TOKEN selects source text token by token and may produce disconnected fragments. SPAN predicts a paragraph class and one contiguous start/end interval, protecting matched date and amount boundaries. SPAN changes both boundary supervision and decoding, so the comparison concerns the representation method as a whole. Preserving complete numbers does not guarantee selection of the correct date or correct reasoning.

NEW uses the current round's paragraphs. MIX adds R10 training paragraphs and complete historical positive training paragraphs. Historical unknown or fully irrelevant relations are not additionally labeled as non-current paragraphs. Within a seed, the four configurations share the frozen encoder, rank-8 / alpha-16 adapter and 5-output-head initialization, and within-group sampling order. The original R8 lexical relation heads, thresholds, and query gate remain frozen. R8_FROZEN and R10_SELECT are zero-update controls.

## New synthetic scenarios

|Configuration|Long /96|Reordered /96|Short /96|All three views /96|Long citation F1|Long F recall|Extraction review|
|---|---:|---:|---:|---:|---:|---:|---:|
|TOKEN_NEW_s17|55|51|56|42|0.5705|0.5000|0|
|SPAN_NEW_s17|56|40|56|32|0.5714|0.5000|0|
|TOKEN_MIX_s17|56|72|56|56|0.5714|0.5000|0|
|SPAN_MIX_s17|56|56|56|48|0.5714|0.5000|0|
|TOKEN_NEW_s29|48|55|56|47|0.5526|0.5000|7|
|SPAN_NEW_s29|40|24|56|24|0.5714|0.5000|32|
|TOKEN_MIX_s29|56|56|40|40|0.5714|0.5000|0|
|SPAN_MIX_s29|40|40|40|40|0.5714|0.5000|32|
|TOKEN_NEW_s43|56|40|72|32|0.5714|0.5000|0|
|SPAN_NEW_s43|40|16|56|16|0.5714|0.5000|32|
|TOKEN_MIX_s43|55|56|56|55|0.5691|0.5000|1|
|SPAN_MIX_s43|40|16|40|16|0.5714|0.5000|32|
|R8_FROZEN_s17|16|40|56|16|0.3636|0.4545|72|
|R8_FROZEN_s29|16|40|40|8|0.3636|0.4545|72|
|R8_FROZEN_s43|16|40|40|8|0.3636|0.4545|72|
|R10_SELECT_s17|56|44|56|38|0.5714|0.5000|0|
|R10_SELECT_s29|30|54|40|20|0.3942|0.1591|41|
|R10_SELECT_s43|40|50|56|26|0.0435|0.0455|0|

Long, short, and reordered inputs each contain 96 examples and share the same underlying facts. “All three views” requires correct answers for every version of a scenario. F recall and extraction review follow the archived metric definitions.

## Historical regressions and predefined criterion

TOKEN_NEW→SPAN_NEW, TOKEN_MIX→SPAN_MIX, TOKEN_NEW→TOKEN_MIX, and SPAN_NEW→SPAN_MIX are the predefined 2×2 main comparisons. R10_SELECT→SPAN_MIX and R8_FROZEN→SPAN_MIX are supplementary comparisons. For every comparison, the pass signals for seeds 17, 29, and 43 are `[False, False, False]`, and the joint pass signal is `False`.

The criterion requires increased long-input and reordered accuracy counts and citation F1 for every seed; no decline on short inputs, any historical regression, or known-F recall; and no increase in unknown cases forced into either definite direction. All seeds are retained.

|Configuration|test|r2_regression|r3_regression|r4_regression|r5_regression|r6_regression|r7_regression|r8_regression|r8_regression_short|r9_regression|r9_regression_short|r9_regression_reordered|r10_regression|r10_regression_short|r10_regression_reordered|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|TOKEN_NEW_s17|116|54|50|50|32|72|37|59|64|56|46|32|64|71|42|
|SPAN_NEW_s17|106|32|6|64|8|88|56|88|96|40|48|40|40|40|16|
|TOKEN_MIX_s17|144|72|6|45|13|41|8|72|88|41|68|45|59|40|64|
|SPAN_MIX_s17|121|66|36|64|96|88|44|8|96|48|48|48|40|40|40|
|TOKEN_NEW_s29|87|51|23|65|8|71|53|32|78|55|79|56|55|64|56|
|SPAN_NEW_s29|72|66|56|48|8|88|16|8|94|88|88|87|72|72|56|
|TOKEN_MIX_s29|144|71|5|35|8|8|16|58|28|47|62|50|64|72|55|
|SPAN_MIX_s29|144|66|36|48|11|88|51|8|96|88|88|88|72|72|71|
|TOKEN_NEW_s43|8|46|42|53|71|67|44|53|64|40|38|40|56|72|51|
|SPAN_NEW_s43|72|72|55|79|96|88|51|8|90|72|64|48|72|72|48|
|TOKEN_MIX_s43|144|72|36|15|6|82|41|74|32|32|48|47|64|64|56|
|SPAN_MIX_s43|144|72|36|45|96|0|8|8|0|72|72|72|72|72|72|
|R8_FROZEN_s17|144|72|72|64|96|88|56|48|88|32|48|8|48|40|24|
|R8_FROZEN_s29|144|72|72|64|64|88|56|40|88|32|88|32|48|72|40|
|R8_FROZEN_s43|144|72|66|88|96|88|56|48|96|32|72|32|16|72|48|
|R10_SELECT_s17|120|66|48|71|88|48|58|63|94|42|52|31|56|64|48|
|R10_SELECT_s29|114|72|0|53|8|85|64|74|95|40|80|46|56|64|56|
|R10_SELECT_s43|102|70|18|68|51|64|55|80|96|40|55|41|54|64|50|

See [independent_audit.json](independent_audit.json) for axis-level scores, field recall, invalid citations, unknown cases forced into definite directions, not-established recall, group-level correctness, and paired seed differences. That record independently checks source anchors, original input hashes, and explicit rules across all 360 prediction sets.

## Source spans and numeric information

|Configuration|View|Character-class F1|Exact-match paragraphs / total|
|---|---|---:|---:|
|TOKEN_NEW_s17|new_dev|0.9091|358/576|
|TOKEN_NEW_s17|new_test|0.9380|472/768|
|TOKEN_NEW_s17|new_test_short|0.8728|664/768|
|TOKEN_NEW_s17|new_test_reordered|0.4047|384/768|
|SPAN_NEW_s17|new_dev|0.9590|504/576|
|SPAN_NEW_s17|new_test|0.9555|752/768|
|SPAN_NEW_s17|new_test_short|0.8899|688/768|
|SPAN_NEW_s17|new_test_reordered|0.4291|480/768|
|TOKEN_MIX_s17|new_dev|0.9439|414/576|
|TOKEN_MIX_s17|new_test|0.9589|384/768|
|TOKEN_MIX_s17|new_test_short|0.8289|688/768|
|TOKEN_MIX_s17|new_test_reordered|0.7919|384/768|
|SPAN_MIX_s17|new_dev|0.9290|407/576|
|SPAN_MIX_s17|new_test|0.9561|752/768|
|SPAN_MIX_s17|new_test_short|1.0000|768/768|
|SPAN_MIX_s17|new_test_reordered|0.7754|656/768|
|TOKEN_NEW_s29|new_dev|0.9280|474/576|
|TOKEN_NEW_s29|new_test|0.9271|467/768|
|TOKEN_NEW_s29|new_test_short|0.8339|672/768|
|TOKEN_NEW_s29|new_test_reordered|0.7878|470/768|
|SPAN_NEW_s29|new_dev|0.9577|510/576|
|SPAN_NEW_s29|new_test|0.9367|744/768|
|SPAN_NEW_s29|new_test_short|0.8899|688/768|
|SPAN_NEW_s29|new_test_reordered|0.6313|584/768|
|TOKEN_MIX_s29|new_dev|0.9583|436/576|
|TOKEN_MIX_s29|new_test|0.9358|410/768|
|TOKEN_MIX_s29|new_test_short|0.8050|688/768|
|TOKEN_MIX_s29|new_test_reordered|0.8986|560/768|
|SPAN_MIX_s29|new_dev|0.9401|499/576|
|SPAN_MIX_s29|new_test|0.9561|752/768|
|SPAN_MIX_s29|new_test_short|1.0000|768/768|
|SPAN_MIX_s29|new_test_reordered|0.9197|736/768|
|TOKEN_NEW_s43|new_dev|0.8947|414/576|
|TOKEN_NEW_s43|new_test|0.9113|392/768|
|TOKEN_NEW_s43|new_test_short|0.8366|568/768|
|TOKEN_NEW_s43|new_test_reordered|0.6115|384/768|
|SPAN_NEW_s43|new_dev|0.9590|504/576|
|SPAN_NEW_s43|new_test|0.9337|744/768|
|SPAN_NEW_s43|new_test_short|0.8899|688/768|
|SPAN_NEW_s43|new_test_reordered|0.4291|480/768|
|TOKEN_MIX_s43|new_dev|0.9430|408/576|
|TOKEN_MIX_s43|new_test|0.9527|552/768|
|TOKEN_MIX_s43|new_test_short|0.9479|680/768|
|TOKEN_MIX_s43|new_test_reordered|0.7968|392/768|
|SPAN_MIX_s43|new_dev|0.9365|432/576|
|SPAN_MIX_s43|new_test|0.9780|760/768|
|SPAN_MIX_s43|new_test_short|0.8016|688/768|
|SPAN_MIX_s43|new_test_reordered|0.5848|552/768|

|Configuration|Long-input field|Paragraphs|Numbers match current facts|Empty selections|All current characters retained|Gold numeric atoms retained / total|Fragmented numeric atoms|
|---|---|---:|---:|---:|---:|---:|---:|
|TOKEN_NEW_s17|B|88|88|0|88|264/264|0|
|TOKEN_NEW_s17|D|88|88|0|0|440/440|0|
|TOKEN_NEW_s29|B|88|88|0|88|264/264|0|
|TOKEN_NEW_s29|D|88|88|0|0|424/440|14|
|TOKEN_NEW_s43|B|88|88|0|0|264/264|0|
|TOKEN_NEW_s43|D|88|88|0|0|440/440|0|
|SPAN_NEW_s17|B|88|88|0|88|264/264|0|
|SPAN_NEW_s17|D|88|88|0|88|440/440|0|
|SPAN_NEW_s29|B|88|88|0|88|264/264|0|
|SPAN_NEW_s29|D|88|88|0|88|440/440|0|
|SPAN_NEW_s43|B|88|88|0|88|264/264|0|
|SPAN_NEW_s43|D|88|88|0|88|440/440|0|
|TOKEN_MIX_s17|B|88|88|0|88|264/264|0|
|TOKEN_MIX_s17|D|88|88|0|0|440/440|6|
|TOKEN_MIX_s29|B|88|88|0|88|264/264|0|
|TOKEN_MIX_s29|D|88|88|0|3|440/440|0|
|TOKEN_MIX_s43|B|88|88|0|88|264/264|0|
|TOKEN_MIX_s43|D|88|88|0|0|439/440|1|
|SPAN_MIX_s17|B|88|88|0|88|264/264|0|
|SPAN_MIX_s17|D|88|88|0|88|440/440|0|
|SPAN_MIX_s29|B|88|88|0|88|264/264|0|
|SPAN_MIX_s29|D|88|88|0|88|440/440|0|
|SPAN_MIX_s43|B|88|88|0|88|264/264|0|
|SPAN_MIX_s43|D|88|88|0|88|440/440|0|
|R10_SELECT_s17|B|88|88|0|88|264/264|0|
|R10_SELECT_s17|D|88|88|0|0|440/440|1|
|R10_SELECT_s29|B|88|88|0|88|264/264|0|
|R10_SELECT_s29|D|88|25|0|0|421/440|19|
|R10_SELECT_s43|B|88|88|0|0|244/264|20|
|R10_SELECT_s43|D|88|0|0|0|176/440|259|

Valid anchor positions do not establish semantic correctness. Numbers can remain unfragmented while an entire paragraph is omitted. Repeated paragraphs across related scenarios are correlated samples. Gold source spans support supervision and independent auditing; inference reads questions, scope, and source text. SPAN supports one contiguous interval.

## Training cost and data scope

The 12 runs contain 11,520 updates and 368,640 paragraph exposures, with 0 relation-anchor exposures. Supervision covers 20,126,143 token-class positions, 184,320 paragraph-class positions, and 230,400 boundary positions. The experiment uses no new teacher calls or reinforcement learning.

|Configuration|Training seconds|Effective encoded tokens|Padded tokens|Paragraphs|Token labels|Paragraph-class labels|Boundary labels|
|---|---:|---:|---:|---:|---:|---:|---:|
|TOKEN_NEW_s17|37.50|6933821|8105056|30720|3585341|0|0|
|SPAN_NEW_s17|39.80|6933821|8105056|30720|0|30720|38400|
|TOKEN_MIX_s17|37.40|6501012|8048320|30720|3117972|0|0|
|SPAN_MIX_s17|39.39|6501012|8048320|30720|0|30720|38400|
|TOKEN_NEW_s29|37.25|6938639|8107776|30720|3590159|0|0|
|SPAN_NEW_s29|39.64|6938639|8107776|30720|0|30720|38400|
|TOKEN_MIX_s29|37.16|6504057|8049920|30720|3121017|0|0|
|SPAN_MIX_s29|39.35|6504057|8049920|30720|0|30720|38400|
|TOKEN_NEW_s43|37.17|6936482|8106912|30720|3588002|0|0|
|SPAN_NEW_s43|39.72|6936482|8106912|30720|0|30720|38400|
|TOKEN_MIX_s43|37.34|6506692|8048736|30720|3123652|0|0|
|SPAN_MIX_s43|39.40|6506692|8048736|30720|0|30720|38400|

The archived end-to-end elapsed time is 875.55 seconds. The backward preflight on the longest valid input performs no optimizer update. Configuration-level `metrics.json` files record development encoding, selection, baseline, and rule-inference costs. Equal exposures do not imply equal FLOPs.

The data contain 1,152 training scenarios, 72 development scenarios, 96 new test scenarios, and 96 short and 96 reordered versions with the same facts. Training, development, and test families and new wording are split separately but share a program structure; the scenarios are synthetic. Once evaluated, these test sets should be treated as seen regression data in later experiments. See [protocol.json](protocol.json) and related archives for data, initialization, sampling-order, code, and protocol fingerprints.

## CPU loading and input limits

|Configuration|Increment MiB|12-scenario p50 seconds|p95 seconds|Peak process MiB|Maximum CPU/GPU score difference|
|---|---:|---:|---:|---:|---:|
|TOKEN_NEW_s17|2.35|0.542|0.588|1498.0|8.8e-08|
|SPAN_NEW_s17|2.35|0.535|0.547|1497.5|1.1e-07|
|TOKEN_MIX_s17|2.35|0.537|0.554|1499.0|8.8e-08|
|SPAN_MIX_s17|2.35|0.545|0.564|1500.0|1.1e-07|
|TOKEN_NEW_s29|2.35|0.545|0.560|1498.5|5e-08|
|SPAN_NEW_s29|2.35|0.539|0.550|1497.7|7.3e-08|
|TOKEN_MIX_s29|2.35|0.545|0.556|1498.2|4.3e-08|
|SPAN_MIX_s29|2.35|0.543|0.553|1497.9|4.5e-08|
|TOKEN_NEW_s43|2.35|0.538|0.549|1498.7|5.5e-08|
|SPAN_NEW_s43|2.35|0.537|0.554|1498.9|5.5e-08|
|TOKEN_MIX_s43|2.35|0.551|0.591|1498.1|5.5e-08|
|SPAN_MIX_s43|2.35|0.540|0.554|1497.7|5.5e-08|

For new test inputs, a single evidence paragraph including its scope prefix has a median length of 238.5 tokens; p95 and maximum are both 269. A case contains at most 1,031 source-text characters across all evidence. Archived CPU checks fully load each of the 12 models in a fresh process and use 4 threads to verify reference scores and selected spans, fields, and decisions for 12 scenarios.

An evidence paragraph plus its normalized scope prefix is rejected above 512 tokens and is not truncated. CPU checks do not cover a production 4-core / 8-GiB hard limit, concurrency acceptance, or acceptance on the 77 formal tasks / LAB9; the results support research evaluation.

After obtaining controlled model parameters and the matching encoder, use the loading interface below from `experiments/student/`, replacing placeholders with actual object paths:

```python
from pathlib import Path
from src.evidence_reasoning.continuous_span_student import ContinuousSpanStudent

model = ContinuousSpanStudent.load(
    Path("/path/to/student/model"),
    Path("/path/to/encoder"),
)
# result = model.review(payload)  # scope / fields / rule / evidence
```

The original experiment shares an approximately 96-MB base encoder and uses branch-specific model parameters. The 24 complete checkpoints record optimizer, scheduler, random-number state, and exposure information. Binary parameters and complete checkpoints are controlled objects; the repository provides corresponding configurations, fingerprints, and results.

## Training implementation

Before the first backward pass, the historical run encountered a variable-name collision: the LoRA module list overwrote paragraph boundary `targets`. The variables were separated at zero optimizer updates, preserving data, losses, initialization, optimizer, exposure, and the training endpoint. Exact reruns should check the corresponding repaired implementation and execution fingerprints and perform a backward preflight. The original run's dedicated recovery script is not included in this repository.

## Post-hoc oracle-extraction diagnosis

The diagnosis supplies manually annotated current source spans to three frozen R8 lexical heads, separating extraction errors from downstream classifier limitations. It supplies neither field classes nor risk answers and changes neither weights nor thresholds. Because it uses gold excerpts, this is post-hoc analysis under ideal extraction conditions. Its 9 additional diagnostic predictions are counted separately from the 360 fixed evaluation sets.

|Seed|View|Oracle-extraction correct /96|Citation F1|
|---|---|---:|---:|
|17|test|56|0.5714|
|17|test_short|56|0.5714|
|17|test_reordered|56|0.5714|
|29|test|40|0.5714|
|29|test_short|40|0.5714|
|29|test_reordered|40|0.5714|
|43|test|40|0.5714|
|43|test_short|40|0.5714|
|43|test_reordered|40|0.5714|

Errors remain after oracle extraction, indicating that correcting extraction alone is insufficient for these inputs. This does not establish the downstream classifier as the sole cause of every error. Completing scenario identifiers in the diagnostic records changes neither weights, spans, thresholds, nor normal evaluation. One additional frozen lexical-head batch is counted separately in cost. Subsequent method comparisons require a new held-out set.
