# De-identification and implications for reproduction

[中文](脱敏说明.md) · [Reproduction guide](../README.en.md)

## Identifiers and task information

Real entities use stable aliases that preserve correspondence across contracts, questions, gold answers, and model responses. Names, organizations and abbreviations, contact details, detailed addresses, bank accounts, registration numbers, contract identifiers, case numbers, source links, local usernames, and server locations have been checked and replaced. Identity mappings are controlled materials.

Personnel and account fields unrelated to the questions have been removed from signing and contact pages; relevant clauses and page numbers are retained. The repository excludes original Word, PDF, Excel, and image files, contract attachments, expert sign-offs, original model parameters, optimizer states, and feature caches.

Class labels, random seeds, update and reading counts, field and evidence identifiers, data splits, sufficient evidence sets, and numeric scores are retained. Task-relevant amounts, dates, percentages, contractual relationships, and public legal names and article numbers are preserved. Fictional names distinguish entities in synthetic scenarios, and different parties retain distinct aliases.

## Evidence positions and fingerprints

Experiment 1 evidence character positions have been relocated and checked after entity replacement. Real-question input hashes agree with input fingerprints in scoring keys, archived generations, and parsed outputs. Whole-file SHA256 hashes for controlled originals and repository files are recorded separately.

Text changes may affect tokenization, hidden representations, and predictions. Original feature caches correspond to original inputs. Included `raw` fields retain de-identified historical generation strings. Logged input token counts and timings describe the original runs. Training or inference on de-identified text should record new input fingerprints and results.

## Annotations and interpretation

Expert answers correspond to reviews of the original questions; E01/E02 are anonymous review identifiers. The de-identified materials retain the recorded answers, predictions, seeds, and scoring definitions. Score recomputation checks archived results. Incorrect predictions remain as recorded.

Amounts, dates, or similar public materials may still support source linkage. Removing direct identifiers does not establish irreversible anonymization and does not replace third-party licenses or usage conditions.

## Student-model source-text windows

The real-material student model divides evidence text into windows. For evidence whose text has been replaced, original window excerpts are controlled content. Their positions are identified as `original_start/original_end`, which refer to the original text. Window identifiers, parent evidence identifiers, model judgments, and confidence scores are retained. The complete de-identified evidence is available under its evidence identifier.

Unchanged evidence retains historical excerpts with individually checked positions. The window inventory is `experiments/real_eval/config/student_input_adapter_audit.json`. Answers, cited evidence identifiers, and scoring definitions remain consistent.
