# gemma4-pl: training plan

The goal is to adapt `google/gemma-4-E4B` to Polish language. Fine tuning runs on a Polish text database (the main corpus). No Formulo reasoning traces are consumed. At most, CKE matura examples can enter a small instruction slice.

## Stage 0. Base model preparation

Download `google/gemma-4-E4B` (and optionally `google/gemma-4-E4B-it`) to `$SCRATCH/models/`, register in the local HF cache, verify bf16 load on one GH200 GPU. Context length target is at least 4096 tokens (8192 if memory allows after packing).

## Stage 1. Polish text corpus

Primary corpus: SpeakLeash. This is the same text pool that powers the Bielik models. It is not a HuggingFace dataset, it is streamed via the `speakleash` Python package. Shards are named (e.g. `plwiki`, `forum_wolnepodroze`, `wolnelektury_pl_txt`) and are pulled individually into local JSONL files under `$GEMMA4_PL_DATA/corpus/raw/speakleash/`.

Auxiliary corpus candidates (optional, can be blended in with `--source`):

1. HPLT v2 Polish subset (`HPLT/HPLT2.0_cleaned`, `pol_Latn`)
2. CulturaX Polish subset (`uonlp/CulturaX`, `pl`)
3. Polish Wikipedia dump (`wikimedia/wikipedia`, `20231101.pl` or newer)
4. OSCAR 2301 Polish subset (gated, needs HF_TOKEN)

Pipeline:

1. Download raw shards to `$GEMMA4_PL_DATA/corpus/raw/`. SpeakLeash shards land under `raw/speakleash/<shard>.jsonl`; HuggingFace sources land under `raw/<source>.jsonl`.
2. Normalize Unicode (NFC), strip control characters, fix mojibake.
3. Deduplicate with MinHash LSH (character 5 gram shingles, Jaccard threshold 0.8).
4. Filter by language ID confidence (fastText lid.176, keep pl with score above 0.9).
5. Filter by length (drop documents under 200 characters and over 200k characters).
6. Tokenize with the Gemma 4 tokenizer, pack to fixed length sequences.
7. Shuffle shards and write to `$GEMMA4_PL_DATA/corpus/packed/` as parquet or JSONL.

Optional instruction slice (at most 5 percent of steps): CKE matura items reformatted as `question / answer` pairs. Source files go to `$GEMMA4_PL_DATA/cke/` and are treated as a separate dataset that is interleaved at the data loader level, not mixed into the pretraining shards.

## Stage 2. Supervised fine tuning (causal LM)

This is the only training stage in the first iteration.

- Objective: next token prediction on packed Polish sequences.
- Loss masking: none (train on full sequence), except inside the CKE instruction slice where the prompt can be masked out.
- Sequence length: 4096 (bump to 8192 if throughput allows).
- Packing: on, EOS between documents.
- Optimizer: AdamW, weight decay 0.01, beta1 0.9, beta2 0.95.
- Learning rate: 5e-5 cosine schedule, warmup 3 percent of steps.
- Batch size: 64 global (micro batch 2 with gradient accumulation 32 per GPU, 4 GPUs per node).
- Precision: bf16 with tf32 matmul.
- Gradient checkpointing: on.
- Epochs: 1 pass over the corpus target; repeat for a second partial pass if the loss is still improving.
- Eval: held out Polish shard, perplexity every 1000 steps.
- Checkpoints: every 2000 steps, keep best 3 by eval ppl.

Output: `checkpoints/sft/`.

## Stage 3. Evaluation

Primary checks (Polish first):

- Perplexity on a held out Polish shard
- PolEval 2024 tasks (sentiment, NER, QA where available)
- MMLU PL (Polish translation of MMLU)
- Matura podstawowa sanity set (10 to 20 items to confirm basic math and Polish literacy, not a benchmark target)
- A qualitative generation probe on 50 Polish prompts

Secondary:

- English MMLU (drift check: should not collapse)
- LAMBADA PL or similar narrative completion (if available)

Targets (relative to the Gemma 4 E4B base):

- Polish perplexity: at least 25 percent reduction on the held out shard
- MMLU PL: at least 3 points absolute improvement
- English MMLU: at most 2 points absolute drop

## Compute budget

One GH200 node handles Gemma 4 E4B comfortably at bf16. Multi GPU is for throughput.

- Data preparation: 1 node, 16 CPUs, 6 to 12 hours
- SFT: 1 node, 4 GPUs, roughly 48 to 96 hours per pass over the corpus
- Evaluation: 1 GPU, roughly 4 hours per sweep

Storage budget: 1 TB on `$SCRATCH` (base model, raw corpus shards, packed shards, checkpoints, logs).

## Open questions

1. Which SpeakLeash shards go into the first run? The default set (`plwiki`, `forum_wolnepodroze`, `forum_gazeta`, `wolnelektury_pl_txt`) is a small sanity run. For a real training pass, decide whether to pull the full corpus with `--speakleash-all` or pick a larger curated subset.
2. Does `google/gemma-4-E4B` load cleanly from the local HF cache on the aarch64 GH200 node? Verify at download time.
3. Tokenizer sanity: how many tokens does "wyroznik" consume in the Gemma 4 tokenizer? If the vocabulary is very English heavy, consider a tokenizer extension before training.
4. Do we need to hold out a specific set of sources (news, Wikipedia) as a clean eval split?
