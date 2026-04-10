# gemma4-fine-tuning: training plan

The goal is to adapt `google/gemma-4-E4B` to Polish language. Fine tuning runs on a Polish text database (the main corpus). No Formulo reasoning traces are consumed. At most, CKE matura examples can enter a small instruction slice.

## Stage 0. Base model preparation

Download `google/gemma-4-E4B` (and optionally `google/gemma-4-E4B-it`) to `$SCRATCH/models/`, register in the local HF cache, verify bf16 load on one GH200 GPU. Context length target is at least 4096 tokens (8192 if memory allows after packing).

## Stage 1. Polish text corpus

Primary corpus candidates (to be decided at download time, pick one or blend):

1. SpeakLeash Bielik corpus (Polish web and literary text, deduplicated)
2. HPLT v2 Polish subset
3. CulturaX Polish subset
4. mC4 Polish subset (as a fallback)
5. Polish Wikipedia dump (small, as a quality seed)

Pipeline:

1. Download raw shards to `$BIELIK_R_DATA/corpus/raw/`.
2. Normalize Unicode (NFC), strip control characters, fix mojibake.
3. Deduplicate with MinHash LSH (character 5 gram shingles, Jaccard threshold 0.8).
4. Filter by language ID confidence (fastText lid.176, keep pl with score above 0.9).
5. Filter by length (drop documents under 200 characters and over 200k characters).
6. Tokenize with the Gemma 4 tokenizer, pack to fixed length sequences.
7. Shuffle shards and write to `$BIELIK_R_DATA/corpus/packed/` as parquet or JSONL.

Optional instruction slice (at most 5 percent of steps): CKE matura items reformatted as `question / answer` pairs. Source files go to `$BIELIK_R_DATA/cke/` and are treated as a separate dataset that is interleaved at the data loader level, not mixed into the pretraining shards.

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

## Compute budget (Helios)

One GH200 node handles Gemma 4 E4B comfortably at bf16. Multi GPU is for throughput.

- Data preparation: 1 node, 16 CPUs, 6 to 12 hours
- SFT: 1 node, 4 GPUs, roughly 48 to 96 hours per pass over the corpus
- Evaluation: 1 GPU, roughly 4 hours per sweep

Storage budget: 1 TB on `$SCRATCH` (base model, raw corpus shards, packed shards, checkpoints, logs).

## Open questions

1. Which Polish corpus exactly? SpeakLeash Bielik shards vs HPLT v2 vs CulturaX vs a blend.
2. Does `google/gemma-4-E4B` exist at that handle, or is the correct handle `google/gemma-3n-E4B-it`? Verify at download time.
3. Tokenizer sanity: how many tokens does "wyroznik" consume in the Gemma 4 tokenizer? If the vocabulary is very English heavy, consider a tokenizer extension before training.
4. Do we need to hold out a specific set of sources (news, Wikipedia) as a clean eval split?
