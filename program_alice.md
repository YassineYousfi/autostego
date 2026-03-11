# autostego

This is an experiment to have LLMs do steganography and steganalysis research

Eve has put her 3 best steganalysis detectors in the `steganalysis/` directory. Read them. She specifically targeted the 3 steganographic algorithms that you have put in `steganography/`.

Your goal is to make the best new steganographic algorithm to embed 0.4 bpp evading Eve's detectors, use the cover images in `data/BOSSbase_1.01/`.

3 SOTA steganographic algorithms are given to you in the `steganography/` folder, feel free to get inspiration from them, tune them, or write a completely new one.

## Setup

1. **Agree on a run tag**: propose a branch tag, the branch `autostego/alice-<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autostego/alice-<tag>` from current branch `autostego/alice`.
3. **Read the in-scope files**: The repo is small. Read all the python files to get full context.
4. **setup uv environement**: use uv to start activate the venv.

Once you do all, kick off the experimentation.

## Experimentation

Each experiment runs on all GPUs available. The entire pipeline runs in `alice.py` but you can run individual scripts (e.g. `steganography/embed_dir.py` or `steganalysis/srnet.py` etc.)

**What you CAN do:**
- You can tune current algorithms under `steganography/`
- You can create new algorithms
- You can read the detectors that Eve wrote and reason about their weaknesses and strengths
- You can experiment on smaller scale datasets / a subset of the detectors by changing `alice.py` config and detectors as you wish - but the final evaluation of your algorithm has to be done on the 10k dataset and using all three unchanged detectors code in `steganalysis/` trained on your embedded images

**What you CANNOT do:**
- You cannot change the detectors that Eve wrote in `steganalysis/` for the final eval
- You cannot change the payload
- You cannot change the dataset for the final evaluations

**The goal is simple: make the most secure hiding algorithm, i.e. the one with the lowest detection accuracy with Eve's detectors**

**The first run**: Your very first run should always be to establish a baseline.

## Experiment Result format
```
algo_name: detector_name_1: best val accuracy
algo_name: detector_name_2: best val accuracy
algo_name: detector_name_3: best val accuracy
```

## Logging results
for each run save the config from `alice.py` and the result file in the desired format in the `runs/` dir.

## The experiment loop

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune the hiding algorithms in `steganography/` such as ``steganography/hill.py` or create a new one
3. Run `alice.py` with potentially a small number of train and val images
4. Use your own judgement to decide if your new detector is better or not
5. If you think it's better, run the full detection loop using `alice.py` and no image limit
6. Read out the results and log them
7. if the accuracy decreased commit your code and try to improve it even more
8. if the accuracy didn't decrease, try harder

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Running on all images can take a long time, start by using a small value for `max_train_files` and `max_val_files` to iterate fast.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
