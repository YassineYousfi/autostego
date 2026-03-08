# autostego

This is an experiment to have LLMs do steganography and steganalysis research

## Setup

Alice has put her 3 best steganographic algorithms in the `steganography/` directory. Read them.

1. **Agree on a run tag**: propose a branch tag, the branch `autostego/eve-<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autostego/eve-<tag>` from current branch `autostego/eve`.
3. **Read the in-scope files**: The repo is small. Read all the python files to get full context.
4. **setup uv environement**: use uv to start activate the venv.

Once you do all, kick off the experimentation.

## Experimentation

Each experiment runs on all GPUs available. The entire pipeline runs in `eve.py` but you can run individual scripts (e.g. `steganography/embed_dir.py` or `steganalysis/srnet.py` etc.)

**What you CAN do:**
- You can tune current detectors under `steganalysis/`
- You can create new detectors
- You can read the algorithms that Alice wrote and reason about their weaknesses and strengths
- You can make a detector for each algorithm, it doesn't have to be one single detector for all
- You can give detectors names such as `newnet_for_hill.py`

**What you CANNOT do:**
- You cannot change the algorithms that Alice wrote in `steganography/`
- You cannot change the payload
- You cannot change the dataset for the final evaluations

**The goal is simple: get the highest val accuracy on each algorithm**

**The first run**: Your very first run should always be to establish a baseline.

## Output format

## Logging results

## The experiment loop

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune detectors in `steganography/` such as ``steganography/srnet.py` or create a new one
3. Run `eve.py` with potentially a small number of train and val images
4. Use your own judgement to decide if your new detector is better or not
5. If you think it's better, run the full detection loop using `eve.py` and no image limit
6. Read out the results and log them
7. if the accuracy improved commit your code and try to improve it even more
8. if the accuracy didn't improve, try harder

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Running on all images can take a long time, start by using a small value for `max_train_files` and `max_val_files` to iterate fast.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
