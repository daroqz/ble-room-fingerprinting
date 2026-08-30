# ble-room-fingerprinting

Room-level indoor positioning from BLE signal strength — for **real phones**,
not cooperative beacons.

Four ESP32 proxies and a Raspberry Pi listen to an iPhone whose Bluetooth
address rotates every few minutes, resolve it via its Identity Resolving Key,
and learn which of five rooms it is in. A robot vacuum with a beacon taped to it
generates ground truth by driving around.

This repository is as much about **what did not work** as about what did. Most of
the interesting content is negative results with numbers attached.

> **Status: instrumentation works, no trained model in production yet.**
> A trivial rule (strongest anchor + hysteresis) currently outperforms the
> learned model on the phone domain. Section [Status](#status) is honest about
> where the line is.

---

## Why another indoor-positioning repo

Most BLE positioning projects track a **cooperative transmitter**: a beacon, a
tag, a dev board. It broadcasts at a constant power, at a constant rate, from a
fixed height, and it is the same device during training and inference.

A phone is none of those things. It rotates its address, transmits
opportunistically, sits in a pocket at 90 cm or on a table at 75 cm, and is
shadowed by a body. Train on a beacon and deploy on a phone, and you have a
**domain shift** — one that this repository measures rather than assumes.

That single difference produces most of the findings below.

---

## Findings

Every number here was measured in this repository and has a script behind it.
Base rates are quoted alongside accuracies, because an accuracy without one is
not a result.

### 1. A model can learn from an anchor's *silence*

The learned model jumped between rooms 399 and 898 times during a five-hour
night in which both phones provably never left the bedroom. The raw signal was
excellent — over 90 % of windows saw four or more anchors, and the strongest
anchor was correctly the bedroom in 97.5 % / 93.9 % of them.

The cause was not noise. One anchor heard the *beacon* from the bedroom in
**0 of 105** calibration points, but heard the *phones* in 48.5 % and 93.4 % of
windows. The model had learned `anchor silent ⇒ bedroom` — a
missing-not-at-random shortcut that is true for the beacon and false for a phone.

| Feature set | in-domain macro-F1 | transfer p1 | transfer p2 |
|---|---|---|---|
| 5 anchors | 0.783 | 82.9 % | **24.1 %** |
| 4 anchors, dropping that one | 0.737 | **95.2 %** | **83.7 %** |

Removing one feature costs 0.046 macro-F1 in-domain and gains **59 percentage
points** out-of-domain.

### 2. Smoothing amplifies a systematic error

The obvious fix for a jumpy classifier is temporal smoothing. Measured over the
same night, k-of-n hysteresis makes the worse case *worse*:

| | k=1 | k=3 | k=5 | k=10 |
|---|---|---|---|---|
| 5 anchors, p2 | 24.1 % | 15.1 % | 9.0 % | **8.7 %** |
| 4 anchors, p2 | 83.7 % | **98.0 %** | 98.4 % | 95.3 % |

Smoothing is a majority vote. A model that is confidently wrong wins it more
convincingly. **Fix the features first, smooth second.**

### 3. The radio picture identifies the *device*, not just the place

Given only the per-anchor levels, a classifier separates "which device is this"
at **99.2 %** accuracy (base rate 75.3 %, n=247, one room).

The consequence is operational: a measurement device that only ever visits one
room teaches the model *"this is what that device sounds like"* rather than
*"this is what that room sounds like"*. Any device used for data collection has
to visit every target room, or it poisons the set.

### 4. A room is a region, but not every room

The same tablet, moved between four positions inside one kitchen, spans **40 dB**
on the kitchen anchor (−83 to −43 dBm). At one of those positions the trivial
rule drops to 58.3 % where it scores 96–100 % at the others. The living room,
measured identically, spans 18 dB and scores 100 % everywhere.

So "rooms have internal structure" is not a general truth to design around — it
is a symptom of one badly placed anchor. Positions inside a room are recorded
explicitly so that cross-validation can group by *standpoint* rather than by
room.

### 5. Trilateration does not survive contact with the data

Fitting a log-distance path-loss model per anchor gives exponents between
**1.34 and 5.15**. Three of them fall outside the physically meaningful range of
2–5. Distance-from-RSSI is not recoverable here, which is why this repository is
fingerprinting-only.

### 6. Most reported improvements are seed luck

A null experiment (`a_schein`) changes no data at all but draws the same random
numbers as the augmentation it is compared against. It alone accounts for
**−0.111 ± 0.023 dB**, better on 10 of 10 seeds. Several findings did not survive
that control:

| Claim | raw | after control | verdict |
|---|---|---|---|
| Masking strategy `b_zwei_anker` | −0.4 dB | 3 of 3 fresh seeds | **holds** |
| Hyperparameter search | 5.83 dB | 2 of 3 seeds | seed luck |
| Sequence length 120 | −0.262 dB, p=0.009 | −0.078 dB, p=0.353 | vanishes |
| Weight decay | 0.044 dB spread | seed spread 0.249 dB | below noise |
| Dropout augmentation | −0.146 dB | −0.035 dB | mostly the RNG |

---

## Self-supervised pretraining

The label economics are brutal: roughly **35,000 unlabelled 10-second windows
per day** against a few thousand labelled points in total. That ratio is the
textbook case for pretraining without labels.

Input is a tensor of `30 windows × 5 anchors × 5 features`
(`rssi_median`, `rssi_std`, `count`, `heard`, `age_s`). Five encoders are trained
on three label-free pretext tasks, then frozen and probed linearly.

| Encoder | Parameters | Probe RMSE (masked-anchor) |
|---|---|---|
| TCN | 43,264 | **6.145 dB** |
| GRU | 30,784 | 6.241 dB |
| Transformer | 74,688 | 6.290 dB |
| Anchor-attention | 42,432 | 6.567 dB |
| MLP | 9,984 | 6.644 dB |

Baselines in the same setup: predicting the global mean gives **11.653 dB**,
predicting each anchor's own mean gives **10.787 dB**. The best masking strategy
reaches **5.49 dB**. The representation halves the error of the trivial
baseline — learned from unlabelled data alone.

On an independent downstream probe (a label-free room proxy), the pretrained
encoder reaches 0.706 balanced accuracy against 0.613 for a raw linear model and
a 0.200 base rate.

**What this is not.** The probe measures decibels, not rooms. No neural network
in this repository has yet classified a room with real labels. The encoder knows
the radio geometry of the flat; it has not been shown the actual task. Closing
that gap — freeze the encoder, attach a room head, evaluate against the same
folds as the gradient-boosting baseline — is the next step, not a finished one.

---

## Status

| Component | State |
|---|---|
| Capture pipeline | works — 46k windows, 10 s cadence, 0 unparsable lines |
| Anchors | 4 ESP32 proxies + host adapter, all online |
| Online detection | works — 10 s latency, rule-based |
| Self-supervised pretraining | works, measured against baselines |
| Calibration via robot | works — 2,381 labelled points over 3 drives |
| **Trained production model** | **none** — label thresholds not met |
| **Neural room classifier** | **not built** — pretraining only |
| Deployment to consumers | shadow mode only |

The honest summary: the instrumentation and the evaluation discipline are the
mature parts. The models are not.

---

## How it is built

```
4× ESP32 BLE proxy ─┐
host BT adapter ────┼─► Home Assistant diagnostics endpoint (3 s poll)
                    │   (bypasses the scanner-ownership arbitration that
                    │    otherwise reduces you to one anchor per device)
Wi-Fi controller ───┤
motion sensors ─────┤
robot vacuum ───────┘   position in mm + room segment  ── ground truth
                              │
                              ▼
                    10-second windowing → JSONL (system of record)
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
        transfer to      online rule      calibration
        the server       + hysteresis      recorder
              │
              ▼
    schema check → feature build → GBM → report + shadow evaluation
              │
              └─► lab: PyTorch, self-supervised pretraining
```

**Hardware.** Four ESP32 (three S3, one C6) running ESPHome as Bluetooth
proxies, one Raspberry Pi as host, one robot vacuum carrying a BLE beacon, one
iPhone and one iPad as the actual targets.

**Why the diagnostics endpoint.** Home Assistant arbitrates a single owning
scanner per device, deduplicates identical advertisement payloads, and filters
vendor frames. Subscribing to advertisements yields ~4 packets per minute from
*one* anchor. Polling the diagnostics endpoint instead yields every anchor's own
view of every device — 3–4 measurements per anchor per window. Without this
there is no fingerprint at all. It is a debug interface with no stability
guarantee; check it after upgrades.

---

## Evaluation

The rules this repository holds itself to. They exist because breaking them
produced numbers that looked good and were not.

1. **Group by session, not by point.** Neighbouring points in a 3-second
   recording stream are near-duplicates. Grouping by 1 m grid cell gives 0.850;
   grouping by session gives **0.741**. The second number is the true one.
2. **Quote the base rate next to every accuracy.** Here: 0.617 accuracy /
   0.153 macro-F1 for the majority class.
3. **Cross-domain transfer is a permanent regression test**, not a one-off
   analysis. It is what exposed finding #1.
4. **Ablate, don't anecdote.** Every new feature is computed twice, with and
   without, in the same run.
5. **Report event metrics, not only F1.** False room changes per quiet hour,
   detection latency, wake-up time error against an independent motion sensor. A
   model that wins on F1 and loses on these is not deployable.
6. **Run a null control** before believing an improvement.

---

## Repository layout

The code lives in German — variable names, comments and commit messages. Only
this file is in English. The directory names follow suit:

| Path | Contents |
|---|---|
| `pi/` | transfer chain, privacy boundary, schema validation on the capture host |
| `collector/` | windowing, IRK resolution, diagnostics polling (runs inside Home Assistant) |
| `wohnungsort/` | *"where in the flat"* — online detection service, degradation ladder, silver labels |
| `funkkarte/` | *"radio map"* — robot-driven calibration recorder |
| `server/` | ingest, feature building, gradient-boosting training, reports |
| `labor/` | *"lab"* — PyTorch encoders, pretext tasks, ablations, null controls |
| `karte/` | reception maps and experiment dashboard |

Person identifiers never leave the capture host: everything downstream sees
`p1`, `p2`, `p3`. A deny-list rejects MAC addresses, entity identifiers and
similar keys at the boundary, and an entire record is dropped rather than
silently stripped.

---

## Limitations

- **Single flat, single deployment.** Every number here comes from one home with
  one anchor layout. Nothing has been replicated elsewhere.
- **Two people.** Person separation is evaluated on a badly unbalanced set.
- **Transit spaces are excluded from the target classes.** A corridor with no
  anchor of its own is confusable with everything; keeping it in cost 0.089
  macro-F1. That is a modelling choice, not a solved problem.
- **No GPU.** Everything trains on CPU, which bounds model size.
- **The strongest baseline is embarrassingly simple.** Argmax of the loudest
  anchor plus 3-of-n hysteresis produces zero room changes across a quiet night
  and detects getting up within 48 seconds of an independent motion sensor. Any
  learned model has to beat that before it ships.

## License

MIT.
