# Duration-model architectures and speech SSL methods that change TTS duration prediction accuracy

> Scope note for the report writer: in this session the egress proxy blocked full-text access to arxiv.org, isca-archive.org, aclanthology.org, openreview.net, semanticscholar.org and huggingface.co (HTTP 403 policy denial). Numbers below therefore come from web-search excerpts of the primary papers (arXiv HTML/PDF, Interspeech PDFs) and, where marked, from secondary summaries. Where a number is from a secondary aggregator it is flagged; treat such numbers as "to verify against the PDF". No number below was invented. Many items the brief asked about (VITS SDP ablation numbers, NaturalSpeech 2/3 duration metrics, Charsiu exact boundary accuracy, speed-perturbation duration error) could not be verified and are listed under Gaps.

## Q1. Stochastic/generative duration predictors (flow, diffusion/flow matching, MaskGIT, AR): do they improve correlation with the reference, or only naturalness/diversity?

### Takeaway
The most directly relevant controlled study (Mehta et al., Interspeech 2024, which includes Matcha-TTS) finds that probabilistic (OT-CFM flow-matching) duration models help naturalness mainly for **spontaneous** speech and for VITS. For Matcha-TTS on **read** speech, deterministic and flow-matching durations were statistically indistinguishable, and FastSpeech 2 got worse. The evidence measures naturalness, intelligibility or distribution match (e.g., Fréchet Duration Distance), not per-utterance correlation with the reference. Sampling-based predictors are not shown to raise ρ against a single reference, so they are unlikely to move a read-speech ρ from 0.74 to 0.8–0.9.

### Cited Findings
- Mehta et al. compare deterministic (DET) duration prediction with OT-CFM flow-matching (FM) durations inside FastSpeech 2, Matcha-TTS and VITS on four corpora (read and spontaneous). Conclusion: "stochastic duration modelling consistently improves probabilistic NAR TTS approaches, especially for spontaneous speech." — [Mehta et al. 2024, arXiv 2406.05401](https://arxiv.org/abs/2406.05401); [Interspeech PDF](https://www.isca-archive.org/interspeech_2024/mehta24b_interspeech.pdf); [project page](https://shivammehta25.github.io/prob_dur/)
- FastSpeech 2 (a regression model) "produced worse speech in all aspects when changing from the deterministic to the stochastic duration model." The authors attribute this to OT-CFM struggling to learn accurate duration distributions inside FS2. — [arXiv 2406.05401](https://arxiv.org/abs/2406.05401)
- Matcha-TTS "synthesised read speech similarly well using both DET and FM, with a significant improvement for spontaneous speech with stochastic duration generation". All CMOS results except LJ Speech and the read-speech set for Matcha were significantly different from zero (p < 0.05). So for Matcha on read speech, FM durations gave **no significant gain**. — [arXiv 2406.05401](https://arxiv.org/abs/2406.05401)
- VITS "demonstrated notable improvements with the stochastic duration predictor … for both read and spontaneous speech." — [arXiv 2406.05401](https://arxiv.org/abs/2406.05401)
- The original VITS paper motivates its flow-based stochastic duration predictor (SDP) as producing "more realistic phoneme duration than the deterministic duration predictor". The evidence is subjective (MOS) and internal to VITS. — [Kim et al. 2021, VITS, arXiv 2106.06103](https://arxiv.org/pdf/2106.06103)
- Counter-evidence on stability: SANE-TTS reports "there are cases where the SDP generates unnatural duration causing unclear pronunciation", so it switched to a deterministic duration predictor (DDP) for stability. — [SANE-TTS, arXiv 2206.12132](https://arxiv.org/pdf/2206.12132)
- VITS2 replaces the plain SDP with an adversarially trained stochastic duration predictor. The search excerpt gave no duration-specific metric. — [VITS2, Interspeech 2023](https://www.isca-archive.org/interspeech_2023/kong23_interspeech.pdf)
- An APSIPA 2025 paper, "Adversarial Learning for Duration Prediction", exists (full text not accessible; no numbers verified). — [APSIPA 2025 P481](https://apsipa.org/proceedings/2025/papers/APSIPA2025_P481.pdf)
- Total-Duration-Aware (TDA) duration modeling (Eskimez et al., Interspeech 2024) conditions phoneme durations on a target total duration and proposes a MaskGIT (discrete-token) duration model. The abstract says it gives "better intelligibility and speaker similarity for various speech rate configurations" and that MaskGIT produces durations "with higher quality and diversity compared to its regression or flow-matching counterparts." — [arXiv 2406.04281](https://arxiv.org/abs/2406.04281); [Interspeech PDF](https://www.isca-archive.org/interspeech_2024/eskimez24_interspeech.pdf)
  - Secondary summary (unverified against the PDF): Fréchet Duration Distance of 0.206 (MaskGIT) vs 0.403 (regression) vs 0.318 (flow matching), and a 58.7% listener preference for TDA on intelligibility. — [Lacuna summary](https://lacuna.tiptreesystems.com/work/total-duration-aware-duration-modeling-for-text). FDD is a distribution distance, not a correlation to the reference.
- DurFormer, from "Adaptive Duration Model for Text Speech Alignment" (Cao, 2025), predicts a mean and variance per phoneme. The abstract claims about an 11.3% "alignment accuracy" improvement and better robustness to prompt/input mismatch in zero-shot TTS. — [arXiv 2507.22612](https://arxiv.org/html/2507.22612v2)
  - A third-party review says the gains are reductions in phoneme-level **MSE** vs FastSpeech 2: about 15.05% (small model) and 11.26% (large model). It also reports about a 5% WER reduction when DurFormer replaces F5-TTS's ratio-scale total-duration rule. — [Pith review](https://pith.science/paper/2507.22612). The metric definition is unverified.
- Diffusion TTS systems that skip phoneme-level duration and predict only total utterance length: "Sample-Efficient Diffusion for TTS" uses a fine-tuned ByT5-base total-duration predictor with **1.4 s RMSE**. The authors argue this avoids cascaded phoneme-duration errors. — [arXiv 2409.03717](https://arxiv.org/html/2409.03717v1)
- ARDiT estimates total duration heuristically: average non-silence phoneme duration × phoneme count, plus punctuation padding. — [ARDiT, arXiv 2406.05551](https://arxiv.org/pdf/2406.05551)
- MaskGCT's released code uses a rule that scales target phoneme length by the prompt's speech-to-phoneme ratio. A 2026 follow-up notes that this breaks for out-of-distribution prompt speaking rates and clips the ratio toward the training mean. — [arXiv 2605.09386](https://arxiv.org/pdf/2605.09386)
- Diff-TTS and StyleTTS 2 both keep a deterministic regression duration predictor (log domain with L1 loss on MFA durations in Diff-TTS; L1 on monotonic-alignment durations in StyleTTS 2). Even prominent diffusion acoustic models did not move duration to a generative model. — [Diff-TTS, arXiv 2104.01409](https://arxiv.org/pdf/2104.01409); [StyleTTS 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC11759097/)

### Inferences
- Single-sample stochastic durations add variance around the conditional mean. Against one reference, the expected per-utterance correlation of a sample is at most that of the conditional mean (the deterministic MSE target). Reports of FM/SDP gains are on MOS/CMOS, FDD or diversity, consistent with this. For a read-speech, ρ-to-reference objective, a generative duration model is unlikely to help unless it is evaluated by its mean or mode, or by averaging many samples.
- A generative predictor could still help as a **better-trained mean estimator**, e.g., DurFormer's mean+variance (Gaussian NLL) head, which reports lower MSE than FS2. A heteroscedastic Gaussian or Laplace output head, read out at its mean, is a low-risk architectural change to try within Matcha.
- In Mehta et al., Matcha-TTS read speech was the "no significant difference" case. Since the Hakka corpus is presumably read speech, expect little from swapping in FM durations.

### Gaps
- No full-text table values (MOS/WER per corpus) from Mehta et al. 2024 could be retrieved. arXiv/ISCA were blocked.
- No source found that reports **Pearson/Spearman ρ of phoneme durations vs reference** for VITS SDP, FM, diffusion (e.g., DurIAN-E, ProDiff) or MaskGIT duration models against a deterministic baseline. NaturalSpeech 2/3 and E2/F5-TTS duration-accuracy metrics were not found or verified in this session.
- AR duration (Tacotron attention, VALL-E implicit): no quantitative duration-correlation comparisons found.

## Q2. Speech SSL (HuBERT, WavLM, wav2vec 2.0, XLS-R) in forced alignment, unit-based duration, prosody pseudo-labels and semi-supervised TTS: any measured duration gains?

### Takeaway
SSL-based aligners trained or fine-tuned on in-domain dialect data can clearly beat off-the-shelf aligners on boundary accuracy. In Chengdu Mandarin (a low-resource Chinese variety, a close analogue to Hakka), an XLS-R frame-classification aligner reached 82.2% of phone boundaries within 50 ms vs 52.8% for the Charsiu-Mandarin baseline. MFA trained on the dialect beat Mandarin-MFA beyond 10 ms. I found no paper that directly measures how better SSL alignment changes TTS duration-predictor ρ. Semi-supervised TTS with HuBERT units (UnitSpeech, QS-TTS) is documented but without duration-accuracy numbers.

### Cited Findings
- **Chengdu Mandarin forced alignment (2026):** the Chengdu-FC-xlsr model (XLS-R frame classification) aligned **82.2%** of phone-tier boundaries within 50 ms vs **52.8%** for the Charsiu-Mandarin-FC baseline. The Mandarin-MFA model had the most predictions within 10 ms, but the dialect-trained **Chengdu-MFA did better at every tolerance above 10 ms**. — [arXiv 2607.21332, "Phonetic forced alignment for low-resource language varieties: Chengdu Mandarin"](https://arxiv.org/pdf/2607.21332)
- Charsiu is a wav2vec2-based phonetic aligner (frame classification, 10 ms frame-rate model) that can recognize phones and do forced alignment. — [Charsiu GitHub](https://github.com/lingjzhu/charsiu/); [Zhu et al., "Phone-to-audio alignment without text: a semi-supervised approach", ICASSP 2022](https://ar5iv.labs.arxiv.org/html/2110.03876)
- TIPAA-SSL (text-independent phone-to-audio alignment with SSL plus knowledge transfer) outperforms Charsiu on TIMIT phoneme-recognition metrics, but the authors note the r-value deteriorates. — [arXiv 2405.02124](https://arxiv.org/pdf/2405.02124)
- FastSpeech 2 reports that its MFA alignments have a mean absolute boundary difference of **12.47 ms**, vs **19.68 ms** for the teacher-model (attention) alignments used in FastSpeech, which motivated switching duration targets to MFA. — [FastSpeech 2, arXiv 2006.04558](https://arxiv.org/pdf/2006.04558) (number from search excerpt)
- Newer SSL alignment work, not verified in detail: multilingual word-level forced alignment with SSL representations plus learned DP ([arXiv 2606.10675](https://arxiv.org/pdf/2606.10675)); fully differentiable neural forced alignment via soft DP ([arXiv 2606.25460](https://arxiv.org/pdf/2606.25460)); SSL-based phone/word metrics for evaluating forced alignment ([arXiv 2608.28508](https://arxiv.org/pdf/2608.28508)).
- **UnitSpeech (Interspeech 2023):** adapts a diffusion TTS with minimal untranscribed speech. HuBERT features are k-means-clustered into units that serve as a pseudo-transcript, the unit encoder replaces the text encoder, and unit durations are modeled. — [UnitSpeech, Interspeech 2023](https://www.isca-archive.org/interspeech_2023/kim23k_interspeech.pdf)
- **QS-TTS:** semi-supervised TTS using HuBERT k-means (512 codewords) pseudo-labels **with durations** to pre-train the acoustic model. A variant trains a CTC ASR on HuBERT features to get phonemes and durations from alignments. — [QS-TTS, arXiv 2309.00126](https://arxiv.org/pdf/2309.00126)
- Unit-based unsupervised TTS is "limited by the quality of the acoustic units, which can be quite noisy." — [Unsupervised TTS via unsupervised ASR, arXiv 2203.15796](https://arxiv.org/pdf/2203.15796)
- Low-resource SSL with SSL-enhanced TTS: HuBERT pre-trained on limited unpaired speech is used to build multi-speaker TTS that generates synthetic data for a better HuBERT. This targets SSL quality, not duration. — [arXiv 2309.17020](https://arxiv.org/pdf/2309.17020)
- Semi-supervised low-resource TTS with a reference model uses teacher pseudo-labels; it has a separate duration predictor but no duration-error numbers were found. — [arXiv 2210.14723](https://arxiv.org/pdf/2210.14723)

### Inferences
- With ~1 h of Hakka, measured ρ is bounded by **label noise in the MFA durations** as well as by predictor capacity. If MFA boundaries jitter by ~10–20 ms on 30–100 ms phones, the noise ceiling on achievable ρ is well below 1. The Chengdu result suggests boundary accuracy for a Chinese dialect can vary by about 30 percentage points (at 50 ms) depending on aligner choice and in-dialect training. Re-aligning with a dialect-adapted aligner (MFA re-trained on Hakka, or a fine-tuned XLS-R/MMS frame classifier) and checking agreement on a small hand-labeled set is a cheap test of how much of the ρ gap is label noise.
- Evaluating ρ against labels from the same aligner rewards consistency, not correctness. A cleaner aligner may raise ρ mostly by reducing random boundary noise in both training targets and evaluation references.
- HuBERT-unit pseudo-durations from untranscribed Hakka audio could pre-train a duration predictor (QS-TTS-style), but their effect on phoneme-duration ρ is unmeasured in the literature found.

### Gaps
- No paper found that reports a TTS **duration-predictor ρ/RMSE change** caused by switching from MFA to an SSL aligner (Charsiu, MMS, wav2vec2-CTC).
- MMS forced-aligner phone-level accuracy for Hakka or Hakka-like varieties: not found.
- Virtuoso, SpeechT5 and similar semi-supervised TTS: no duration-accuracy metrics verified.

## Q3. Reference/prompt-conditioned duration (speech prompts, utterance-level speaking-rate or style embeddings, phone-level prosody latents): how much do they raise duration correlation?

### Takeaway
Conditioning the duration predictor on an utterance-level embedding extracted from the **target utterance** raised duration concordance (CCC) from 0.771 to 0.804 and cut MAE from 2.045 to 1.869 frames. Same-speaker embeddings gave 0.801. Random-noise conditioning lowered CCC to 0.733. This is about +0.03 CCC, roughly the size of the gap between the user's 1 h (0.74) and 48 h (0.77–0.80) systems. Zero-shot prompt-conditioned duration (TDA, DurFormer, MaskGCT-style ratio rules) is mainly shown to help intelligibility and robustness, with no ρ numbers found. Global speaking-rate/total-duration conditioning likely explains a meaningful share of per-utterance duration variance when an oracle is available.

### Cited Findings
- arXiv 2406.19243 ("Application of ASV for Voice Identification after VC and Duration Predictor Improvement in TTS Models") evaluates a phoneme duration predictor with MAE, RMSE, weighted duration error rate (WDER) and concordance correlation coefficient (CCC):

  | Duration-predictor input | MAE ↓ | RMSE ↓ | CCC ↑ |
  |---|---|---|---|
  | Phonemes only | 2.045 | 3.836 | 0.771 |
  | Phonemes + random noise | 2.198 | 4.366 | 0.733 |
  | Phonemes + same-speaker ASV embedding | 1.870 | 3.607 | 0.801 |
  | Phonemes + same-utterance ASV embedding | 1.869 | 3.503 | 0.804 |

  — [arXiv 2406.19243](https://arxiv.org/html/2406.19243); [secondary review](https://www.themoonlight.io/en/review/application-of-asv-for-voice-identification-after-vc-and-duration-predictor-improvement-in-tts-models). Numbers are from search excerpts of the arXiv HTML; verify against the table. Units are presumably frames.
- Total-Duration-Aware modeling conditions phoneme durations on the target total duration, with zero-shot evaluation on LibriSpeech using a 3 s prompt whose phoneme durations condition the model. It reports better intelligibility and speaker similarity across speech-rate configurations. — [Eskimez et al. 2024, arXiv 2406.04281](https://arxiv.org/abs/2406.04281)
- MobileSpeech trains a prompt-duration extractor alongside the duration predictor (both MSE loss, MFA targets). — [arXiv 2402.09378](https://arxiv.org/pdf/2402.09378)
- DurFormer claims improved robustness to prompt vs input audio mismatch in zero-shot TTS, and about a 5% WER reduction when replacing F5-TTS's ratio-scale total-duration estimate (per secondary review). — [arXiv 2507.22612](https://arxiv.org/html/2507.22612v2); [Pith review](https://pith.science/paper/2507.22612)
- Multilingual prosody transfer (supervised vs transfer learning) is studied in [arXiv 2406.00022](https://arxiv.org/pdf/2406.00022). No duration-correlation numbers were verified.

### Inferences
- An "oracle total duration / speaking rate" experiment is a quick diagnostic: rescale the predicted durations of each test utterance so their sum matches the reference total, then recompute ρ. Pooled phoneme-level ρ is computed across utterances, so a substantial part of the remaining error can be utterance-level speaking-rate mismatch. Note that per-utterance Pearson ρ is invariant to a global scale factor, so this only helps if ρ is pooled across utterances or if RMSE is the target metric. The user should state which ρ definition is used.
- With a predicted rather than oracle global embedding, gains are smaller. In 2406.19243, same-speaker embeddings (not utterance-specific) already captured almost all the gain (0.801 vs 0.804). In a single-speaker 1 h corpus, a speaker embedding adds nothing, so gains would have to come from utterance-level style predicted from text, e.g., sentence type or punctuation context.

### Gaps
- No source found quantifying ρ with oracle vs text-predicted phone-level prosody latents (e.g., the NaturalSpeech prosody encoder, DelightfulTTS phoneme/utterance prosody, prosody-transfer systems).
- Zero-shot prompt TTS papers report WER and similarity, not phoneme-duration correlation to ground truth.

## Q4. Multi-speaker/multilingual pretraining, speed-perturbation augmentation and teacher pseudo-labeling for low-resource duration

### Takeaway
Cross-lingual or multi-speaker pretraining and speed/pitch augmentation are standard low-resource TTS recipes with reported naturalness gains. **No source found reports duration-error or ρ before and after** these interventions. The user's own data point (1 h ρ≈0.74 vs 48 h ≈0.77–0.80) suggests data scale alone gives only about +0.03–0.06.

### Cited Findings
- Cross-lingual transfer: pretraining on high-resource English or Japanese, then fine-tuning on the target low-resource language; "multi-speaker models were found to be more effective for constructing a single-speaker, low-resource TTS model." The study also uses data augmentation. — [EURASIP J. Audio Speech Music Proc. 2021, "TTS for low-resource language using cross-lingual transfer learning and data augmentation"](https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-021-00225-4)
- Augmentation scale reported in the low-resource TTS literature: noise and speed perturbation used to grow a target low-resource dataset 10-fold; SoX pitch/speed alteration turning 30 min of target data into 13 h of augmented audio. No duration metrics reported. — [search excerpts from the EURASIP 2021 paper and APSIPA 2021 multi-speaker low-resource TTS](https://www.apsipa.org/proceedings/2021/pdfs/0000849.pdf)
- A multilingual training strategy for low-resource TTS uses a length regulator with a phoneme duration predictor; no duration-error results found. — [arXiv 2409.01217](https://arxiv.org/pdf/2409.01217)
- Tone learning in low-resource bilingual TTS (relevant to tonal Hakka): [Liu et al., Interspeech 2020](https://www.isca-archive.org/interspeech_2020/liu20n_interspeech.pdf). Not read in full; no duration numbers verified.
- Teacher pseudo-labels: semi-supervised low-resource TTS based on a reference model uses pseudo-labels from a pre-trained reference model to guide fine-tuning. — [arXiv 2210.14723](https://arxiv.org/pdf/2210.14723)

### Inferences
- Speed perturbation uniformly rescales all phone durations in an utterance. It teaches a global-rate shift but adds no new **relative** (within-utterance) duration patterns, which are what a correlation metric rewards. It may help robustness but is unlikely to raise ρ much; it could even blur the rate distribution unless tagged with a rate condition.
- Related-language multi-speaker data (e.g., other Hakka sub-dialects, Mandarin, Hokkien) for duration-predictor pretraining is the most plausible data-side lever, since duration patterns (phrase-final lengthening, tone-related lengthening) partly transfer. The user's 48 h ceiling of about 0.77–0.80 implies that reaching 0.85–0.9 likely also needs better targets (Q2) and/or richer conditioning (Q3), not just more data.

### Gaps
- No paper found with phoneme-duration RMSE/ρ ablations for speed perturbation, multilingual pretraining or teacher-TTS pseudo-durations in low-resource settings.
- No published inter-annotator or alignment-noise ceiling for phoneme-duration ρ was found. This would bound what 0.8–0.9 means in practice.
