# Empirical ceiling of phone/syllable duration prediction accuracy, and how forced-alignment error limits it

> Research context: Hakka TTS (Matcha-TTS, FastSpeech-style conv duration predictor). Pearson ρ between predicted and real vowel-nucleus durations: ~0.74 with 1 h of data, ~0.77–0.80 with 48 h; a text-feature gradient-boosting model gives ~0.74. Both reference and synthesized durations are measured with MFA. The question is whether ρ 0.8–0.9 is attainable.
>
> **Method caveat (read first):** In this sandbox arxiv.org, isca-archive.org, PMC and Semantic Scholar were blocked by the network proxy, and WebFetch could not resolve hosts. So **every number below comes from web-search result excerpts of the cited papers, not from reading the full PDFs.** Numbers that a search excerpt quoted verbatim from a table are marked "(verbatim excerpt)". Numbers seen only in a secondary or aggregator citation are marked "(secondary)". Check against the original PDFs before putting them in a publication.

---

## Q1. What correlations and RMSE/MAE do papers report for phone/syllable duration prediction?

### Takeaway
Published Pearson r for duration prediction from text against held-out natural speech (forced-aligned or hand-labelled) usually falls between **~0.55 and ~0.82** for neural and statistical TTS duration models. Values of **~0.88–0.94** show up mainly when (a) all phones are pooled, consonants included, so phone identity alone explains much of the variance, (b) small, controlled, single-speaker lab corpora are used, or (c) the target is utterance or segment total duration under reference/prompt conditioning, not free text-to-duration prediction. Modern FastSpeech/FastSpeech2/VITS papers almost never report a duration correlation. They report MSE/MAE in log-frames, or nothing at all.

### Cited Findings
- **Ronanki et al. 2016 (Interspeech, "Median-based generation of synthetic speech durations…")**: phone-level **LSTM r = 0.765, RMSE 7.789 frames**; phone-level **DNN r = 0.750, RMSE 8.037 frames**; frame-level LSTM variants were slightly lower. RMSE/MAE are in frames per phone (5 ms frames are typical for Merlin-style systems, so ≈39 ms, but the frame shift was not confirmed in the excerpt). Single-speaker English. — [arXiv 1608.06134](https://arxiv.org/pdf/1608.06134); [poster](https://people.kth.se/~ghe/pubs/present/ronanki2016median-poster.pdf)
- **DNN-based TTS for Indian languages (2016)**: duration DNN scored by RMSE and Pearson r against forced-aligned durations. Best (G2P) variant: **r = 0.818 Telugu, 0.657 Tamil, 0.564 Hindi** (search excerpt did not confirm the exact table pairing, so treat as approximate). Shows a large spread across languages and corpora with the same method. — [arXiv 1608.05374](https://arxiv.org/pdf/1608.05374)
- **Serbian tree-based phone duration model (Elektronika ir Elektrotechnika, 2014)**: regression tree over the **full phone set (vowels + consonants)**, **RMSE 14.89 ms, r = 0.8796**. The authors call this "comparable with those reported in the literature for Czech, Greek, Lithuanian, Korean, Hindi, Telugu, Turkish." — [EEJournal 4090](https://eejournal.ktu.lt/index.php/elt/article/view/4090)
- **Mandarin multiple-regression duration models**: **syllable duration r = 0.668, RMSE 43.1 ms**; initial-consonant model r = 0.889, RMSE 18.6 ms; a third model r = 0.621, RMSE 36.9 ms (verbatim excerpt: "multiple correlation coefficients … 0.889, 0.621 and 0.668 … RMS errors 18.6, 36.9 and 43.1 ms"). This is directly relevant to the Hakka syllable/nucleus case: the syllable- and final-level models sit at r ≈ 0.62–0.67, and the high r comes from the consonant (initial) model. — search excerpt from [Learning syllable duration and intonation of Mandarin Chinese (ResearchGate)](https://www.researchgate.net/publication/221487447_Learning_syllable_duration_and_intonation_of_Mandarin_Chinese) / [Modeling durations of syllables using neural networks (CSL)](https://www.sciencedirect.com/science/article/abs/pii/S0885230806000234). Which of these two papers the excerpt came from was not confirmed.
- **Korean vowels**: **RMSE 32.13 ms, r = 0.68**, i.e. vowel-only, the same kind of target as the Hakka nucleus measure. **TIMIT with sums-of-products models**: **RMSE 9 ms, r = 0.94** (secondary: both appeared in the same search summary as "non-Mandarin reference points"; original papers not identified). — via search over [Mandarin syllable-duration literature summary](https://www.sciencedirect.com/science/article/abs/pii/S0885230806000234) (secondary)
- **Mandarin hierarchical TTS prosody model**: reports the share of syllables within 10% relative error: **37% (inside test), 43% (outside test)**, which shows how hard syllable-duration prediction is. — [JCIE 2005](https://www.tandfonline.com/doi/abs/10.1080/02533839.2005.9671006)
- **Hierarchical prosody coding (Mandarin, 2018)**: syllable-duration RMSE ≈ **4.8 ms** (speaker-dependent) and 9.3 ms in another condition. This is *coding/quantization* error with access to the real speech, not prediction from text, but it shows what an "oracle prosody" representation can reach. — [EURASIP JASMP 2018](https://link.springer.com/article/10.1186/s13636-018-0129-5)
- **van Santen 1994 sums-of-products duration model** (Computer Speech & Language 8(2):95–128): the classic rule/statistical model. The often-quoted r ≈ 0.9+ for single-speaker vowel durations could **not** be verified from the excerpts. — [CSL 1994](https://www.sciencedirect.com/science/article/abs/pii/S0885230884710059)
- **FastSpeech 2** trains its duration predictor with MSE on **log-domain MFA durations** and reports no duration correlation. — [arXiv 2006.04558](https://arxiv.org/pdf/2006.04558)
- **PAMA-TTS** reports MAE between predicted and forced-aligner phoneme durations, and **FluentSpeech** reports word-level duration MSE against MFA on VCTK. Neither reports a correlation. — [arXiv 2110.04486](https://arxiv.org/pdf/2110.04486); [arXiv 2305.13612](https://arxiv.org/pdf/2305.13612)
- **T5Gemma-TTS technical report (2026)**: Pearson **r ≈ 0.92** between generated and target durations on JSUT. This is a reference-audio/target-duration setting with a different model, so it is not comparable to free prediction from text. — [arXiv 2604.01760](https://arxiv.org/html/2604.01760v1)

### Inferences
- The Hakka results (ρ ≈ 0.74 at 1 h, 0.77–0.80 at 48 h, vowel nucleus only) are **already at or above** most published vowel- or syllable-level figures: Korean vowels 0.68, Mandarin syllables 0.67, English phone-level LSTM 0.765. The r ≈ 0.88–0.94 figures come from pooled phone sets that include consonants, where intrinsic phone-identity differences inflate between-unit variance, or from controlled lab corpora.
- Restricting to vowel nuclei removes the "easy" variance from phone identity (consonant vs vowel, stop closure vs fricative). An r of ~0.8 on nuclei is roughly comparable to ~0.9 on all phones.
- GBM ≈ 0.74 ≈ the 1 h neural model, and only a small gain at 48 h. Together these suggest the text-predictable variance is mostly captured, and the rest is free prosodic variation plus measurement noise (see Q2/Q3).

### Gaps
- I found no FastSpeech/FastSpeech2/VITS/Matcha paper that reports Pearson r of per-phone duration against MFA ground truth on a standard corpus (LJSpeech, VCTK, AISHELL-3, etc.).
- Exact dataset sizes for Ronanki 2016 and the Indian-language DNN paper were not retrievable (full text blocked).
- The original sources of the Korean (r = 0.68) and TIMIT (r = 0.94) figures were not identified.

---

## Q2. Human/natural variability ceiling: how consistent are segment durations across repeated readings by the same speaker?

### Takeaway
I found **no study that directly reports a correlation between segment durations across repeated readings of the same sentences by one speaker**, which is the true "test-retest" ceiling. Indirect evidence says within-speaker token-to-token duration variability is substantial, and cross-speaker duration correlation on the same text is only r ≈ 0.66. A deterministic text→duration model cannot exceed the repeat-reading correlation, and that correlation is plausibly well below 1.0 for vowel nuclei in connected speech.

### Cited Findings
- **Cross-speaker**: segment durations from two speakers reading the same corpus correlated at **r = 0.66** (Toda 2010, secondary citation). — cited in search results around [Speaker-specific variability of phoneme durations (van Heerden & Barnard)](https://repository.up.ac.za/items/518c5f0d-65bf-4def-8df7-b916ba3b5fc3/full) (secondary)
- **van Heerden & Barnard (2008)**: phoneme durations covary across speakers, most strongly within the same broad phonetic class. — [UP repository](https://repository.up.ac.za/items/518c5f0d-65bf-4def-8df7-b916ba3b5fc3/full)
- **van den Heuvel et al. (Eurospeech 1991)**: 10 speakers × 24 CVCa nonsense words × **10 replications**. **Long vowels showed more speaker variability than short segments**, sonorant consonants less, and individual speaking rate mattered. — [ISCA Eurospeech 1991](https://www.isca-archive.org/eurospeech_1991/heuvel91_eurospeech.html)
- **2025 JASA Express Letters, "Formant variability is related to vowel duration across speakers"**: reanalysing repeated isolated-word datasets, it finds that **"the duration of a single vowel produced by a single speaker can vary substantially even when repeated in the same context."** — [JASA EL 5(11):115202](https://pubs.aip.org/asa/jel/article/5/11/115202/3371042/Formant-variability-is-related-to-vowel-duration); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13003517/)
- **Smith (ICSLP 2002)**: durational variability across **ten readings of one passage by one American English speaker**. The correlation values were not in the excerpt. — [ISCA ICSLP 2002](https://www.isca-archive.org/icslp_2002/smith02_icslp.html)
- **Audibert et al. (Interspeech 2022)**: intra-speaker phonetic variation across many recording sessions of read speech. They note that within-speaker variability is rarely studied. — [ISCA Interspeech 2022](https://www.isca-archive.org/interspeech_2022/audibert22_interspeech.pdf)

### Inferences
- If the reference speaker re-read the same Hakka sentences, the correlation of vowel-nucleus durations between the two readings, measured with MFA, would set the **practical ceiling**. Given substantial token-to-token variation and r ≈ 0.66 across speakers, a within-speaker repeat correlation of roughly 0.8–0.9 is plausible but **unverified**. If it is ~0.85, then ρ = 0.80 already captures most of the attainable correlation.
- Long vowels vary the most (van den Heuvel 1991). Nuclei that carry phrase-final lengthening or emphasis are where a deterministic predictor, which outputs the conditional mean, will diverge most from any single realization.

### Gaps
- No published number for the same-speaker, same-sentence, repeated-reading correlation of phone/syllable durations was found. **Recommended experiment:** record 50–100 Hakka sentences twice (or use existing duplicates in the corpus), align both with MFA, and compute ρ between the takes. This directly measures the ceiling for this speaker and pipeline.

---

## Q3. Forced-alignment boundary accuracy, and how alignment error attenuates duration correlation

### Takeaway
On English benchmark corpora, MFA places about **48% of phone boundaries within 10 ms, ~76% within 20 ms, ~84% within 25 ms and ~95% within 50 ms** (TIMIT). Mean absolute phone-boundary error is roughly **12–20 ms**, depending on MFA version and corpus. That is close to human inter-annotator agreement, but MFA has more medium-to-large errors (>25 ms). MFA still beats WhisperX and MMS for alignment (Interspeech 2024). Because a duration is the difference of two noisy boundaries, and both the reference and the synthesized durations are MFA-measured, measurement noise alone can plausibly cap the observable ρ at roughly 0.8–0.9 for short units such as vowel nuclei, by classical attenuation.

### Cited Findings — boundary accuracy
- **MAPS paper (Kelley et al., Laboratory Phonology 2024)**, MFA baseline on the **TIMIT test set**, by tolerance <10/<20/<25/<50/<100 ms: **47.96% / 75.84% / 84.14% / 94.97% / 98.56%** (verbatim excerpt of Table 5). The authors note that "<20 ms counts as accurate" is the common binary criterion. MFA had fewer large errors than the neural MAPS model, while MAPS was better at the low-error end. — [arXiv 2310.15425](https://arxiv.org/pdf/2310.15425); [Lab Phonology / De Gruyter](https://www.degruyterbrill.com/document/doi/10.1515/phon-2024-0015/html?lang=en)
- **McAuliffe et al. 2017 (MFA, Interspeech)**: word-boundary mean manual/aligned difference **24 ms**, against **26 ms inter-transcriber** agreement on Buckeye. Phone and word alignment are "comparable to human annotators on average," but there are **more medium-to-large errors (>25 ms)**. MFA beat FAVE and Prosodylab-Aligner. A secondary citation gives Buckeye phone-boundary displacement of **17 ms (MFA) vs 19.3 ms (FAVE)**. A later secondary citation of the original evaluation gives Buckeye phone boundaries at **77% within 25 ms, 93% within 50 ms**. — [ISCA Interspeech 2017](https://www.isca-archive.org/interspeech_2017/mcauliffe17_interspeech.pdf); [ResearchGate](https://www.researchgate.net/publication/319185277_Montreal_Forced_Aligner_Trainable_Text-Speech_Alignment_Using_Kaldi)
- **MFA 3.x (2026 MFA paper, "MFA and the state of speech-to-text alignment in 2026")**: mean boundary error dropped from **16.4 → 12.1 ms on TIMIT** and **17.6 → 13.9 ms on Buckeye** (ARPA phone set), MFA 1.0 → 3.0. A 2026 word-level benchmark gives MFA English ARPA mean **19.1 ms / median 12.0 ms**, **68.2% within 20 ms** on TIMIT, and Buckeye held-out mean 21.5 ms. — [arXiv 2606.18466](https://arxiv.org/pdf/2606.18466) (numbers via search excerpts); [MFA blog update](https://memcauliffe.com/update-on-montreal-forced-aligner-performance.html)
- **Rousso, Cohen, Keshet, Chodroff, "Tradition or Innovation" (Interspeech 2024)**: on TIMIT and Buckeye, **MFA outperformed WhisperX and MMS** on alignment. The reason given is the HMM's 10 ms temporal resolution, against modern ASR systems that operate over longer stretches. The neural systems had lower WER. — [arXiv 2406.19363](https://arxiv.org/pdf/2406.19363); [ISCA DOI](https://api.openalex.org/works/doi:10.21437%2FINTERSPEECH.2024-429)
- **Older HMM aligners on TIMIT**: **80–89% of boundaries within 20 ms** of manual segmentation. A boundary-model system reached **96.8% within 20 ms**. — [Microsoft ICASSP 2014](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/icassp2014-phoneseg-v10-fixed.pdf)
- **Charsiu (wav2vec2 forced alignment, Zhu et al.)**: mean phone alignment error **≈17.8 ms on TIMIT** (benchmark listing, secondary). The original wav2vec2 frame rate (~20 ms) is "too coarse," so they doubled it. — [arXiv 2110.03876](https://arxiv.org/pdf/2110.03876); [listing (secondary)](https://www.wizwand.com/paper/68fa6bb61e209131cea2fc93)
- **Bournemouth Forced Aligner (2025)** cites MFA at **72.3% (TIMIT) and 60.6% (Buckeye) of phone boundaries within 20 ms** (secondary, unverified). Its own README claims most boundaries are within ±30 ms. — [arXiv 2509.23147](https://arxiv.org/html/2509.23147v1); [GitHub](https://github.com/tabahi/bournemouth-forced-aligner)
- **Gonzalez, Grama & Travis 2020 (Linguistics Vanguard)**, Australian English: **MFA and LaBB-CAT best**, in some cases not significantly different from humans, then FAVE, then MAUS. Aligners are **less accurate on boundaries after a vowel than before one**, which bears directly on vowel-nucleus offsets. A secondary citation gives an MFA mean onset difference of 28 ms. — [ANU repository](https://researchportalplus.anu.edu.au/en/publications/comparing-the-performance-of-forced-aligners-used-in-sociophoneti/); [Speech Communication 2024 L2 aligner study](https://www.sciencedirect.com/science/article/pii/S0167639324000141)
- **Gradient boundaries via ensembles (2025)** argues that forced-alignment boundaries carry real uncertainty and proposes confidence intervals from model ensembles. — [arXiv 2506.01256](https://arxiv.org/pdf/2506.01256)

### Cited Findings — alignment on synthesized speech and in TTS pipelines
- **No peer-reviewed study was found that measures MFA boundary accuracy on TTS output against manual labels.**
- **"Evaluating Speech–Phoneme Alignment and its Impact on Neural TTS" (ICASSP 2023)**: compared five aligners. Errors **below ~75 ms did not reduce synthesis quality**. One model's median error was **7.5 ms on TIMIT but 70.1 ms on TC-STAR** because of longer utterances, so aligner accuracy depends strongly on the corpus. — [ResearchGate](https://www.researchgate.net/publication/371288151_Evaluating_Speech-Phoneme_Alignment_and_its_Impact_on_Neural_Text-To-Speech_Synthesis)
- **Aligner-dependent duration bias**: CTC-alignment-based durations had "more inconsistent shape" than HMM-GMM ones, with generally shorter spoken phonemes. — [arXiv 2310.08132](https://arxiv.org/pdf/2310.08132)
- **Aligner-Guided Training (2024)** describes MFA as the de facto source of "ground-truth" TTS durations. That reliability is assumed rather than tested. — [arXiv 2412.08112](https://arxiv.org/pdf/2412.08112)

### Inferences — attenuation arithmetic (my reasoning, not from a source)
- **Classical attenuation (Spearman):** observed r_xy = true r × √(rel_x · rel_y), where rel = var_true / (var_true + var_error). Both the Hakka reference durations *and* the synthesized durations are MFA-measured, so **both sides are attenuated**.
- **Duration error from boundary error:** a nucleus duration = offset − onset. If each boundary has error SD σ_b and the two are independent, σ_dur = √2·σ_b. Mean absolute boundary error ≈ 12–20 ms implies σ_b ≈ 15–25 ms for a roughly Laplacian/normal distribution. That gives **σ_dur ≈ 20–35 ms**. Positively correlated errors on adjacent boundaries, such as a systematic shift, cancel in the difference and lower this. Frame quantization (MFA 10 ms frames) adds only ≈ 4 ms (√2 × 10/√12).
- **Worked example:** assume the true SD of Hakka vowel-nucleus durations is 50 ms.
  - σ_dur = 20 ms → rel ≈ 0.86. If both sides are like this, the ceiling factor is 0.86, so a perfect predictor would show ρ ≈ 0.86.
  - σ_dur = 30 ms → rel ≈ 0.74 → ceiling ≈ 0.74 if the synthesized audio is aligned equally badly.
  - Synthesized speech is cleaner and more canonical, so MFA error there is probably smaller (rel_syn ≈ 0.9+). The ceiling would then be ≈ √(0.74–0.86 × 0.9) ≈ **0.82–0.88**.
- **Implication:** an observed ρ ≈ 0.77–0.80 may correspond to a "true" ρ of ≈ 0.88–0.95 after correcting for attenuation. **Measurement noise alone could make observed ρ > 0.85–0.9 nearly unreachable on vowel nuclei**, regardless of model quality. This should be checked empirically: (a) hand-correct the boundaries of ~200 nuclei and recompute ρ; (b) estimate MFA test-retest by aligning the same audio with different acoustic models or beam settings, or by aligning two TTS renders with identical durations (for example, the same durations through different vocoders/noise seeds) to get rel_syn directly; (c) **better still, read the synthesized durations straight from the model's duration predictor (the Matcha/FastSpeech expanded durations) instead of re-aligning the synthesized audio with MFA.** That removes one entire attenuation factor.
- Vowel-offset boundaries are the least accurate (Gonzalez et al. 2020), so vowel-nucleus durations are probably *more* noise-affected than an average phone.
- MFA accuracy figures are for English with well-trained models. A Hakka MFA model trained on 1–48 h of data, possibly with a G2P or lexicon mismatch, plausibly has larger boundary errors than the TIMIT/Buckeye figures (unverified).

### Gaps
- No direct measurement of MFA boundary accuracy on Hakka, on Sinitic tonal languages in general, or on TTS-synthesized speech was found.
- The full McAuliffe 2017 Table 2 values and Rousso 2024 per-tolerance numbers could not be read (ISCA/arXiv blocked here).
- No paper was found that explicitly applies correction for attenuation to TTS duration-correlation metrics.

---

## Q4. Are there reports of ρ 0.8–0.9 for duration prediction, and under what conditions?

### Takeaway
Yes, but mostly under favorable conditions. r ≈ **0.82 (Telugu DNN)**, **0.88 (Serbian tree, all phones, 14.9 ms RMSE)**, **0.889 (Mandarin initials)**, **0.94 (TIMIT sums-of-products, secondary)** and **0.92 (T5Gemma-TTS, target/reference durations)**. Under conditions like the Hakka setup (vowel or syllable nuclei only, read connected speech, free text→duration prediction), reported values cluster around **0.62–0.77**. ρ 0.8 on nuclei is good, and 0.9 is probably not attainable without changing the evaluation unit, the measurement, or the conditioning.

### Cited Findings
- 0.8796 r, RMSE 14.89 ms: Serbian, **all phones**, regression tree. — [EEJournal](https://eejournal.ktu.lt/index.php/elt/article/view/4090)
- 0.818 r: Telugu DNN, against forced-aligned durations, same method only 0.564 for Hindi. — [arXiv 1608.05374](https://arxiv.org/pdf/1608.05374)
- 0.889 r for Mandarin **initial consonants** vs **0.668 for syllables**, same study. — [ResearchGate](https://www.researchgate.net/publication/221487447_Learning_syllable_duration_and_intonation_of_Mandarin_Chinese)
- 0.94 r, RMSE 9 ms: TIMIT sums-of-products (secondary, original not identified) vs 0.68 r, RMSE 32 ms for Korean vowels. — via search summary (secondary)
- 0.765 r: English phone-level LSTM (Ronanki 2016). — [arXiv 1608.06134](https://arxiv.org/pdf/1608.06134)
- ≈0.92 r: T5Gemma-TTS generated vs target durations (JSUT, reference/target-conditioned). — [arXiv 2604.01760](https://arxiv.org/html/2604.01760v1)
- Hierarchical prosody coding reaches syllable-duration RMSE ~4.8 ms when coding from real speech, an "oracle prosody" upper bound. — [EURASIP 2018](https://link.springer.com/article/10.1186/s13636-018-0129-5)

### Inferences
- **Conditions under which 0.8–0.9 is realistic:**
  1. Evaluate over **all phones, or initials + finals** rather than nuclei only. Phone-identity variance alone lifts r into the high 0.8s.
  2. Use **model-internal predicted durations** instead of MFA re-alignment of synthesized audio, which removes one attenuation factor (Q3).
  3. Use **cleaner reference durations**: manually corrected boundaries, or an MFA model adapted to the speaker with a good lexicon.
  4. Add **oracle or reference prosody conditioning**: phrase-break labels, prominence/emphasis, speaking-rate tokens, or a reference-audio/prompt embedding. Much residual nucleus-duration variance is phrase-final lengthening and speaker-chosen rate, which text alone does not determine.
  5. **Aggregate units**: syllable or word duration including consonants, or utterance-normalized z-scores computed per utterance, which removes rate variation across utterances.
- **Conditions under which 0.8–0.9 is unrealistic:** a deterministic predictor from text only, vowel nuclei only, MFA on both sides, and a single-take reference. Here 0.77–0.80 is likely near the measurement-plus-variability ceiling, and 0.9 would probably mean overfitting or leakage.
- Saturation from 1 h (0.74) to 48 h (~0.78) is consistent with a ceiling set by unpredictable prosodic variation and measurement noise, not by model capacity.

### Gaps
- No recent (2020–2026) neural TTS paper was found that reports per-phone duration Pearson r on a standard corpus, so there is no direct modern benchmark against which to place 0.77–0.80.
- No Hakka- or Taiwanese-specific duration-modelling correlation numbers were found.
