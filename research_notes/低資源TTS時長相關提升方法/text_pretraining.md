# Text-side / linguistic methods for improving duration (and prosody) prediction in TTS

Scope note: the target is a ~1 h Hakka Matcha-TTS model with a conv duration predictor (log-MSE on MFA durations, cross-dialect transfer init), with duration Pearson/Spearman rho ~0.74 against MFA (48 h model: ~0.77-0.80).

Method caveat: the researcher's WebFetch/curl access was blocked in this session (DNS failure for arxiv.org, isca-archive.org, pmc.ncbi.nlm.nih.gov and others). Every number below comes from search-engine extracts of the primary papers or project pages, not from reading full PDFs. Exact table values I could not confirm are listed under Gaps, not filled in from memory.

## Q1. Phoneme/text self-supervised pretraining (PnG BERT, MP-BERT, PL-BERT, XPhoneBERT, Dict-TTS, CLAPSpeech): do they improve duration accuracy, and how much data do they need?

### Takeaway
Almost none of the phoneme-BERT papers report a duration-specific metric. They report naturalness MOS/CMOS, with gains of about +0.05 to +0.30 CMOS (or MOS) at full data. The one large low-resource result is XPhoneBERT: with 5% of the training data, MOS rose 2.88→3.22 (English) and 1.59→3.35 (Vietnamese). Prosody-targeted text pretraining that does report duration (Dict-TTS, CLAPSpeech) shows small absolute duration-error reductions (for example 35.7→34.4 ms on Biaobei vs. a BERT-embedding baseline). There is no direct evidence that any of these moves duration correlation by 0.05-0.15.

### Cited Findings
- **PnG BERT (Jia et al., Interspeech 2021, Google):** a BERT encoder that takes both phonemes and graphemes plus their word-level alignment. It is pretrained with MLM only on a large text corpus, then fine-tuned as the TTS encoder. It gives "more natural prosody and more accurate pronunciation" than a phoneme-only, non-pretrained baseline, and raters showed no statistically significant preference between PnG BERT TTS and professional recordings. The search extracts did not show the MOS values or the size of the pretraining corpus. — [PnG BERT, arXiv 2103.15060](https://arxiv.org/abs/2103.15060); [Google Research page](https://research.google/pubs/png-bert-augmented-bert-on-phonemes-and-graphemes-for-neural-tts/)
- **PnG BERT limitation:** it works only over a fixed set of grapheme tokens and can fail for words unseen in training, according to the PL-BERT authors. — [PL-BERT, arXiv 2301.08810](https://arxiv.org/abs/2301.08810)
- **Japanese PnG BERT (Yasuda & Toda, IEEE JSTSP 2022):** beat a baseline Tacotron on accent correctness in a listening test. No duration figures were found. — [arXiv 2212.08321](https://arxiv.org/abs/2212.08321)
- **Mixed-Phoneme BERT (Zhang et al., Interspeech 2022, Microsoft):** CMOS +0.30 over a FastSpeech 2 baseline. Against PnG BERT the difference is only +0.03 CMOS ("on par or slightly better"), and the paper treats CMOS >0.05 as significant. Mel generation is about 3x faster than PnG BERT (RTF 1.32e-2 vs 3.99e-2). No duration-error metric appeared in the extracts. — [arXiv 2203.17190](https://arxiv.org/abs/2203.17190); [arXiv HTML v1](https://arxiv.org/html/2203.17190v1); [project page](https://speechresearch.github.io/mpbert/)
- **PL-BERT (Li et al., ICASSP 2023; the text encoder used in StyleTTS 2):** a 12-layer ALBERT-style encoder trained on phonemes only, predicting masked phonemes and the corresponding graphemes. Downstream TTS used LJSpeech (~24 h, 12,500 training samples). In-distribution MOS was 4.24 with PL-BERT vs 4.19 for StyleTTS, against 4.34 for ground truth. Out-of-distribution MOS was 3.64 vs 3.49, and CMOS was +0.16 vs an MP-BERT variant. No duration metric was reported in the extracts, and there was no experiment on low-resource or reduced data. The authors stress that the gains appear mainly on OOD text. — [arXiv 2301.08810](https://arxiv.org/abs/2301.08810); [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10417533/); [GitHub](https://github.com/yl4579/PL-BERT)
- **XPhoneBERT (Nguyen et al., Interspeech 2023, VinAI):** BERT-base trained with the RoBERTa recipe on about 330M phoneme-level sentences from about 100 languages and locales, then used as the input phoneme encoder for VITS. **Limited-data setting (5% of the training clips):** MOS rose 2.88→3.22 (English) and 1.59→3.35 (Vietnamese). With full data the gain was about +0.14 (EN) and +0.15 (VI). Duration-specific metrics were not found in the extracts. — [Interspeech 2023 PDF](https://www.isca-archive.org/interspeech_2023/thenguyen23_interspeech.pdf); [VinAI page](https://research.vinai.io/xphonebert-a-pre-trained-multilingual-model-for-phoneme-representations-for-text-to-speech/); [GitHub](https://github.com/VinAIResearch/XPhoneBERT)
- A later low-resource language-adaptation study reports that XPhoneBERT "helps boost the performance of low-resource languages." — [arXiv 2406.08911](https://arxiv.org/html/2406.08911)
- **Dict-TTS (Jiang et al., 2022; Biaobei Mandarin, PortaSpeech backbone):** reports character-level duration error, defined as the MSE of character-level durations from MFA. Dict-TTS scored 34.4 vs 35.7 for the "BERT Embedding" baseline (about 3.6% relative), and average pitch error was 1232.3 vs 1312.1. The paper calls these "significantly lower than the baseline systems." The per-number values come from a third-party leaderboard that copies Table 6. — [Dict-TTS, arXiv 2206.02147](https://arxiv.org/abs/2206.02147); [SOTA2 leaderboard copy](https://www.sota2.com/research/sota/prosodic-realization-analysis-on-biaobei)
- **CLAPSpeech (Ye et al., ACL 2023):** contrastive text-speech pretraining of a text encoder so that it learns prosody from context. Duration is measured as average absolute duration error (DE) and pitch as DTW distance. It beats BERT and A3T representations on MOS, pitch accuracy and duration accuracy on LJSpeech, Biaobei and a multi-speaker English set. A third-party review gives LJSpeech MOS of 4.09 (CLAPSpeech) vs 3.82 (BERT) vs 3.85 (A3T). The DE values were not visible in the extracts. — [arXiv 2305.10763](https://arxiv.org/abs/2305.10763); [ACL 2023](https://aclanthology.org/2023.acl-long.518.pdf); [secondary review](https://liner.com/review/clapspeech-learning-prosody-from-text-context-with-contrastive-languageaudio-pretraining)

### Inferences
- For a ~1 h target, the strongest text-SSL evidence is XPhoneBERT's 5%-data result, since its multilingual phoneme pretraining can cover an unseen language through IPA. That result is MOS, not duration correlation, so the effect on rho is unmeasured. A plausible way to test it is to swap or augment the Matcha text encoder (or just the duration predictor's input) with frozen or fine-tuned XPhoneBERT features computed on Hakka IPA.
- Where duration error is reported (Dict-TTS, CLAPSpeech), the gains over a plain BERT-embedding baseline are only a few percent relative. That suggests text-SSL alone is unlikely to move rho from 0.74 to 0.85+. A gain of about 0.02-0.05 is a more realistic expectation, and it is a guess, not a measured number.
- PL-BERT and MP-BERT need large monolingual phoneme corpora (English Wikipedia scale). For Hakka, a PL-BERT-style model would need a large Hakka or Hakka+Mandarin+Hokkien phonemized text corpus. XPhoneBERT, or continued pretraining of it on the available Hakka text, is the cheaper route.
- Gains concentrate on OOD and long, complex sentences (PL-BERT OOD, 2111.07549). Measure rho on held-out sentences that differ structurally from training, or improvements may be hidden.

### Gaps
- No phoneme-BERT paper found reports **duration correlation**. The duration MSE/MAE ablation tables of PnG BERT, MP-BERT, PL-BERT and XPhoneBERT, if they exist, could not be read because full-text fetching was blocked.
- Pretraining corpus sizes for PnG BERT, PL-BERT and MP-BERT were not confirmed in this session. Check the papers directly.
- TaCL-BERT and "Phoneme-level BERT for enhanced prosody" (which is the PL-BERT title) produced no separate TTS duration results.
- The audio duration that XPhoneBERT's "5% of training data" corresponds to (roughly 1.2 h if the base was LJSpeech's 24 h) is not confirmed.

## Q2. Pretrained LMs (BERT, mBERT, LLMs) and semantic/syntactic features (POS, phrase structure, word segmentation, dependency parse) as duration-predictor inputs

### Takeaway
BERT and word-level LM features consistently give modest but significant prosody MOS gains, and gains grow roughly logarithmically with LM size. Syntax-graph inputs (SyntaSpeech) are built specifically to feed the duration predictor. However, extracted evidence on duration-specific numbers is thin: most papers report MOS/CMOS or pitch metrics.

### Cited Findings
- **Hayashi et al. (Interspeech 2019):** BERT embeddings of input phrases added as an extra input to Tacotron 2 on LJSpeech (24 h). MOS improved "modestly but significantly" over the no-embedding baseline, and context helped most at the subword level. — [ISCA archive](https://www.isca-archive.org/interspeech_2019/hayashi19_interspeech.html)
- **Kenter, Sharma & Clark (Interspeech 2020, Google):** adding a BERT model pretrained on unlabeled text and fine-tuned to the speech domain to an RNN TTS improves prosody. Fine-tuning matters, and an ablation on LM size was run for F0 prediction. Numeric values were not obtained. — [ISCA archive](https://www.isca-archive.org/interspeech_2020/kenter20_interspeech.html)
- **Granero-Moya et al. (SSW 2023):** compared 15 PLMs for prosody prediction and 12 for pause prediction. Prosody quality grows **logarithmically with model size**, and expressive prosody is much harder than neutral. For pause prediction, BERT-tiny and BERT-mini lag well behind, while ELECTRA-small through RoBERTa-large differ only slightly. Results correlate strongly with GLUE scores. — [arXiv 2309.01576](https://arxiv.org/html/2309.01576v1); [ISCA SSW](https://www.isca-archive.org/ssw_2023/moya23_ssw.html)
- **Li et al. 2021, "Improving Prosody for Unseen Texts..." (Mandarin):** a BERT front-end fine-tuned jointly on polyphone disambiguation, word segmentation + POS, and prosody-structure prediction, combined with a FastSpeech 2 model pretrained on large noisy data. Both components improve prosody, with the largest gains on structurally complex sentences. The duration predictor is trained on MFA durations with MAE loss. The extracts did not show numeric duration MAE. — [arXiv 2111.07549](https://arxiv.org/abs/2111.07549)
- **SyntaSpeech (Ye et al., IJCAI 2022):** a dependency-tree graph encoder is fed into the duration predictor (word-level log-duration MSE) and the variational generator of PortaSpeech. The extracts showed only learning curves of duration loss on LJSpeech, and a low-quality secondary summary claims faster convergence of the duration predictor. — [IJCAI 2022](https://www.ijcai.org/proceedings/2022/0620.pdf); [arXiv 2204.11792](https://ar5iv.labs.arxiv.org/html/2204.11792)
- **GraphSpeech (2020):** syntax-aware graph attention using Stanza dependency parses. Its RMSE figures are pitch (Hz), not duration. — [arXiv 2010.12423](https://arxiv.org/pdf/2010.12423)
- **GST-BERT-TTS (Interspeech 2025, Japanese):** BERT predicts character-level log-F0, energy and duration without accent labels. No numbers were extracted. — [ISCA 2025](https://www.isca-archive.org/interspeech_2025/ogura25_interspeech.pdf)
- **MSStyleTTS (Mandarin):** pretrained BERT feeds a multi-scale (including paragraph-level) style predictor. No duration RMSE was extracted. — [arXiv 2307.16012](https://arxiv.org/pdf/2307.16012)
- **BreezyVoice (Taiwanese Mandarin, 2025):** conditions phoneme prediction on BERT outputs through learnable softmax layer weights, for polyphone disambiguation rather than duration. — [arXiv 2501.17790](https://arxiv.org/html/2501.17790v1)

### Inferences
- For Hakka, no good monolingual Hakka LM is likely to exist. The options are a multilingual or Chinese LM (Hakka in Han characters shares much vocabulary with Mandarin, though grammar and particles differ), or an LLM used to produce word segmentation and phrase boundaries. Upsampling word-level features to phonemes and concatenating them into the duration predictor input is the common recipe (Cauliflow-style, 2111.07549).
- The duration-relevant part of these features is most likely word-boundary and phrase-final information (final lengthening, pauses). A cheap first experiment is to add word-boundary and position-in-word/phrase indicators, which are deterministic given segmentation, before trying full BERT embeddings.

### Gaps
- No found paper reports duration Pearson/Spearman correlation with vs. without BERT features. Numeric duration MAE/RMSE for 2111.07549, SyntaSpeech and Kenter 2020 could not be extracted.
- No study was found that applies a PLM to duration for Hakka, Hokkien or Cantonese. The only Hakka-specific prosody study found is an older CART/Bayesian break-prediction paper (best precision 80.17% with CART). — [IEEE 6016704](https://ieeexplore.ieee.org/document/6016704)

## Q3. Explicit linguistic/prosodic features (phrase breaks, ToBI, syllable position, stress/tone) vs. plain phoneme input

### Takeaway
Evidence gathered here is indirect. Phrase-break and pause prediction from PLMs is well studied, and pause placement is a major driver of duration. Quantified duration gains of explicit features over plain phonemes in neural duration models were not found in this session.

### Cited Findings
- Pause prediction is framed as a per-word binary task and evaluated with F0.5, because a missed pause is less harmful than a wrong one. ELECTRA-small is already near the best. — [arXiv 2309.01576](https://arxiv.org/html/2309.01576v1)
- A study of pause duration in modern TTS systems found that held-out R² of pause duration is only 0.14 (Parler-TTS), 0.30 (SpeechT5) and 0.44 (Tacotron 2), which shows how hard pause length is to predict. This is a secondary extract and the original paper was not identified with certainty. — via search extract associated with [arXiv 2309.01576 results](https://arxiv.org/html/2309.01576v1) (treat with caution)
- Speaker-conditioned phrase-break prediction uses a phoneme-level PLM (PL-BERT-style). — [arXiv 2509.00675](https://arxiv.org/pdf/2509.00675)
- Prosody labeling with Phoneme-BERT plus speech foundation models (2025). — [arXiv 2507.03912](https://arxiv.org/pdf/2507.03912)
- Automatic prosody annotation with a pretrained text-speech model. — [arXiv 2206.07956](https://arxiv.org/pdf/2206.07956)
- "Duration-aware pause insertion using a pre-trained language model for multi-speaker TTS." — [ResearchGate](https://www.researchgate.net/publication/368843089_Duration-aware_pause_insertion_using_pre-trained_language_model_for_multi-speaker_text-to-speech)

### Inferences
- In a Sinitic tonal language, tone identity, tone-sandhi context, syllable position in word/phrase and phrase-final position are classic strong duration predictors. If the Hakka phoneme input lacks explicit tone tokens, word-boundary markers or punctuation/phrase-break tokens, adding them is the lowest-cost lever.
- MFA-derived pauses and silences strongly affect rho. Check whether rho is computed with or without silence/pause tokens. Pauses have high variance and low R², so they can dominate correlation.
- Speech-side prosody-label prediction models (2507.03912, 2206.07956) could auto-label phrase breaks on the 1 h data and on any larger untranscribed Hakka audio, then be fed as features. Gains are unquantified.

### Gaps
- A quantified ablation of "phonemes + explicit break/position/tone features vs phonemes only" for a FastSpeech-style duration predictor was not retrieved. ToBI-feature neural duration results were not found.

## Q4. Knowledge distillation / teacher models / pseudo-labels for duration

### Takeaway
Teacher-derived durations are a standard technique (FastSpeech). In low-resource expressive TTS, an explicit duration model with a 15-minute target matched a 10-hour Tacotron 2 model. Semi-supervised pseudo-labeling with a reference model exists for low-resource TTS. No study was found that quantifies duration-correlation gains from LLM-TTS-generated duration pseudo-labels.

### Cited Findings
- **FastSpeech:** duration targets are extracted from an autoregressive teacher's attention and trained with MSE on log durations. — [arXiv 1905.09263](https://arxiv.org/pdf/1905.09263); [Microsoft blog](https://www.microsoft.com/en-us/research/blog/fastspeech-new-text-to-speech-model-improves-on-speed-accuracy-and-controllability/)
- **Shah et al. (SSW 2021, Amazon), "Non-Autoregressive TTS with Explicit Duration Modelling for Low-Resource Highly Expressive Speech":** replacing attention with an external duration model plus cGAN fine-tuning closes the gap to recordings by 23.3% (naturalness) and 16.3% (speaker similarity). With **15 minutes** of target data it matches a full-data (~10 h) Tacotron 2 model, and with 30 min or more it is significantly better. With 3 h it beats a full-data model of similar architecture. — [arXiv 2106.12896](https://arxiv.org/abs/2106.12896); [ISCA SSW](https://www.isca-archive.org/ssw_2021/shah21_ssw.html)
- **Semi-supervised learning based on a reference model for low-resource TTS (2022):** a FastSpeech 2-style model with pseudo-labels and knowledge distillation (index terms). How the duration targets are produced was not visible. — [arXiv 2210.14723](https://arxiv.org/pdf/2210.14723)
- **DMOSpeech 2 (AAAI):** reinforcement learning for duration prediction in a distilled TTS model. Details were not extracted. — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/40450/44411)
- **ComedicSpeech:** an expressive TTS for low-resource scenarios that uses prosody representations. — [arXiv 2305.12200](https://arxiv.org/pdf/2305.12200)

### Inferences
- For Hakka, a practical distillation route is to (a) take a large multilingual or LLM-based TTS (CosyVoice-class, F5-TTS) that can read the text, or Mandarin/Hokkien-trained models, (b) synthesize many Hakka (or closely related) sentences, (c) align them with MFA, and (d) pretrain the duration predictor on these pseudo-durations before fine-tuning on the 1 h real data. Whether teacher prosody matches native Hakka timing is the key risk. Unmeasured.
- A cheaper semi-supervised variant is to run MFA (or a CTC aligner) on untranscribed but ASR-transcribable Hakka audio, such as the Hakka MOE corpora, to multiply duration training data for the duration predictor only. This directly targets the data-size gap between the 1 h and 48 h models (rho 0.74 vs 0.77-0.80).
- The 48 h model reaching only ~0.77-0.80 suggests a ceiling set by intrinsic duration variability and MFA label noise rather than data alone. Reaching 0.85-0.9 with text features alone is probably unrealistic. Context (sentence-level, speaking rate), better alignments, and evaluating correlation without pauses may matter as much. This is an inference; no source gives a duration-correlation ceiling.

### Gaps
- No paper found quantifies duration RMSE or correlation for LLM-TTS pseudo-label distillation into a small student.
- How much the cross-dialect transfer init already captures of what a text-SSL encoder would add is unknown. No study isolates the two.
