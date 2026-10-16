# Duration Modeling in TTS for Tonal and Low-Resource Languages (Mandarin, Cantonese, Taigi, Hakka, Vietnamese, Thai)

> Method note: in this session the egress proxy blocked direct page fetches from arxiv.org, isca-archive.org and researchgate.net (`connect_rejected` / DNS failure). All numbers below come from search-engine extracts of the cited papers, not from reading the full text. The figures were quoted consistently across extracts, but the report writer should mark them "as reported in the abstract or extract". Anything I could not confirm is listed under Gaps.

## Q1. Which tonal-language TTS / duration-modeling papers report duration RMSE or correlation, and what are the numbers?

### Takeaway
Classic statistical work on tonal languages reports predicted-vs-observed duration correlations of about 0.62–0.79 at the vowel/final or syllable level. Only consonant initials reach about 0.89. The user's vowel-nucleus ρ≈0.74 therefore already matches the published range for vowel/final-level duration. Reaching ρ 0.8–0.9 on nuclei would exceed what these classic papers report. That is more plausible on held-in data, with phrase-boundary features, or with a different unit/metric (syllable, log-duration, or excluding pauses and phrase-final lengthening).

### Cited Findings
- **Mandarin (ISCSLP 2000, multiple regression, "A study of phoneme and syllable duration characteristics of Mandarin Chinese"):** RMS error of 18.6 / 36.9 / 43.1 ms and correlation of **0.889 / 0.621 / 0.668** for Initial / Final / Syllable. The following syllable contributes more to duration than the preceding one. — [ISCA archive, zhu00b_iscslp](https://www.isca-archive.org/iscslp_2000/zhu00b_iscslp.pdf)
- **Mandarin neural-network syllable duration:** test-set syllable RMSE of **47.1 ms** (reported in search extract for "Learning syllable duration and intonation of Mandarin Chinese"). — [ResearchGate 221487447](https://www.researchgate.net/publication/221487447_Learning_syllable_duration_and_intonation_of_Mandarin_Chinese)
- **Mandarin hierarchical prosodic model (HPM, Taiwan) with embedded English spelling (Springer IJST 2010):** reconstructed duration RMSE of **8.93 ms (train) / 9.00 ms (test)**. This is an analysis-by-synthesis *reconstruction* error, not open text-to-duration prediction, so it is not comparable to TTS prediction RMSE. — [Springer s10772-010-9067-z](https://link.springer.com/article/10.1007/s10772-010-9067-z)
- **Mandarin–English HMM bilingual TTS:** phone-duration RMSE of about **38–52 ms** across voices. Errors concentrate at language-switch points. — [ResearchGate 224363995](https://www.researchgate.net/publication/224363995_HMM-Based_Mixed-Language_Mandarin-English_Speech_Synthesis)
- **Mandarin HMM (HTS) context:** state durations are modeled as Gaussian, conditioned on Initial/Final identity and prosodic tags. — [ResearchGate 220758474](https://www.researchgate.net/publication/220758474_An_HMM-Based_Mandarin_Chinese_Text-To-Speech_System)
- **Mandarin decision-tree duration (Guo & Katae, Fujitsu, Speech Prosody 2006):** units are Initials and Finals. Phrase boundary and phone context are the main factors for Final duration. The extract did not include the numeric metrics. — [ISCA archive, guo06_speechprosody](https://www.isca-archive.org/speechprosody_2006/guo06_speechprosody.pdf)
- **Thai segmental duration (Hayashi's categorized linear regression with tone, phone identity, neighbours, word/phrase position and POS):** correlation **0.788**, RMSE **33.14 ms**. Error was "fairly high" but consistent on unseen data. — [ResearchGate 228943663](https://www.researchgate.net/publication/228943663_Analysis_of_Segmental_Duration_for_Thai_Speech_Synthesis)
- **Thai syllable duration (Hansakunbuntheung et al., Eurospeech 2003, 2,480 words):** total correlation **0.73** on the test set. A related extract gives about **0.79** on train and test. Vowel-length category is the dominant factor; tone has a moderate effect, strongest for the falling tone. — [ISCA archive, Eurospeech 2003](https://www.isca-archive.org/eurospeech_2003/hansakunbuntheung03_eurospeech.pdf); [ResearchGate 221490621](https://www.researchgate.net/publication/221490621_Analysis_and_modeling_of_syllable_duration_for_Thai_speech_synthesis)
- **Thai multi-model neural duration (2014):** syllables were grouped by vowel articulatory features, including vowel length, with one feed-forward network per group. Average prediction error dropped by **23.21%** relative. — [ACL Anthology W14-5106](https://aclanthology.org/W14-5106.pdf)
- **Vietnamese (VnVoice, CART, 1 h single-speaker story-reading corpus):** the main syllable-duration factors are tone type, number of phones, position in phrase and depth of the right syllable boundary. Phone-level and syllable-level CART models were built. Correlations were **lower at syllable level than at phone level**. A listening test showed no significant MOS difference, and the perceived gap between synthetic and natural durations was 0.7 on a 5-point scale. Exact ρ values were not in the extract. — [ResearchGate 228943672](https://www.researchgate.net/publication/228943672_Duration_Modeling_in_a_Vietnamese_Text-to-Speech_System)
- **Vietnamese (Fujisaki/VnVoice line):** syllable duration depends "much more on tone than position". Some tones are consistently shorter because they end in stops or need a glottal stop. This is the Vietnamese analogue of checked tones. — [ResearchGate 221478949](https://www.researchgate.net/publication/221478949_Fujisaki_model_based_F0_contours_in_vietnamese_TTS)
- **Vietnamese HMM TTS (Hanoi):** spectrum, F0 and phone duration are modeled jointly in HMMs, with decision-tree contexts that include tone type. — [ResearchGate 289650181](https://www.researchgate.net/publication/289650181_HMM-based_TTS_for_hanoi_Vietnamese_Issues_in_design_and_evaluation)
- **Cantonese:** a Stem-ML F0 study (79,528 syllables) gives a mean syllable duration of **0.192 s**. It states that the entering vs non-entering tone difference "is primarily duration", and modeling entering tones separately did not help F0 (RMS F0 error 10.5–11.9 Hz). — [ResearchGate 221488070](https://www.researchgate.net/publication/221488070_Modeling_Tones_In_Continuous_Cantonese_Speech)
- **Mandarin word-level (2026, embeddings → spoken duration of monosyllabic words):** mean correlation **0.557** empirical, **0.399** on held-out data. This is spontaneous-speech word duration, so it is a much harder target than read-speech TTS. — [arXiv 2607.02002](https://arxiv.org/abs/2607.02002)
- **Cross-language reference (singing, Sinsy):** phoneme-duration RMSE of 13.32 frames (decision tree) vs 11.23 frames (DNN). — [arXiv 2108.02776](https://arxiv.org/pdf/2108.02776)

### Inferences
- The ranking holds across languages: consonant/initial duration is easy (ρ≈0.89 in Mandarin), while vowel/final and syllable duration are hard (ρ≈0.62–0.79). The user's 0.74 on vowel nuclei is mid-range for classic tonal-language results, which were trained on single-speaker corpora larger than 1 h.
- Correlation is very sensitive to the population being pooled. Including checked (short) and unchecked (long) nuclei in one pool inflates ρ through the large between-class difference. Within-class ρ (checked only, unchecked only) is likely much lower and is the more diagnostic number. Report both.
- Most correlations above 0.8 in this literature come from initials, or from reconstruction rather than open prediction (e.g., HPM 9 ms RMSE). An open-test vowel-nucleus ρ of 0.9 would exceed any tonal-language figure found.

### Gaps
- No modern neural (FastSpeech2/VITS/Matcha) paper was found that reports phone-level duration Pearson ρ for Mandarin, Cantonese, Vietnamese or Thai with tone ablations. Neural papers mostly report MOS/CER, not duration ρ.
- The Guo & Katae 2006 metrics, the VnVoice exact ρ, and Cantonese HMM duration RMSE could not be retrieved, because full text was blocked.
- No figures were found on inter-utterance natural variability (the human "ceiling" for duration ρ when the same speaker repeats the same sentence) for any tonal language.

## Q2. Which techniques helped duration in tonal languages (tone features, prosodic boundaries, BERT, syllable-level modeling, sandhi)?

### Takeaway
The most consistent duration drivers in tonal-language studies are **prosodic boundary level (especially the right-hand boundary)**, **position in phrase**, **tone category**, **vowel-length class / syllable structure**, and **right-context syllable**. Pretrained-LM front ends (BERT) help mainly by predicting boundaries better. Syllable-level duration or speaking-rate modeling fits Chinese better than phoneme-level.

### Cited Findings
- **Mandarin:** whether the right phrase-boundary level is below the prosodic-word level is "the most important determinant of finals duration". Prosodic-word-final syllables are lengthened. — [ISCA archive, guo06_speechprosody](https://www.isca-archive.org/speechprosody_2006/guo06_speechprosody.pdf); [ISCA archive, pin04_iscslp](https://www.isca-archive.org/iscslp_2004/pin04_iscslp.pdf)
- **Mandarin (ISCSLP 2004, integrated prosodic model):** syllable duration depends on both syllable constitution and "upper layer prosodic structures". — [ISCA archive, pin04_iscslp](https://www.isca-archive.org/iscslp_2004/pin04_iscslp.pdf)
- **Mandarin:** the following syllable's contribution to duration is significantly larger than the preceding syllable's. — [ISCA archive, zhu00b_iscslp](https://www.isca-archive.org/iscslp_2000/zhu00b_iscslp.pdf)
- **Vietnamese:** tone type, number of phones, phrase position and **depth of right syllable boundary** are the key predictors. — [ResearchGate 228943672](https://www.researchgate.net/publication/228943672_Duration_Modeling_in_a_Vietnamese_Text-to-Speech_System)
- **Thai:** vowel-length category is the most dominant factor and tone is moderate. Splitting models by vowel class reduced error by 23.21%. — [ResearchGate 221490621](https://www.researchgate.net/publication/221490621_Analysis_and_modeling_of_syllable_duration_for_Thai_speech_synthesis); [ACL Anthology W14-5106](https://aclanthology.org/W14-5106.pdf)
- **Taiwan HPM (Chiang, NTPU / NCTU line):** a hierarchical prosodic model ties syllable duration, pitch, energy and pause to inferred prosodic structure (break types and prosodic states) and linguistic features. It also models speaking rate explicitly. — [EURASIP JASMP 2018](https://link.springer.com/article/10.1186/s13636-018-0129-5); [ISCA archive, Interspeech 2012 hsieh12](https://www.isca-archive.org/interspeech_2012/hsieh12_interspeech.pdf)
- **BERT front ends for Mandarin:** a Chinese BERT fine-tuned multi-task for prosodic boundary prediction, distilled with TinyBERT to about 25% of benchmark pipeline size while staying competitive. — [arXiv 2012.15404](https://arxiv.org/pdf/2012.15404)
- **Sogou Blizzard 2020:** a BERT-LSTM predicts two-level prosodic boundaries that feed a FastSpeech-based model. — [ISCA archive, meng20_vccbc](https://www.isca-archive.org/vccbc_2020/meng20_vccbc.pdf)
- **Pretraining the duration predictor on noisy external data:** the FastSpeech2 encoder and duration predictor were pretrained on noisy AISHELL-1 (ASR data) and then fine-tuned. The authors report better prosody on unseen texts. — [arXiv 2111.07549](https://arxiv.org/pdf/2111.07549)
- **PL-BERT (phoneme-level BERT):** a phoneme-level LM with grapheme-prediction pretraining, reported to improve TTS prosody. — [arXiv 2301.08810](https://arxiv.org/pdf/2301.08810)
- **Cross-Lingual F5-TTS:** the text-length-ratio duration heuristic breaks across languages. Speaking-rate predictors at phoneme, syllable and word level were trained, and the **syllable-level** predictor worked best for Chinese (phoneme-level best for English). — [arXiv 2509.14579](https://arxiv.org/pdf/2509.14579)
- **Accented TTS:** the FastSpeech2 duration predictor takes an accent-conditioned phoneme embedding instead of the plain encoder output, an analogue of dialect- or tone-conditioned duration. — [arXiv 2209.10804](https://arxiv.org/pdf/2209.10804)
- **Duration labels:** a neural TTS survey notes that how duration labels are obtained "is critical for the prosody and quality". — [arXiv 2106.15561](https://arxiv.org/pdf/2106.15561)
- **Tone sandhi:** recent Taigi ASR work (T-SANDHI) adds explicit sandhi-aware modules for low-resource Taiwanese Hokkien. That is evidence that sandhi tone, not citation tone, is the right conditioning variable, although it was shown for ASR, not TTS duration. — [arXiv 2609.18194](https://arxiv.org/pdf/2609.18194)

### Inferences
- For a 1 h Hakka model, the strongest cheap features are likely: (a) explicit prosodic-boundary / punctuation / word-boundary tokens (right-boundary depth), (b) a syllable-structure class (checked -b/-d/-g vs open vs nasal-coda) as a duration-predictor input, (c) right-neighbour syllable features, and (d) surface (post-sandhi) tone instead of citation tone.
- A two-level design is consistent with Mandarin, Thai and Vietnamese findings and with Cross-Lingual F5-TTS: predict syllable duration first, then distribute it to initial, nucleus and coda. Syllable-level targets are also less noisy than MFA phone boundaries inside checked syllables, where the vowel–stop boundary is ambiguous.
- Thai's 23% error reduction from vowel-class-specific models suggests separate duration-predictor heads, or mixture components, for checked vs unchecked rhymes may help.

### Gaps
- No paper was found that quantifies a duration ρ/RMSE gain from adding BERT features or tone embeddings in a neural Mandarin, Cantonese or Vietnamese duration predictor. The evidence is qualitative (MOS or prosody preference).
- No duration study was found that treats tone sandhi (Taigi/Hakka) as an explicit duration factor.

## Q3. What Hakka and Taiwanese Hokkien TTS work exists (duration results, corpora, text resources, alignment tools)?

### Takeaway
No Hakka or Taigi TTS paper was found that reports duration accuracy. Hakka TTS work (2006 RNN prosody, NTUST Hailu thesis, VoxHakka 2024) reports MOS/intelligibility at most. Usable resources are VoxHakka's about 180 h of web-crawled Hakka speech, the HAT-Vol1/Vol2 ASR corpora (about 80 h each, including Sixian, Hailu and Nan-Sixian), and TAT/TAT-TTS for Taigi.

### Cited Findings
- **VoxHakka (2024):** multi-speaker, dialect-diverse Taiwanese Hakka TTS. **180.53 h** of transcribed Hakka speech were collected online, with radio as the largest source. The per-dialect tables list about 65.15 h Sixian / 54.77 h Hailu in one table and 51.01 h / 43.76 h in another; the difference is likely before vs after filtering, but I could not confirm this. Sixian and Hailu account for 57.5% and 35.8% of speakers. — [arXiv 2409.01548](https://arxiv.org/html/2409.01548v1)
- **Formospeech HF model "taiwanese-hakka-tts-sixian-hailu-mix":** YourTTS at 16 kHz, 4 speakers, learnable speaker and language (dialect) embeddings. — [Hugging Face](https://huggingface.co/formospeech/taiwanese-hakka-tts-sixian-hailu-mix)
- **Early Hakka TTS (2006):** 671 base syllables as synthesis units, with an **RNN-based prosody generator** (syllable duration, pitch, energy). — [ACM DL](http://dl.acm.org/citation.cfm?id=2127170)
- **NTUST master's thesis 客語語音合成之初步研究 (Hailu):** Hailu has 7 tones and 736 syllables. It used rule-based syllable-length and amplitude generation, and listening tests showed the prosody rules were "not well tuned". — [臺灣博碩士論文知識加值系統](https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi?o=dnclcdr&s=id%3D%22090NTUST392001%22.&searchmode=basic)
- **Hakka tone modeling with the command-response (Fujisaki) model:** entering (Ru) tones end with an unreleased stop and are shorter than the contrasting tones. — [ResearchGate 221483434](https://www.researchgate.net/publication/221483434_Modeling_tones_in_hakka_on_the_basis_of_the_command-response_model)
- **HAT-Vol1 (FSR-2023 Hakka ASR challenge):** collected across Taiwan in 2022, about 100 speakers, about 80 h, with Hanzi and Pinyin tracks. — [NYCU IAIS](https://iais.nycu.edu.tw/en/news_detail43.html); [FSW challenge 2023](https://sites.google.com/nycu.edu.tw/fsw/home/challenge-2023). One participant reports 59.43 h train plus 10.01 h pilot test, and about 86.5 h after adding Hakka dictionary and Hakka Affairs Council (HAC) material. — [arXiv 2310.03443](https://arxiv.org/html/2310.03443). Taiwan AI Labs used 3-way speed perturbation for 156.3 h. — [ROCLING 2023.53](https://aclanthology.org/2023.rocling-1.53.pdf)
- **HAT-Vol2 (FSR-2025):** about 100 speakers, about 80 h. — [FSW challenge 2025](https://sites.google.com/nycu.edu.tw/fsw/home/challenge-2025?authuser=0%EF%BC%89%E3%80%82). The FSW page lists HAT Phase I as "2*150*8 hours", which is inconsistent with about 80 h. — [FSW](https://sites.google.com/nycu.edu.tw/fsw)
- **Dialect-aware Hakka ASR (Peng, Huang, Lo, Lee, Wang, Chen; LREC 2026):** the HAT corpus includes Hailu, Sixian and Nan-Sixian. A dialect-balanced 60-speaker subset has 82.32 h (73.91 h train). Dialect-aware RNN-T conditioning reduced relative error by 57.00% (Hanzi) and 40.41% (Pinyin). — [arXiv 2602.22522](https://arxiv.org/abs/2602.22522)
- **ROCLING 2023:** Hakka dialects differ not only in phone inventories but also in tone pitch and **syllable-length prosody**. — [ROCLING 2023.54](https://aclanthology.org/2023.rocling-1.54.pdf)
- **ROCLING 2025:** Hakka phrase-final lengthening and colloquial prosody cause ASR errors such as spurious particles. — [ROCLING 2025.58](https://aclanthology.org/2025.rocling-main.58.pdf)
- **Taigi corpora:** TAT has about 300 h from 600 speakers. TAT-TTS has 2 male and 2 female speakers at about 10 h each. — [FSW TAT](https://sites.google.com/speech.ntut.edu.tw/fsw/home/tat-corpus); [FSW TAT-TTS](https://sites.google.com/nycu.edu.tw/fsw/home/tat-tts-corpus)
- **Taigi TTS:** a Tacotron2-based Taiwanese voice-conversion and TTS system was trained on about 140 h from 204 speakers (IJCLCLP 2022). — [ACL Anthology 2022.ijclclp-2.6](https://aclanthology.org/2022.ijclclp-2.6.pdf)
- **Breeze Taigi (2026):** a Taigi TTS fine-tuned on about 10,000 h of *synthetic* Taigi speech, evaluated with ASR metrics and human pronunciation and naturalness ratings. No duration metrics were reported. — [arXiv 2603.19259](https://arxiv.org/html/2603.19259)
- **Self-supervised models on TAT:** evaluation of SSL speech models on Taiwanese Hokkien. — [arXiv 2312.06668](https://arxiv.org/abs/2312.06668)

### Inferences
- HAT-Vol1/Vol2 (about 160 h, multi-speaker, Sixian and Hailu labeled) plus VoxHakka's crawl are the obvious sources for (a) training a better Hakka MFA acoustic model, so the target speaker's alignments are less noisy, and (b) pretraining a multi-speaker duration predictor that is then fine-tuned on the 1 h target. Both address duration-label noise, which probably limits ρ more than model capacity does.
- Hakka dictionary and HAC text resources (Pinyin-annotated) can supply the word-boundary / prosodic-word labels found important in Q2.

### Gaps
- No public Hakka-specific MFA acoustic model or pronunciation dictionary was found in this search. Whether VoxHakka or Formospeech release alignments was not verified.
- No Hakka or Taigi TTS paper reports duration RMSE or ρ.
- No search was done for an MOE Hakka dictionary API or downloadable lexicon (time budget).

## Q4. What has low-resource (<2 h) TTS shown for cross-lingual transfer, multilingual pretraining, phone mapping and augmentation, especially for duration?

### Takeaway
Low-resource TTS papers show that cross-lingual pretraining plus symbol mapping and data augmentation can make 15–30 min of target data usable. They evaluate with MOS/CER, not duration ρ. Duration-specific evidence is limited to the finding that duration/speaking-rate heuristics transfer poorly across languages and that syllable-level rate predictors suit Chinese.

### Cited Findings
- **Tu et al., Interspeech 2019 (arXiv 1904.06508):** a Phonetic Transformation Network trained with CTC maps source to target symbols. About **15 min** of paired target data gave a "relatively good" system. — [arXiv 1904.06508](https://arxiv.org/abs/1904.06508); [ISCA archive, chen19f](https://www.isca-archive.org/interspeech_2019/chen19f_interspeech.pdf)
- **Cross-lingual transfer plus augmentation (2021):** 30 min of target data, pretrained on English (24 h) and Japanese (10 h). About 3 h of *augmented* target data approached a 12 h baseline. — [ResearchGate 356778163](https://www.researchgate.net/publication/356778163_Text-to-speech_system_for_low-resource_language_using_cross-lingual_transfer_learning_and_data_augmentation)
- **Few-shot cross-lingual TTS with transferable phoneme embeddings.** — [arXiv 2206.15427](https://arxiv.org/pdf/2206.15427)
- **DSE-TTS:** simulated low-resource German and Spanish with a few hours each, using MFA phoneme durations as ground truth. — [arXiv 2306.14145](https://arxiv.org/pdf/2306.14145)
- **Cross-Lingual F5-TTS:** language-specific speaking-rate predictors are needed, and the syllable level is best for Chinese. — [arXiv 2509.14579](https://arxiv.org/pdf/2509.14579)
- **Meta MMS-TTS (e.g., Vietnamese `facebook/mms-tts-vie`):** VITS with a stochastic duration predictor. No tone- or duration-accuracy metrics are published per language. — [Dataloop MMS-TTS-vie](https://dataloop.ai/library/model/facebook_mms-tts-vie/)
- **YourTTS-style Hakka model:** dialect handled through learnable language embeddings. — [Hugging Face Formospeech](https://huggingface.co/formospeech/taiwanese-hakka-tts-sixian-hailu-mix)
- **Hakka ASR:** dialect-conditioning (separating dialect style from content) gave large gains on HAT. This supports explicit dialect conditioning when sharing a model across Sixian and Hailu. — [arXiv 2602.22522](https://arxiv.org/abs/2602.22522)

### Inferences
- For Sixian/Hailu transfer, the closest analogue is dialect conditioning (language/dialect embedding) inside a shared duration predictor, as in YourTTS-Hakka, accented FastSpeech2 and dialect-aware RNN-T. Sixian and Hailu tone systems differ, with roughly mirror-image tone values, so tone embeddings should be dialect-specific (dialect × tone) rather than shared, or the duration predictor gets the wrong tone prior.
- Speed perturbation, used for Hakka ASR, is an augmentation that changes durations uniformly. It may help the acoustic model but could blur the tone-dependent and checked-vs-open duration contrasts the predictor must learn. Use it cautiously for duration-predictor training.

### Gaps
- No paper was found that reports duration ρ or RMSE as a function of target-data size (e.g., 15 min / 1 h / 5 h) in a cross-lingual or cross-dialect setting.
- No reported duration metrics were found for MMS-TTS, XTTS or YourTTS on tonal languages.

## Q5. What do phonetic studies show about checked-tone (entering-tone) duration in Hakka, Southern Min and Cantonese?

### Takeaway
Checked syllables are consistently shorter. Their realization is highly variable: codas get deleted or glottalized, with compensatory vowel lengthening, register-dependent duration cues, and ongoing sound change. Part of the checked-syllable duration variance is therefore not predictable from text. Coda deletion lengthens the vowel, which directly hurts vowel-nucleus ρ in a Hakka -b/-d/-g model.

### Cited Findings
- **Sixian Hakka:** six contrasting tones, including two entering tones that are "short and rapid", with words ending in /p t k/. Aspirated stops have significantly shorter VOT in entering-tone syllables, though not consistently across datasets. — [CLCLP, ACL Anthology O09-6](https://aclanthology.org/O09-6.pdf)
- **Hakka (command-response model):** Ru-tone syllables end in an unreleased stop and are shorter. — [ResearchGate 221483434](https://www.researchgate.net/publication/221483434_Modeling_tones_in_hakka_on_the_basis_of_the_command-response_model)
- **Meixian Hakka vowels and tones:** acoustic and perceptual PhD thesis (CityU). No entering-tone duration figures appeared in the extract. — [CityU thesis](http://lbms03.cityu.edu.hk/theses/c_ftt/phd-ctl-b40860632f.pdf)
- **Taiwan Min checked tones (Pan, JIPA; EGG):** **over 80% of /ʔ/ codas and under 20% of /p t k/ codas were deleted**. Coda realizations include full closure, energy damping, or irregular glottal vibration. After coda deletion there is **compensatory vowel lengthening**, F0 lowering and more periodic phonation. — [Cambridge JIPA](https://www.cambridge.org/core/journals/journal-of-the-international-phonetic-association/article/glottalization-of-taiwan-min-checked-tones/8E06F7AE20E4F2D59F4B377D34794BC0); [NYCU PDF](https://fl.nycu.edu.tw/wp-content/uploads/2021/08/glottalization-of-taiwan-min-checked-tones.pdf)
- **Taiwan Min sound change (Interspeech 2016/2021):** coda-stop weakening leads to **duration lengthening** and loss of non-modal phonation in checked tones. F0 lowering is dialect-dependent. — [ISCA archive, pan16](https://www.isca-archive.org/interspeech_2016/pan16_interspeech.pdf); [ISCA archive, pan21b](https://www.isca-archive.org/interspeech_2021/pan21b_interspeech.pdf); [ICPhS 2015](http://www.internationalphoneticassociation.org/icphs-proceedings/ICPhS2015/Papers/ICPHS0282.pdf)
- **Perception (Taiwan Min):** duration reliably cues checked vs unchecked tones in the mid register. In the high register, reliance on duration is subtle and glottalization guides perception. — [LabPhon19 paper 38](https://labphon.org/sites/default/files/labphon19/Papers/LabPhon19_paper_38.pdf)
- **Cantonese:** entering tones occur only on /p t k/-final syllables and are shorter, which "seem[s] to conflict" with syllable isochrony. — [John Benjamins, LALI 00098](https://www.jbe-platform.com/content/journals/10.1075/lali.00098.zha?crawler=true). Tonal context also modulates the tone–duration interaction. — [JASA-EL 3(3) 035203](https://pubs.aip.org/asa/jel/article/3/3/035203/2879180/Tonal-context-influences-tone-duration-interaction)
- **Cantonese:** the entering vs non-entering difference is "primarily duration". — [ResearchGate 221488070](https://www.researchgate.net/publication/221488070_Modeling_Tones_In_Continuous_Cantonese_Speech)
- **Checked/unchecked merger (Qixian Jin Chinese):** the acoustic cues behind checked–unchecked merger show the contrast can erode. — [JASA-EL 4(12) 125202](https://pubs.aip.org/asa/jel/article/4/12/125202/3325440/The-role-of-acoustic-cues-in-the-checked-unchecked)
- **Vietnamese analogue:** stop-final and glottal tones are consistently shorter. — [ResearchGate 221478949](https://www.researchgate.net/publication/221478949_Fujisaki_model_based_F0_contours_in_vietnamese_TTS)

### Inferences
- In a Hakka checked syllable, the MFA vowel/coda boundary is ill-defined when the stop is unreleased, weakened or deleted. Label noise on the nucleus duration is therefore probably larger for checked than for open syllables. Measuring ρ at the rhyme level (nucleus + coda) for checked syllables, or using syllable-level ρ, should give a more stable and likely higher number than nucleus-only ρ.
- Where codas weaken, the speaker's own variability (coda kept vs deleted, with compensatory lengthening) is not predictable from text. This sets a ceiling below 1.0 for checked-syllable nucleus ρ. The size of this ceiling is unmeasured for Hakka.
- Practical check: compute ρ separately for checked and unchecked nuclei and inspect checked-syllable alignments where the stop closure is missing. If many are misaligned, fixing alignment (better Hakka MFA model from HAT, or merging nucleus+coda into one duration unit) may yield more ρ than predictor changes.

### Gaps
- No acoustic study was found that gives Sixian or Hailu entering-tone duration means and SDs, or coda-deletion rates for -b/-d/-g (Hakka analogues of Pan's Taiwan Min data).
- No study was found comparing MFA or forced-alignment accuracy on checked vs open syllables in any Sinitic language.
