# 實驗 C:持續時間感知建模(程式)

對應教學手冊第 25 章。目標:先**診斷**入聲母音偏長從哪裡來,再依診斷結果**修正**,
最後用事先登記的標準**判讀**改善是否成立。

> 這些程式是在看不到你電腦的情況下寫的:Matcha-TTS 的部分依照官方 0.0.7.2 原始碼,
> 你自己的資料與腳本(PUA 對照表、TextGrid、checkpoint 命名、evaluate_v2.py)依照手冊的描述。
> 所以**第一步一定是執行 `probe_env.py`,把 `probe_report.txt` 貼回對話**,有不符的地方我再改。

所有程式都**不會修改**你既有的腳本或 Matcha 原始碼。會寫入的只有:
`~/hakka_tts/exp_c/`(所有輸出)、C2 的 `durations/*.npy` 與 `*_mfadur.txt` 清單,
以及 `c2_setup.py` 新增的兩個 config 檔。

---

## 0. 放到你的電腦

```bash
cd ~                                   # 或任何你放程式的地方
git clone https://github.com/Ffffkg/my-first-repo.git    # 已 clone 過就 git pull
cd my-first-repo/experiments/duration_c
conda activate tts
```

路徑預設是手冊第 14.2 節的 `~/hakka_tts`、`~/Matcha-TTS`。不一樣的話:

```bash
export HAKKA_ROOT=/你的路徑/hakka_tts
export MATCHA_ROOT=/你的路徑/Matcha-TTS
```

## 1. 環境檢查(先做這個)

```bash
python probe_env.py          # 產生 probe_report.txt,整份貼回對話
```

接著跑單元測試(`pip install pytest` 一次即可)。`tests/test_matcha_io.py` 會用**你本機的 Matcha**
建一個隨機小模型,驗證這裡的 MAS 與合成路徑和 Matcha 原始程式逐點一致:

```bash
python -m pytest tests -q            # 應全部通過;若 test_matcha_io 失敗,代表你的 Matcha 有改動,把輸出貼回來
```

`probe_env.py` 會檢查:Matcha 版本是否支援 `load_durations`、PUA 對照表的結構、符號序列能否與
TextGrid 逐一對應、梅爾長度是否等於「樣本數 // 256」、各資料量 checkpoint 找不找得到、
`evaluate_v2.py` 的函式介面、`resample_audio.py` 的裁切邏輯,以及**自寫合成路徑是否與
`model.synthesise()` 逐點相同**。

## 2. C0:偏差來源診斷(不重訓,約幾十分鐘)

### 2.1 回歸核對:先確認這套程式量得出主表的數字

```bash
python c0_diag.py --exp hakka_joint_sixian_ladder_48h --dialect sixian --first 100
python c0_report.py ~/hakka_tts/exp_c/c0/hakka_joint_sixian_ladder_48h_test_first100.csv
```

報表第 0 節會和主表比較(四縣 48h:ET-VBias +23.3、NonET +0.4;海陸 47.2h:+18.8、−0.6)。
**出現 ✓ 才往下做**;若沒有重現,把報表貼回來,一起找第一個分歧點(手冊第 11 章的做法)。
海陸的 48h 實驗名稱若不同(例如 `_47h`),用 `--exp` 換成實際名稱。

### 2.2 正式診斷

每個腔調、至少最大資料量與 1h 兩個模型,各跑測試集與驗證集(驗證集給 C1 用):

```bash
for d in sixian hailu; do
  for h in 1 48; do
    exp=hakka_joint_${d}_ladder_${h}h
    python c0_diag.py --exp $exp --dialect $d                      # 全部測試句
    python c0_diag.py --exp $exp --dialect $d --split val          # 驗證集
    python c0_report.py ~/hakka_tts/exp_c/c0/${exp}_test_all.csv
  done
done
# (選做)訓練集抽 1000 句,看訓練目標本身:
python c0_diag.py --exp hakka_joint_sixian_ladder_48h --dialect sixian --split train_full --max-utts 1000
```

### 2.3 怎麼讀報表

報表把主表的 ET-VBias 拆成三段(恆等式,報表會檢查加總):

| 成分 | 意思 | 若它是主因 |
|---|---|---|
| 目標偏差 MAS − MFA | 訓練時時長預測器學的目標本身就偏長 | **C2**:改用 MFA 時長當目標 |
| 預測偏差 w − MAS | 預測器沒學好 | **C3**:預測器改良 |
| 取整偏差 ceil(w) − w | 推論時 Matcha 對每個符號(含 blank)無條件進位 | **C1**:改用累積四捨五入 |

關於「取整偏差」:Matcha 的 `synthesise()` 用 `torch.ceil(w)`,每個符號平均多約半個 frame,
blank 也一樣;merge_blank 之後,每個母音大約多 1 個 frame(≈ 11.6 ms)。這可能是合成語音
整體偏長(總長比 1.22–1.30)的原因之一。不過它對入聲、舒聲大致一樣,所以**應該主要影響
ET-VBias 與總長,不太影響 ΔBias**。這是待驗證的假設,看報表第 1 節的數字再下結論。

報表第 2 節(blank 規則)、第 3–4 節(是不是「短母音都被高估」)、第 5 節(韻尾/調值/停頓前/
性別分組,也就是實驗 F1 的預覽),第 6 節會依手冊第 25.3 節的決策表自動給建議。

## 3. C1:推論時校正(不重訓、不需 GPU)

```bash
exp=hakka_joint_sixian_ladder_48h
python c1_calibrate.py --val ~/hakka_tts/exp_c/c0/${exp}_val_all.pkl \
                       --test ~/hakka_tts/exp_c/c0/${exp}_test_all.pkl
python c_compare.py --base   ~/hakka_tts/exp_c/c1/${exp}/base.csv \
                    --method ~/hakka_tts/exp_c/c1/${exp}/class_cum.csv
```

五個變體:`base`(原本)、`cum`(只換取整)、`global`(整體語速)、`class_ceil` / `class_cum`
(入聲/舒聲母音核各自縮放)。參數只在驗證集估。要聽、要算 MCD:

```bash
python c_eval.py --exp $exp --dialect sixian --synth --selfcheck                       # baseline
python c_eval.py --exp $exp --dialect sixian --synth \
       --calib ~/hakka_tts/exp_c/c1/${exp}/calib.json --variant class_cum              # 校正後
python c_compare.py --base ~/hakka_tts/exp_c/eval/${exp}__base \
                    --method ~/hakka_tts/exp_c/eval/${exp}__class_cum
```

> C1 直接改時長,所以「時長指標變好」幾乎是必然的(ρ 不應改變)。它是**基準線**:
> 後面要重訓的方法至少要比它好。真正的效果要用實驗 D(波形)與 E(聽測)確認。

## 4. C2:用 MFA 時長監督訓練(需重訓)

### 4.1 產生 durations:先決定「裁切起點」怎麼取得

TextGrid 對齊的是**原始未裁切**音檔;22 kHz 訓練音檔被 `resample_audio.py` 裁過句首靜音,
所以每句要知道 offset。看 `probe_report.txt` 裡 `resample_audio.py` 的相關行,選一種:

| 情況 | 用法 |
|---|---|
| 有記錄每句 offset | `--offset-mode csv --offset-csv 檔案`(每行 `stem,秒`) |
| 裁切規則是「第一個音素前留 X ms」 | `--offset-mode margin --margin-ms X` |
| 不確定 | `--offset-mode xcorr --orig-root <原始 wav 根目錄>`(用互相關自動對位,最穩) |
| 已在 22 kHz 音檔上重跑過 MFA | `--offset-mode zero` |

你的 `resample_audio.py` 是「第一個與最後一個 word 前後各留 `MARGIN = 0.05` 秒」(probe 已確認),
所以用 margin 模式、50 ms。程式也會檢查句尾留白是否約等於 50 ms,不符的句子(例如沒被裁切的)會被排除。
先試跑 200 句、不寫檔,看句首/句尾留白是否都集中在 50 ms:

```bash
python c2_mfa2durations.py --dialect sixian --offset-mode margin --margin-ms 50 --dry-run --limit 200
python c2_mfa2durations.py --dialect sixian --offset-mode margin --margin-ms 50          # 正式
```

會寫出 `data/sixian/wav22k/durations/<stem>.npy`(Matcha 規定的位置)與每份清單的
`*_mfadur.txt`(排除轉換失敗的句子)。每個符號至少 1 frame(Matcha 的時長損失取 log);
blank 一律向左邊音素借 frame,因此內部音素經 `merge_blank` 後會**恰好**還原成 MFA 的長度。

### 4.2 產生 config 並訓練

```bash
python c2_setup.py --base-exp hakka_joint_sixian_ladder_1h        # 只改 data 與 use_precomputed_durations
# 依畫面印出的指令:先用 --cfg job 確認 load_durations / use_precomputed_durations 都是 true,再背景訓練
```

建議用實驗 B 的固定 40,000 步規則當基準(兩邊只差訓練目標,約 2 小時/個):
`--base-exp fair1h_joint_sixian_scratch_s1234`、`--base-exp fair1h_joint_hailu_scratch_s1234`。
新實驗名稱會是 `c2_mfadur_fair1h_joint_<腔>_scratch_s1234`。有方向再跑最大資料量。

### 4.3 評估

```bash
python c_eval.py --exp hakka_joint_sixian_ladder_1h --dialect sixian --synth
python c_eval.py --exp c2_mfadur_joint_sixian_ladder_1h --dialect sixian --synth
python c_compare.py --base   ~/hakka_tts/exp_c/eval/hakka_joint_sixian_ladder_1h__base \
                    --method ~/hakka_tts/exp_c/eval/c2_mfadur_joint_sixian_ladder_1h__base
```

`c_compare.py` 最後會列出手冊第 25.8 節的成功標準與通過與否。門檻是**建議值**
(NonET ±5 ms、MCD +0.1 dB、F0 +0.05 st),與老師確認後用參數改成登記的數字。

## 5. D1:從波形量母音時長(評估 C2 一定要做)

C0 顯示 MAS 把很多時間放在母音前後的 blank(每個 30–80 ms),所以「attn 推得的母音時長」很依賴
blank 怎麼分。baseline(MAS)和 C2(blank 只有 1 frame)的 blank 結構不同,**只用 attn 版比較兩者
不公平**,一定要從波形量。做法:用當初對齊真人錄音的同一套 MFA 模型,對齊合成語音。

```bash
python c_eval.py --exp fair1h_joint_sixian_scratch_s1234 --dialect sixian --synth      # 合成 + 存 wav 與邊界
python d1_prepare_mfa.py --eval-dir ~/hakka_tts/exp_c/eval/fair1h_joint_sixian_scratch_s1234__base --dialect sixian
#   → 切到 MFA 的環境,執行畫面印出的 mfa align 指令
python d1_compare.py --eval-dir ~/hakka_tts/exp_c/eval/fair1h_joint_sixian_scratch_s1234__base --dialect sixian
```

`d1_summary.md` 會回答:從波形量入聲還偏長嗎、attn 版差多少、哪種 blank 規則最接近波形。
兩個系統都跑完 D1 後,用波形版比較:

```bash
python c_compare.py --base   <baseline eval 資料夾>/syllables_wave.csv \
                    --method <C2 eval 資料夾>/syllables_wave.csv
```

## 5b. 一行跑完 D1、多種子、聽測

```bash
# 合成 + MFA 對齊 + 波形比較(一個系統一行)
bash d1_pipeline.sh sixian base_sixian_step40000 fair1h_joint_sixian_scratch_s1234 --which step40000
python c_compare.py --base ~/hakka_tts/exp_c/eval/base_sixian_step40000 --method ~/hakka_tts/exp_c/eval/c2_sixian_step40000 --syl wave
python f1_subgroups.py <系統1>/syllables_wave.csv <系統2>/syllables_wave.csv      # 實驗 F1:依韻尾/調值/停頓前/性別

# 換訓練種子(不用新 config:Hydra 覆寫 seed 與 run_name,輸出資料夾跟著 run_name)
cd ~/Matcha-TTS && python matcha/train.py experiment=c2_mfadur_fair1h_joint_sixian_scratch_s1234 seed=123 run_name=c2_mfadur_fair1h_joint_sixian_scratch_s123

# 實驗 E2:成對比較聽測
python e_build_ab.py --a <系統A 的 eval 資料夾> --b <系統B 的 eval 資料夾> --n 30 --out ~/hakka_tts/exp_e/<名稱>
#   把 <out>/listening_test/ 整個資料夾交給聽者(瀏覽器開 index.html,作答完下載 CSV 回傳);key.csv 自己保留
python e_analyze.py --key ~/hakka_tts/exp_e/<名稱>/key.csv --responses <放回傳 CSV 的資料夾>
```

## 6. C3

依 C0 的結果決定要不要做(手冊第 25.6 節)。等 C0 報表出來後,我再依結果寫
(預測器加音節結構特徵 / 音節層級時長 / 損失改良)。

## 7. 檔案一覽

| 檔案 | 用途 | 需要 GPU |
|---|---|---|
| `probe_env.py` | 環境與資料檢查,產生 probe_report.txt | 選用 |
| `c0_diag.py` | C0:真人錄音 MAS、預測時長、四種 blank 規則 → 逐音節 CSV + pickle | 是(只跑編碼器) |
| `c0_report.py` | C0 報表:拆解、blank 敏感度、分箱、迴歸、分組、建議 | 否 |
| `c1_calibrate.py` | C1:驗證集估校正參數,測試集評估五個變體 | 否 |
| `c_eval.py` | 任一模型的 ET 指標;`--synth` 合成算 MCD,存 wav 與邊界(實驗 D 會用) | 是 |
| `c_compare.py` | 兩系統配對比較 + 成功標準檢查 | 否 |
| `c2_mfa2durations.py` | C2:TextGrid → durations/*.npy、*_mfadur.txt | 否 |
| `c2_setup.py` | C2:由 baseline 產生 data / experiment config | 否 |
| `d1_prepare_mfa.py` | D1:合成音檔 → MFA corpus(wav + 拼音 .lab),印出 mfa align 指令 | 否 |
| `d1_compare.py` | D1:波形 vs attn 母音時長、blank 比例、波形版 ET 指標 | 否 |
| `d1_pipeline.sh` | 一行跑完 合成 → MFA 對齊 → 波形比較 | 是 |
| `f1_subgroups.py` | F1:依韻尾、調值、停頓前、性別分組的 VBias / VDE / ρ | 否 |
| `e_build_ab.py` | E2:產生成對比較聽測(HTML + 音檔,響度正規化,含注意力檢查題) | 否 |
| `e_analyze.py` | E2:CMOS 與偏好比例,聽者×句子交叉重抽 CI,注意力檢查排除 | 否 |
| `hakka_c/` | 共用模組(路徑常數、TextGrid、符號對應、時長換算、統計、Matcha 介面) | — |
| `tests/` | 單元測試:`python -m pytest tests -q`(不需 GPU) | 否 |

輸出都在 `~/hakka_tts/exp_c/`:`c0/`、`c1/<模型>/`、`eval/<模型>__<變體>/`。

## 8. 我做的假設(probe_env.py 會逐一檢查)

1. 文字 cleaner 名稱是 `hakka_cleaners`(不同就加 `--cleaners`)。
2. 「PUA 字元 ↔ 音素」對照:你的 benchmark 裡沒有 `hakka_matcha_vocab.json`,程式會先找其他 JSON
   (每個候選都用 TextGrid 驗證),找不到就**直接從 filelist + TextGrid 學出來**,存成
   `~/hakka_tts/exp_c/pua_map_learned.json`(可以打開檢查,`readable` 欄位是人看得懂的版本)。
3. 符號名稱去掉 `@聲調` 後與 MFA 音素標籤相同;停頓符號叫 `sp` / `sil`。
4. TextGrid 有 `words` 與 `phones` 兩層,words 層每個 word 是一個拼音音節(如 `siid5`)。
5. checkpoint 在 `logs/train/<exp>/runs/<日期>/checkpoints/best_*_<驗證損失>.ckpt`,取最新一次 run。
6. 從零訓練模型的 mel buffer 等於 data config 的統計量(手冊第 9.3 節);遷移模型要加 `--data-yaml`。
   較早訓練的 joint checkpoint 沒有 `encoder.tone_table`(patch_joint_tonetable.py 之後才加),
   載入時只容許缺這兩個查找表,其他任何不符都會報錯。
7. `evaluate_v2.analyze_wav()` 可以直接吃 numpy 波形,回傳 `(f0, mcep)`。
   (torch ≥ 2.6 的 `torch.load` 預設 `weights_only=True` 會拒絕 Matcha checkpoint;程式已自動改用
   `weights_only=False` 載入你自己訓練的 checkpoint。)
8. 音節結構(母音核位置、是否入聲)取自 TextGrid,不依賴前端;與 diag_duration.py 的結果應一致,
   2.1 節的回歸核對就是在驗證這一點。
