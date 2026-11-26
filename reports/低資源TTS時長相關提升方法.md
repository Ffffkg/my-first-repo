# 先修量測再修模型:客語時長 ρ 衝 0.9

在「純文字輸入、確定性時長預測、只看入聲母音核、真人與合成兩端都用 MFA 量測、真人只錄一次」這個設定下,ρ 0.74 已落在文獻同類目標(母音/韻母/音節層級)的 **0.62–0.79** 區間中段;48 h 同架構的 0.77–0.80 很可能已貼近「自然變異 + 對齊雜訊」構成的可觀測上限。光靠換模型、加資料,很難把這個數字推到 0.85 以上。0.8–0.9 實際上要靠三件事。第一是改量測:合成端直接讀模型輸出的時長,不再對合成音重新對齊;參考端用人工校正或重新訓練的客語對齊器;用多次朗讀的平均當參考。第二是改評估單位:入聲改用韻母或音節,或把入聲與舒聲合併計算。第三是補上文字中缺少的局部韻律資訊:片語邊界、表層聲調、韻尾類型、句中位置,或在應用允許時加入參考韻律條件。文獻中各類模型方法(音素自監督預訓練、BERT/句法特徵、隨機時長模型、跨語遷移、資料增強、蒸餾)幾乎都只報 MOS/CMOS,能找到的時長指標增益只有個位數百分比,例如 CCC +0.03、時長誤差 −3.6% 到 −15%,沒有一篇顯示時長相關能提升 0.1。你自己的數據也指向同一件事:整句語速 oracle 只帶來 +0.01–0.02,GBM 與 1 h 神經模型同為 0.74,48 倍資料只多 0.03–0.06。殘差主要在句內的局部變異和量測雜訊,不在整句語速,也不在模型容量。因此,成本效益最高的順序是:先花一兩週改量測與報告方式,估出「這位發音人、這條管線」的真實上限;再決定要不要投入重訓客語對齊器、加顯式韻律特徵、做多語者時長預訓練。

> **來源與核對說明**:研究者所在的沙盒封鎖了 arxiv.org、isca-archive.org、PMC、ResearchGate、Semantic Scholar 等全文來源,因此**本報告所有文獻數字都只來自搜尋引擎摘要,沒有讀過全文表格,引用前務必回原文核對**。另外兩類標記如下:**【二手】**表示數字出自第三方摘要、排行榜或彙整,原始論文未確認,可靠性更低;**【推論】**表示是研究者或本報告依文獻推導出來的估計,不是任何論文的實測值。你專案內的數字(0.74、0.77–0.80、GBM 0.74、+0.01–0.02)沿用題述,不另加標註。

## 母音核層級的時長相關在文獻中多落在 0.62–0.79

先校準期望:**近年的 FastSpeech/FastSpeech 2/VITS/Matcha 類論文幾乎都不報告時長的 Pearson 相關**。它們只報 log 幀 MSE/MAE,或乾脆只報 MOS。例如 FastSpeech 2 用 MFA 時長的 log-MSE 訓練,但沒有給出相關係數([FastSpeech 2](https://arxiv.org/pdf/2006.04558));PAMA-TTS、FluentSpeech 也只報 MAE/MSE([PAMA-TTS](https://arxiv.org/pdf/2110.04486);[FluentSpeech](https://arxiv.org/pdf/2305.13612))。能直接拿來比對的相關係數,大多來自 2000–2016 年的統計式或早期神經式時長模型。

這批數字有很清楚的結構:**子音或「全音素混合」的 r 很高,母音/韻母/音節的 r 明顯較低**。Mandarin 多元迴歸模型中,聲母 r = **0.889**(RMSE 18.6 ms),韻母只有 **0.621**(36.9 ms),音節 **0.668**(43.1 ms)([zhu00b, ISCSLP 2000](https://www.isca-archive.org/iscslp_2000/zhu00b_iscslp.pdf))。泰語節段時長以含聲調、詞/片語位置、詞性的分類迴歸達 r = **0.788**,RMSE 33.14 ms([Thai segmental duration](https://www.researchgate.net/publication/228943663_Analysis_of_Segmental_Duration_for_Thai_Speech_Synthesis));泰語音節時長測試集 r 約 **0.73**,另一摘要給出約 0.79([Hansakunbuntheung et al., Eurospeech 2003](https://www.isca-archive.org/eurospeech_2003/hansakunbuntheung03_eurospeech.pdf))。韓語母音 r = **0.68**,RMSE 32.13 ms【二手,原論文未確認】([彙整來源](https://www.sciencedirect.com/science/article/abs/pii/S0885230806000234))。英語單一語者音素層 LSTM r = **0.765**,DNN r = 0.750([Ronanki et al. 2016](https://arxiv.org/pdf/1608.06134))。印度語言 DNN 對強制對齊時長的 r,泰盧固語 **0.818**、坦米爾語 0.657、印地語 0.564,同一方法跨語料差距很大(語言與數值的配對未完全確認)([arXiv 1608.05374](https://arxiv.org/pdf/1608.05374))。Mandarin 階層式韻律模型的開放測試中,只有 **43%** 音節的相對誤差在 10% 以內([JCIE 2005](https://www.tandfonline.com/doi/abs/10.1080/02533839.2005.9671006))。

接近 0.9 的報告都有「有利條件」。塞爾維亞語迴歸樹 r = **0.8796**、RMSE 14.89 ms,算的是**包含子音的全音素**([EEJournal 4090](https://eejournal.ktu.lt/index.php/elt/article/view/4090))。TIMIT 上 sums-of-products 模型 r = **0.94**、RMSE 9 ms【二手,原論文未確認】。T5Gemma-TTS 在 JSUT 上生成時長與目標時長的 r ≈ **0.92**,但那是有參考/目標時長條件的設定,不是從文字自由預測([T5Gemma-TTS](https://arxiv.org/html/2604.01760v1))。台灣階層式韻律模型可以把音節時長重建到 RMSE 約 9 ms,韻律編碼甚至到 4.8 ms,但都是**從真實語音分析再重建**,屬於 oracle 韻律,不是文字預測([IJST 2010](https://link.springer.com/article/10.1007/s10772-010-9067-z);[EURASIP JASMP 2018](https://link.springer.com/article/10.1186/s13636-018-0129-5))。van Santen 1994 常被引用的「單一語者母音 r ≈ 0.9+」在摘要中無法核實([CSL 1994](https://www.sciencedirect.com/science/article/abs/pii/S0885230884710059))。

| 條件 | 報告值 | 與本專案可比性 |
|---|---|---|
| 韻母 / 音節(Mandarin 迴歸) | r 0.621 / 0.668 | 高:同為 Sinitic 母音性單位 |
| 母音(韓語) | r 0.68【二手】 | 高 |
| 音素層 LSTM(英語) | r 0.765 | 中:含子音 |
| 節段 / 音節(泰語,聲調語言) | r 0.788 / 0.73–0.79 | 中高 |
| 聲母(Mandarin) | r 0.889 | 低:子音時長容易預測 |
| 全音素(塞爾維亞) | r 0.8796 | 低:音素身分拉大變異 |
| 有目標時長條件(T5Gemma-TTS) | r ≈ 0.92 | 不可比:任務不同 |

【推論】只算母音核時,「子音 vs 母音」「塞音閉塞 vs 擦音」這類由音素身分決定的大變異被拿掉了,所以**母音核上的 0.8 大約等於全音素上的 0.9**。另外要先確認題述的 ρ 是只算入聲核,還是入聲加舒聲混合。若是混合,入聲短、舒聲長的類別差異本身就會撐高 ρ,類別內的 ρ 應該更低。若只算入聲核,入聲核的時長變異範圍窄,0.74 反而比表面上更難得。

## MFA 誤差和自然變異可能把可觀測 ρ 壓在 0.8–0.88

**對齊精度的實測值。**MFA 在 TIMIT 測試集上,音素邊界落在 10/20/25/50 ms 內的比例分別是 **47.96% / 75.84% / 84.14% / 94.97%**([Kelley et al., MAPS](https://arxiv.org/pdf/2310.15425))。MFA 3.x 的平均邊界誤差在 TIMIT 從 16.4 ms 降到 **12.1 ms**,在 Buckeye 從 17.6 ms 降到 13.9 ms([MFA 2026](https://arxiv.org/pdf/2606.18466);[MFA 部落格](https://memcauliffe.com/update-on-montreal-forced-aligner-performance.html))。原始 MFA 論文指出,詞邊界平均誤差 24 ms,與人工標註者之間的 26 ms 相當,但 **>25 ms 的中大型誤差比人工多**([McAuliffe et al. 2017](https://www.isca-archive.org/interspeech_2017/mcauliffe17_interspeech.pdf))。Interspeech 2024 的比較中,MFA 的對齊仍勝過 WhisperX 與 MMS([Rousso et al. 2024](https://arxiv.org/pdf/2406.19363))。FastSpeech 2 自報 MFA 邊界誤差 **12.47 ms**,注意力對齊則是 19.68 ms([FastSpeech 2](https://arxiv.org/pdf/2006.04558))。和本專案最相關的一點是:**對齊器在「母音之後」的邊界比「母音之前」的邊界更不準**([Gonzalez, Grama & Travis 2020](https://researchportalplus.anu.edu.au/en/publications/comparing-the-performance-of-forced-aligners-used-in-sociophoneti/))。入聲母音核的終點正是「母音→不除阻塞音」邊界,幾乎是最難對準的位置。這些精度數字都來自訓練良好的英語模型。研究者**找不到任何 MFA 在客語、Sinitic 聲調語言或 TTS 合成音上的邊界精度實測**。

**方言內訓練的對齊器能大幅改善邊界。**成都話研究中,方言內訓練的 XLS-R frame-classification 對齊器讓 **82.2%** 邊界落在 50 ms 內,Charsiu-Mandarin 基線只有 **52.8%**;方言內訓練的 MFA 在 10 ms 以上的每個容差都勝過 Mandarin MFA([Chengdu Mandarin forced alignment](https://arxiv.org/pdf/2607.21332))。由此可推論,用 Mandarin 或跨腔模型直接對齊客語,誤差很可能明顯大於上述 TIMIT 數字【推論】。

**衰減算術【推論】。**觀測到的 ρ 等於真實 ρ 乘上 √(rel_ref × rel_syn),其中信度 rel = 真實變異 /(真實變異 + 量測誤差變異)。時長是兩個邊界相減,若每個邊界誤差的 SD 是 σ_b,則時長誤差約為 √2·σ_b。平均邊界誤差 12–20 ms 換算成時長誤差 SD 約 **20–35 ms**。若真實母音核時長 SD 為 50 ms,時長誤差 20 ms 時 rel ≈ 0.86,30 ms 時 rel ≈ 0.74。**入聲核本身短、變異範圍窄**,假設真實 SD 只有 30 ms,同樣 20 ms 的誤差會讓 rel 掉到約 0.69。雙邊相乘後,「完美預測器」能觀測到的 ρ 只剩約 0.74–0.88。反過來推,若 rel_ref = 0.8、rel_syn = 0.9,觀測到的 0.74 對應真實 ρ 約 **0.87**,48 h 的 0.80 對應約 0.94。研究者沒有找到任何論文對 TTS 時長相關做過衰減校正。這組算術方向可信,實際大小必須在你的資料上量出來。

需要注意一個反向效應【推論】:真人與合成音由同一個 MFA 模型對齊,系統性偏誤(例如某類音境固定偏長)可能在兩端同向出現,反而**膨脹** ρ。所以改用模型輸出時長後,ρ 不一定上升。但改完之後的數字更乾淨,也更容易解釋。

**自然變異的上限沒有人量過。**研究者找不到任何研究報告「同一人重複朗讀同一句」的音段時長相關。間接證據顯示這個上限明顯低於 1:兩位語者讀同一語料的音段時長相關只有 **r = 0.66**【二手】([van Heerden & Barnard 周邊引用](https://repository.up.ac.za/items/518c5f0d-65bf-4def-8df7-b916ba3b5fc3/full));10 次重複的 CVCa 實驗中,**長母音的變異大於短音段**([van den Heuvel et al. 1991](https://www.isca-archive.org/eurospeech_1991/heuvel91_eurospeech.html));2025 年 JASA-EL 指出,同一語者在同一語境重複發同一母音,時長也「可以有相當大的差異」([JASA-EL 5(11)](https://pubs.aip.org/asa/jel/article/5/11/115202/3371042/Formant-variability-is-related-to-vowel-duration))。

**上限的計算方式【推論】。**設兩次朗讀時長的相關為 ρ_tt,它同時包含發音變異與參考端 MFA 雜訊。一個完美輸出「條件期望值」的確定性預測器,與單次參考的最大期望相關是 **√ρ_tt**;若參考改用 k 次朗讀的平均,上限依 Spearman-Brown 公式升為 √(kρ_tt /(1+(k−1)ρ_tt))。例如 ρ_tt = 0.70 時,單次參考上限 **0.84**,兩次平均 **0.91**,三次平均 0.935;ρ_tt = 0.60 時,單次上限只有 0.775,兩次平均 0.87。換句話說,**要在單次參考上觀測到 0.9,ρ_tt 至少要 0.81**。這是否可能,只有實驗能回答。

**統計顯著性也是量測問題【推論】。**依 Fisher z 估算,ρ = 0.74 時,若測試集只有 300 個入聲核,95% 信賴區間約 **±0.05**;1000 個核時約 ±0.03。整句語速 oracle 的 +0.01–0.02,以及 1 h 與 48 h 的部分差距,可能落在抽樣誤差內,需要用同一批核做配對 bootstrap 才能判定。

## 模型面方法多以 MOS 報告,時長指標增益只有個位數百分比

把八大類方法攤開來看,共同點很一致:**沒有任何一篇論文報告某方法讓音素時長的 Pearson ρ 提升 0.05 以上**。多數只報 MOS/CMOS。少數有時長指標的,增益落在相關 +0.03、或誤差降 3–15%。

| 方法類別 | 實證證據(時長相關指標優先) | 量級 | 對本專案預期【推論】 |
|---|---|---|---|
| 參考語音/語句嵌入條件 | 時長預測器加入同一語句的 ASV 嵌入,CCC **0.771→0.804**,MAE 2.045→1.869 幀;同語者嵌入 0.801;隨機雜訊 0.733([arXiv 2406.19243](https://arxiv.org/html/2406.19243)) | CCC +0.03 | 與你的整句語速 oracle +0.01–0.02 吻合:**句級資訊頂多 +0.03,此路已近乎用盡** |
| 隨機/生成式時長 | Mehta et al. 2024 在 Matcha-TTS **朗讀語料上,確定性與 OT-CFM 時長無顯著差異**;FastSpeech 2 換成隨機時長後全面變差;VITS 與自發語料則受益,但指標是 MOS([Mehta et al. 2024](https://arxiv.org/abs/2406.05401));TDA 的 MaskGIT 時長 FDD 0.206,迴歸 0.403、flow matching 0.318【二手】([Eskimez et al. 2024](https://arxiv.org/abs/2406.04281);[摘要](https://lacuna.tiptreesystems.com/work/total-duration-aware-duration-modeling-for-text)) | 改善的是分布與自然度,不是對單一參考的相關 | 對單一參考取樣,期望 ρ 不會超過條件均值;**不建議為 ρ 採用** |
| 均值+變異輸出頭 | DurFormer 相對 FastSpeech 2 音素時長 MSE 降約 **15.05% / 11.26%**【二手】([arXiv 2507.22612](https://arxiv.org/html/2507.22612v2);[Pith 評論](https://pith.science/paper/2507.22612)) | MSE −11~15% | 低成本可試,取均值輸出;ρ 增益小 |
| 文字/音素自監督預訓練 | XPhoneBERT(約 100 語言、3.3 億音素句)在 **5% 訓練資料下 MOS 2.88→3.22(英)、1.59→3.35(越)**,全資料時只 +0.14/+0.15([XPhoneBERT](https://www.isca-archive.org/interspeech_2023/thenguyen23_interspeech.pdf));MP-BERT 對 FastSpeech 2 CMOS +0.30([arXiv 2203.17190](https://arxiv.org/abs/2203.17190));PL-BERT MOS 4.19→4.24,增益集中在域外文本([arXiv 2301.08810](https://arxiv.org/abs/2301.08810)) | 只有 MOS,無時長指標 | 低資源時有效,但對時長 ρ 的效果未知;估計 +0.02–0.05 |
| 韻律導向文字預訓練 | Dict-TTS 字級時長誤差 **35.7→34.4**(−3.6%)【二手,來自排行榜轉抄】([arXiv 2206.02147](https://arxiv.org/abs/2206.02147);[SOTA2](https://www.sota2.com/research/sota/prosodic-realization-analysis-on-biaobei));CLAPSpeech 時長誤差優於 BERT/A3T,數值未取得([arXiv 2305.10763](https://arxiv.org/abs/2305.10763)) | 誤差 −數% | 同上 |
| 語言模型/句法特徵 | BERT 嵌入讓 Tacotron 2 的 MOS「小幅但顯著」提升([Hayashi et al. 2019](https://www.isca-archive.org/interspeech_2019/hayashi19_interspeech.html));韻律品質隨 PLM 大小**呈對數成長**([Granero-Moya et al. 2023](https://arxiv.org/html/2309.01576v1));SyntaSpeech 只有時長損失收斂曲線([IJCAI 2022](https://www.ijcai.org/proceedings/2022/0620.pdf));句中停頓長度在現代 TTS 中的 R² 只有 0.14–0.44【二手,出處未確認】 | 只有 MOS | 對時長有用的是**邊界與片語末資訊**,不是語意本身 |
| 語音自監督與對齊器 | 成都話 XLS-R 對齊器 50 ms 內 **82.2% vs 52.8%**([arXiv 2607.21332](https://arxiv.org/pdf/2607.21332));UnitSpeech、QS-TTS 用 HuBERT 單元與單元時長做半監督,沒有時長準確度數字([UnitSpeech](https://www.isca-archive.org/interspeech_2023/kim23k_interspeech.pdf);[QS-TTS](https://arxiv.org/pdf/2309.00126)) | 邊界精度 +29 個百分點 | **同時淨化訓練標籤與評估參考**,是模型以外最大的槓桿 |
| 跨語遷移/多語者預訓練 | 30 分鐘目標語料加英日預訓練,擴增到約 3 h 後接近 12 h 基線(MOS)([EURASIP 2021](https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-021-00225-4));以顯式時長模型搭配 **15 分鐘**目標語料,自然度可追上 10 h 的 Tacotron 2([Shah et al. 2021](https://arxiv.org/abs/2106.12896)) | 只有 MOS | 你已做跨腔初始化;多語者客語時長預訓練最多把 1 h 拉到 48 h 水準 |
| 蒸餾/偽標籤 | FastSpeech 從教師注意力取時長([FastSpeech](https://arxiv.org/pdf/1905.09263));低資源參考模型偽標籤([arXiv 2210.14723](https://arxiv.org/pdf/2210.14723)) | 無時長數字 | 教師的韻律不一定是道地客語節奏,風險大 |
| 資料增強(變速) | 30 分鐘擴到 13 h、或資料量擴 10 倍,都沒有時長指標([APSIPA 2021](https://www.apsipa.org/proceedings/2021/pdfs/0000849.pdf)) | 無時長數字 | 變速是整句等比縮放,不增加句內相對時長樣式,**對 ρ 幾乎無益**,還可能模糊入聲/舒聲對比 |

你自己的資料規模曲線是最有說服力的證據:**1 h → 48 h(約 1.7 個數量級)只多 0.03–0.06**。若照對數線性外推,從 0.80 到 0.85 還需要 1.4–2.8 個數量級,也就是上千到數萬小時,實際上還可能更早飽和【推論】。GBM 只靠文字特徵就達到 0.74,和 1 h 神經模型一樣,表示文字可預測的那部分已大致被抓到【推論】。剩下的變異來自三處:句內的片語末延長與強調、入聲韻尾的實現方式(保留、弱化或刪除)、量測雜訊。這三者都不是加大時長預測器或改用生成式時長能解決的。

## 聲調語言證據指出:入聲是雜訊與變異最集中的地方

聲調語言的時長研究結論相當一致。最強的時長預測因子依序是:**韻律邊界層級(尤其是右邊界)**、片語內位置、聲調類別、母音長短類別或音節結構、右鄰音節。Mandarin 研究指出,右邊界是否低於韻律詞層級,是「韻母時長最重要的決定因素」([Guo & Katae 2006](https://www.isca-archive.org/speechprosody_2006/guo06_speechprosody.pdf));後接音節對時長的貢獻顯著大於前接音節([zhu00b](https://www.isca-archive.org/iscslp_2000/zhu00b_iscslp.pdf))。越南語以聲調、音素數、片語位置、右音節邊界深度為主要因子([VnVoice](https://www.researchgate.net/publication/228943672_Duration_Modeling_in_a_Vietnamese_Text-to-Speech_System))。泰語母音長短類別影響最大,**依母音類別分開建模後,誤差相對下降 23.21%**([ACL W14-5106](https://aclanthology.org/W14-5106.pdf))。Cross-Lingual F5-TTS 發現中文最適合在**音節層級**預測語速,英語則是音素層級([arXiv 2509.14579](https://arxiv.org/pdf/2509.14579))。台灣 Chiang 團隊的階層式韻律模型把斷點類型、韻律狀態與語速一起建模([EURASIP 2018](https://link.springer.com/article/10.1186/s13636-018-0129-5))。中文 BERT 前端對時長的主要貢獻,是更準地預測韻律邊界([arXiv 2012.15404](https://arxiv.org/pdf/2012.15404);[Sogou Blizzard 2020](https://www.isca-archive.org/vccbc_2020/meng20_vccbc.pdf))。

入聲有兩個特殊問題:**實現方式本身不穩定**,而且**邊界最難對**。四縣客語的兩個入聲「短促」,以 /p t k/ 收尾([CLCLP O09-6](https://aclanthology.org/O09-6.pdf));入聲音節以不除阻塞音結尾,時長較短([Hakka command-response model](https://www.researchgate.net/publication/221483434_Modeling_tones_in_hakka_on_the_basis_of_the_command-response_model))。台灣閩南語的 EGG 研究發現,**超過 80% 的 /ʔ/ 韻尾、不到 20% 的 /p t k/ 韻尾被刪除**,刪除後母音出現**補償性延長**([Pan, JIPA](https://www.cambridge.org/core/journals/journal-of-the-international-phonetic-association/article/glottalization-of-taiwan-min-checked-tones/8E06F7AE20E4F2D59F4B377D34794BC0));韻尾弱化也造成入聲整體變長,屬於進行中的音變([Pan, Interspeech 2016](https://www.isca-archive.org/interspeech_2016/pan16_interspeech.pdf);[Interspeech 2021](https://www.isca-archive.org/interspeech_2021/pan21b_interspeech.pdf))。粵語入聲與非入聲的差異「主要是時長」([Modeling Tones in Continuous Cantonese](https://www.researchgate.net/publication/221488070_Modeling_Tones_In_Continuous_Cantonese_Speech))。客語腔調之間也存在「音節長度韻律」的差異([ROCLING 2023.54](https://aclanthology.org/2023.rocling-1.54.pdf)),片語末延長會讓 ASR 產生錯誤([ROCLING 2025.58](https://aclanthology.org/2025.rocling-main.58.pdf))。【推論】客語 -b/-d/-g 若也有類似的弱化或刪除,三件事會同時發生:這部分變異無法從文字預測,會壓低 ρ 的天花板;母音→塞音的邊界變得模糊,MFA 雜訊升高;弱化與刪除時的補償性延長,會讓「母音核」的定義本身不穩定。研究者找不到客語入聲時長均值/標準差或韻尾刪除率的聲學研究,也找不到任何 Sinitic 語言中「入聲 vs 舒聲」對齊精度的比較。

客語/台語 TTS 文獻**完全沒有時長準確度報告**:2006 年的 RNN 韻律、NTUST 海陸碩論、VoxHakka 都只報 MOS 或可懂度([VoxHakka](https://arxiv.org/html/2409.01548v1);[NTUST 碩論](https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi?o=dnclcdr&s=id%3D%22090NTUST392001%22.&searchmode=basic))。可用資源如下:VoxHakka 約 **180.53 h** 網路收集的轉寫客語(四縣與海陸的分腔時數在不同表格不一致,需核對);HAT-Vol1 與 HAT-Vol2 各約 **80 h**、約 100 位語者,含四縣、海陸、南四縣([NYCU IAIS](https://iais.nycu.edu.tw/en/news_detail43.html);[FSW 2025](https://sites.google.com/nycu.edu.tw/fsw/home/challenge-2025?authuser=0%EF%BC%89%E3%80%82))。在 HAT 上,腔調感知的 RNN-T 讓相對錯誤率下降 57.00%(漢字)與 40.41%(拼音),支持跨腔共享模型時明確輸入腔調條件([arXiv 2602.22522](https://arxiv.org/abs/2602.22522))。台語 ASR 的 T-SANDHI 顯示,應以**變調後的表層聲調**而非本調作為條件([arXiv 2609.18194](https://arxiv.org/pdf/2609.18194))。【推論】四縣與海陸的調值大致呈鏡像,跨腔遷移時聲調嵌入應設計為「腔 × 調」,不要讓兩腔共用同一個調嵌入。

## 0.8–0.9 只在量測、單位或條件改變時達得到

綜合上面的證據,下表整理不同設定下的合理 ρ 區間。**表中的區間全部是【推論】**,錨點是你的實測值與前述文獻。

| 設定 | 合理 ρ 區間 | 主要依據 |
|---|---|---|
| 現況:純文字、確定性、入聲核、雙邊 MFA、單次參考 | 0.74–0.80 | 你的 1 h/48 h 結果;文獻母音/韻母 0.62–0.79 |
| 合成端改讀模型輸出時長 | 約 0.76–0.83(若 rel_syn 為 0.8–0.9) | 移除一個衰減因子;也可能因共享偏誤消失而不升 |
| 再加上參考端用客語重訓的對齊器或人工校正 | 約 0.80–0.85 | 成都話方言對齊器的增益;衰減算術 |
| 參考改用 2–3 次朗讀平均 | 0.85–0.9 有機會 | Spearman-Brown;取決於 ρ_tt |
| 衰減校正後的「真 ρ」 | 0.85–0.95 | 只用於詮釋,不可當觀測值宣傳 |
| 單位改成韻母/音節,或入聲與舒聲混合 | 0.85–0.9 | 全音素 0.88、聲母 0.889 這類「身分變異」效應 |
| 允許參考語音或音素級韻律條件 | 可 >0.9 | T5Gemma-TTS 0.92;但任務已變成「跟讀」 |

依成本效益,建議分四級執行。

**第一級:幾乎零成本,一週內完成,先把量測做對。**一、合成端直接讀取長度調節器實際使用的整數幀時長,換算成秒後與參考比較,同時保留舊的「MFA 重對齊合成音」數字,兩者並列。兩者的差距本身就估出 rel_syn 與共享偏誤的方向。二、報告拆細:入聲、舒聲、合併各算一次 ρ;分別給 log 時長 Pearson 與 Spearman;排除停頓與句末靜音;剔除明顯的對齊失敗(例如異常長的母音核),並揭露剔除數量;所有比較都用同一批核做**配對 bootstrap 95% 信賴區間**,不然 +0.01–0.02 這類差異無從判斷。三、入聲同時以「母音核」與「韻母(核+塞音尾)」或整音節為單位報告,迴避不穩定的母音→塞音邊界。四、對殘差做分組分析:片語末 vs 非片語末、聲調、韻尾類型、後接音節、MFA 對齊信心。若殘差集中在片語末,下一步就是邊界特徵;若集中在入聲韻尾附近,下一步就是對齊。

**第二級:低成本,數天到兩週,量出真正的天花板。**五、人工校正 200–300 個入聲母音核的邊界,另取少量合成音核做同樣校正,計算 MFA 與人工的相關以估 rel_ref 與 rel_syn,再做衰減校正。費用約是數小時的標註工。六、重複朗讀上限實驗:請同一位發音人把 50–100 句測試句再錄 1–2 次,同樣用 MFA 對齊,計算 ρ_tt 與 √ρ_tt。若 √ρ_tt 本來就只有 0.8 左右,0.9 這個目標就應改寫成「達到上限的 95%」,或改用多次朗讀的平均作參考。七、補齊顯式特徵:表層(變調後)聲調、韻尾類型(-b/-d/-g、鼻尾、開尾)、詞/片語邊界與標點、片語內位置、右鄰音節特徵;並考慮入聲專用的輸出頭或混合成分,對應泰語分類建模降 23% 誤差的做法。可以先看 GBM 的特徵重要度,找出神經模型還沒吃到的特徵。

**第三級:中等成本,數週,同時改善標籤與資料。**八、用 HAT-Vol1/Vol2 與 VoxHakka 重新訓練客語 MFA 聲學模型,或微調 XLS-R/MMS frame classifier;以第五項的人工集比較邊界精度,再重抽 1 h 的訓練標籤與測試參考。這一步同時淨化訓練目標與評估參考,是模型面之外最可能帶來實質提升的一步。九、在對齊後的 HAT/VoxHakka 上預訓練多語者時長預測器,輸入語者嵌入與「腔 × 調」嵌入,再在 1 h 資料上微調;合理預期是把 1 h 拉到接近 48 h 的水準(+0.03–0.06)。十、加入韻律邊界預測器的輸出,或中文 BERT/XPhoneBERT 特徵,上採樣後串接進時長預測器;預期增益 +0.02–0.05,未經實證。十一、把時長預測器改成高斯 NLL 均值+變異輸出頭,推論時取均值。

**第四級:不建議為了 ρ 投入。**包括 flow-matching 或隨機時長(Matcha 在朗讀語料上已證實無顯著差異)、變速擴增、用 LLM-TTS 產生的偽時長做蒸餾(教師的節奏不是道地客語)、更多整句級的語速或嵌入條件(你的 oracle 實驗與 2406.19243 都顯示這條路頂多 +0.03)。唯一例外是應用場景本來就允許輸入參考語音或人工韻律標記,例如有真人示範的教材朗讀。這時加入音素級韻律條件可以把 ρ 推過 0.9,但報告時必須說清楚衡量的是「跟讀」而非「從文字預測」。

## 結論

ρ 0.74 → 0.9 的瓶頸更可能在**量測的信度**,而不在**模型的能力**。兩組證據互相印證:文獻顯示母音層級的開放預測很少超過 0.8;你的 GBM、48 h 與 oracle 語速實驗則顯示,文字與整句資訊大致已被用盡。因此,與其直接追求 0.9,下一步應把目標改寫成可驗證的形式:「觀測 ρ / √ρ_tt」,也就是達到這位發音人重複朗讀上限的比例,並用模型輸出時長、人工校正參考、分入聲/舒聲、配對信賴區間來報告。若重複朗讀實驗顯示 √ρ_tt 高於 0.9,而校正後仍有明顯缺口,才值得投入對齊器重訓與邊界特徵;若 √ρ_tt 本來就在 0.8 附近,現有系統其實已接近天花板,正確的作法是改定義,而不是改模型。

另一個值得提出的方向是:入聲韻尾弱化、刪除與補償性延長,在台灣閩南語已有明確記錄,在客語卻沒有聲學數據。你的系統若能區分「韻尾保留 vs 弱化」的時長,對客語語音學本身也會是一筆新的實證。這也提示在客語 TTS 中,入聲母音核時長很可能不是一個完全由文字決定的量,評估指標應該接受這部分的不確定性。
