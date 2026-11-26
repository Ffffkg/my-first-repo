// ICEIC 2027 投稿後研究進度報告(Word)。數字全部來自實驗 log;圖由 make_figs_202611.py 產生。
//   NODE_PATH=/opt/node22/lib/node_modules node build_progress_doc_202611.js <圖片資料夾> <輸出.docx>
// 格式:A4、微軟正黑體 + Calibri、全形標點、不用斜體、頁尾頁碼、縮寫第一次出現時寫出全名。
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, AlignmentType,
  BorderStyle, ShadingType, LevelFormat, HeadingLevel, Footer, Header, PageNumber, VerticalAlign, TableLayoutType,
} = require("docx");

const FIG = process.argv[2] || ".";
const OUT = process.argv[3] || "progress.docx";
const EA = "Microsoft JhengHei", LAT = "Calibri";
const FONT = { ascii: LAT, hAnsi: LAT, eastAsia: EA, cs: LAT };
const NAVY = "1F2A44", GRID = "BFC5CF", MUTED = "5A6372", RED = "B5262C", LIGHT = "EEF1F5";
const W = 9070;   // A4 寬 11906 − 左右邊界各 1418

// ---------- 小工具 ----------
// 文字中的 **粗體** 轉成粗體 run;不使用斜體
function runs(text, base = {}) {
  return text.split(/(\*\*[^*]+\*\*)/).filter((t) => t).map((t) => {
    const bold = t.startsWith("**");
    return new TextRun({ text: bold ? t.slice(2, -2) : t, bold: bold || !!base.bold, italics: false, font: FONT,
                         size: base.size || 22, color: base.color });
  });
}
const P = (text, opt = {}) => new Paragraph({ children: runs(text, opt), spacing: { after: 120, line: 320 },
  alignment: opt.align || AlignmentType.JUSTIFIED, indent: opt.indent ? { firstLine: 440 } : undefined, keepNext: opt.keepNext });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })] });
const B = (t, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, children: runs(t),
  spacing: { after: 60, line: 300 }, alignment: AlignmentType.LEFT });
const N = (t, ref = "steps") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t),
  spacing: { after: 60, line: 300 }, alignment: AlignmentType.LEFT });
const caption = (t, keepNext = false) => new Paragraph({ alignment: AlignmentType.CENTER, keepNext,
  spacing: { before: 60, after: 160 }, children: runs(t, { size: 20, color: MUTED, bold: false }) });
const tcap = (t) => new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 80 },
  children: runs(`**${t}**`, { size: 20 }) });
const note = (t) => new Paragraph({ children: runs(t, { size: 18, color: MUTED }), spacing: { before: 40, after: 160 } });

const border = { style: BorderStyle.SINGLE, size: 4, color: GRID };
const borders = { top: border, bottom: border, left: border, right: border };
function table(head, rows, widths, opt = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, w, isHead, ri) => new TableCell({
    width: { size: w, type: WidthType.DXA }, borders, verticalAlign: VerticalAlign.CENTER,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    shading: isHead ? { fill: NAVY, type: ShadingType.CLEAR, color: "auto" }
      : (opt.zebra && ri % 2 === 1 ? { fill: LIGHT, type: ShadingType.CLEAR, color: "auto" } : undefined),
    children: [new Paragraph({ alignment: opt.align && !isHead ? opt.align : AlignmentType.LEFT, spacing: { after: 0, line: 280 },
      children: runs(String(text), { size: opt.size || 19, bold: isHead, color: isHead ? "FFFFFF" : undefined }) })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths, layout: TableLayoutType.FIXED,
    rows: [new TableRow({ tableHeader: true, cantSplit: true, children: head.map((h, i) => cell(h, widths[i], true, 0)) }),
      ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, widths[i], false, ri)) }))],
  });
}
function figure(file, wpx, hpx, cap) {
  return [new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 40 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(FIG, file)), transformation: { width: wpx, height: hpx },
      altText: { title: cap, description: cap, name: file } })] }), caption(cap)];
}
const gap = () => new Paragraph({ spacing: { after: 60 }, children: [] });

// ---------- 內容 ----------
const C = [];
// 標題區
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600, after: 120 },
  children: [new TextRun({ text: "ICEIC 2027 投稿後研究進度報告", bold: true, size: 40, font: FONT, color: NAVY })] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
  children: [new TextRun({ text: "客語一小時語音合成：對齊器只用一小時、換成端對端架構，以及後續研究方向", size: 24, font: FONT, color: MUTED })] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: runs("Chien-Chih Fang｜中原大學 智慧運算與大數據碩士學位學程", { size: 21 }) }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 360 }, children: runs("報告期間：2026 年 11 月 6 日至 11 月 22 日｜撰寫日期：2026 年 11 月 22 日", { size: 21, color: MUTED }) }));

// 摘要
C.push(H1("摘要"));
C.push(P("本報告整理 ICEIC（International Conference on Electronics, Information, and Communication）2027 論文於 2026 年 11 月 6 日投稿之後，所完成的三組實驗與一項文獻調查。論文提出的 XAD 方法（結合跨腔初始化、對齊監督時長與放大的時長預測器，詳見 1.2 節），是在只有 1 小時目標腔錄音的條件下，改善客語文字轉語音（TTS，Text-to-Speech）系統的入聲時長。投稿後的工作要回答三個問題：方法是否真的只需要 1 小時目標腔資料、方法能否推廣到其他架構，以及接下來最值得做的方向。", { indent: true, align: AlignmentType.LEFT }));
C.push(B("**實驗一（對齊器只用 1 小時）：**連強制對齊器都只用 1 小時資料訓練，XAD 仍比 Base 高 8.0（四縣）與 13.0（海陸）個百分點；海陸與原本無差異，四縣只少 2.3 點。「目標腔從頭到尾只需 1 小時」的主張成立。"));
C.push(B("**實驗二（端對端架構上的對齊監督時長）：**換成端對端架構 VITS（Variational Inference with adversarial learning for end-to-end Text-to-Speech）後，只加對齊監督時長在四縣使入聲命中率 +2.3，未達顯著，但時長相關係數與音質都有改善。"));
C.push(B("**實驗三（VITS 架構上的 XAD）：**完整 XAD 在四縣顯著有效（+6.5），在海陸則沒有改善（−1.5）；海陸只加對齊監督時長甚至顯著變差（−7.7）。跨腔初始化在兩個腔調都穩定改善音質。偏差分析顯示，差異來自時長預測的精準度，而不是整體偏長或偏短。"));
C.push(B("**文獻調查與後續方向：**整理目前五大類時長建模方式，並評估三個後續方向，建議優先做「把強制對齊結果當成提示，而非標準答案」。"));
C.push(tcap("表 1 投稿後各項結論在兩個腔調的成立情況"));
C.push(table(["結論", "四縣", "海陸", "兩腔一致"], [
  ["Matcha-TTS：XAD 改善入聲時長", "+10.3 ✓", "+12.8 ✓", "一致"],
  ["Matcha-TTS：對齊器只用 1 小時，XAD 仍改善入聲時長", "+8.0 ✓", "+13.0 ✓", "一致"],
  ["VITS：XAD 改善入聲時長", "+6.5 ✓", "−1.5", "不一致"],
  ["VITS：只加對齊監督時長改善入聲時長", "+2.3", "−7.7（顯著變差）", "不一致"],
  ["VITS：XAD 優於只加對齊監督時長", "+4.2 ✓", "+6.1 ✓", "一致"],
  ["VITS：跨腔初始化改善音質（梅爾倒頻譜失真 MCD，Mel-Cepstral Distortion，降低，dB）", "−0.095 ✓", "−0.222 ✓", "一致"],
], [4170, 1500, 1900, 1500], { zebra: true }));
C.push(note("數字為入聲命中率的差（百分點）或 MCD 的差（dB），三個訓練種子平均；✓ 表示 95％ 信賴區間不含 0，即差異顯著。"));

// 一、背景
C.push(H1("一、背景：ICEIC 2027 論文回顧"));
C.push(H2("1.1 研究問題"));
C.push(P("客語的入聲字以不除阻的 -b、-d、-g 結尾，母音短而急促。合成語音時，入聲母音的長度若不正確，聽起來就不像入聲，整句的節奏也會不自然。目前的非自迴歸 TTS 由時長預測器決定每個音素佔幾個音框（1 個音框 = 256 ÷ 22,050 ≈ 11.6 毫秒），但目標腔只有 1 小時資料時，模型自己學出的時長並不可靠。", { indent: true }));
C.push(H2("1.2 方法：XAD"));
C.push(P("XAD 以 Matcha-TTS 為基礎，由三個部分組成：", { indent: true }));
C.push(B("**跨腔初始化（X，cross-variety initialization）：**先載入另一腔用約 48 小時資料訓練的模型，再用目標腔 1 小時資料訓練。"));
C.push(B("**對齊監督時長（AD，alignment-supervised duration）：**用蒙特婁強制對齊器（MFA，Montreal Forced Aligner）切出的音素時長，取代模型訓練時自己找的對齊，也就是單調對齊搜尋（MAS，Monotonic Alignment Search）。"));
C.push(B("**放大的時長預測器（DP，duration predictor）：**卷積核由 3 改為 5、通道數由 256 改為 512。"));
C.push(P("Base 是原始的 Matcha-TTS（使用 MAS、原尺寸時長預測器、從零訓練）。所有系統都訓練固定的 40,000 步，使用三個訓練種子（1234、123、2026）。", { indent: true }));
C.push(H2("1.3 論文主要結果"));
C.push(tcap("表 2 ICEIC 論文的主要結果（1 小時目標腔資料，三個種子平均）"));
C.push(table(["腔調", "系統", "入聲命中率（%）", "舒聲命中率（%）", "入聲 ρ", "MCD 差（dB）", "聲調辨識（%）"], [
  ["四縣", "Base", "71.0", "63.7", "0.535", "—", "64.3"],
  ["四縣", "+AD", "77.0", "70.4", "0.651", "−0.12", "69.0"],
  ["四縣", "XAD", "**81.3**", "**74.0**", "**0.738**", "**−0.33**", "**73.0**"],
  ["四縣", "48.0 小時 Base（參考）", "81.1", "73.9", "0.767", "—", "—"],
  ["海陸", "Base", "69.4", "54.9", "0.576", "—", "62.1"],
  ["海陸", "+AD", "77.7", "58.9", "0.676", "−0.10", "64.2"],
  ["海陸", "XAD", "**82.2**", "**65.9**", "**0.743**", "**−0.24**", "**67.3**"],
  ["海陸", "47.2 小時 Base（參考）", "83.0", "68.6", "0.797", "—", "—"],
], [700, 2070, 1300, 1300, 900, 1300, 1500], { zebra: true }));
C.push(note("MCD 差為與同腔 Base 相比；真人錄音的聲調辨識率為 71.5％（四縣）。"));
C.push(H2("1.4 投稿後要回答的問題"));
C.push(N("論文中 MFA 是用整個腔調的錄音訓練的（論文中的補充檢查改用訓練集，仍有數十小時）。如果目標腔真的只有 1 小時，對齊器的品質是否足夠？（實驗一）", "q"));
C.push(N("論文只用 Matcha-TTS 一種架構。換成端對端的 VITS，方法是否仍然有效？兩種架構差多少？（實驗二、實驗三）", "q"));
C.push(N("接下來最值得做的方向是什麼？（文獻調查與方向評估）", "q"));

// 二、評估方式
C.push(H1("二、評估方式"));
C.push(P("所有實驗都沿用論文的評估流程，以便直接與論文結果比較。", { indent: true }));
C.push(tcap("表 3 評估指標"));
C.push(table(["指標", "計算方式", "判讀"], [
  ["入聲命中率（主要指標）", "合成語音再用 MFA 重新對齊，比較每個入聲母音與同一句真人錄音的長度，計算相差在 ±30 毫秒（約 2.6 個音框）內的比例", "越高越好"],
  ["舒聲命中率", "同上，但看非入聲的音節", "越高越好"],
  ["入聲時長相關係數 ρ", "合成與真人入聲母音長度的皮爾森相關係數，反映「哪些長、哪些短」的趨勢是否一致", "越接近 1 越好"],
  ["梅爾倒頻譜失真（MCD，Mel-Cepstral Distortion）", "合成語音與真人錄音的頻譜距離，每句合成 5 次取平均", "越低越好（dB）"],
  ["聲調辨識率", "只用真人訓練語音訓練的聲調分類器，辨識合成語音每個音節的聲調", "越高越好"],
], [2300, 5070, 1700], { zebra: true }));
C.push(B("**測試資料：**每個腔調 100 句測試句（兩位語者各 50 句），與論文相同。"));
C.push(B("**統計：**三個訓練種子平均；95％ 信賴區間以重抽測試句計算。區間不含 0，才視為顯著差異。"));
C.push(B("**取整方式：**時長預測值轉成整數音框時，有無條件進位與累加取整兩種方式，每個系統都在驗證集上事先選定，測試集不參與選擇。"));

// 三、實驗一
C.push(H1("三、實驗一：對齊器只用 1 小時"));
C.push(H2("3.1 動機"));
C.push(P("論文中產生時長標籤的 MFA，是用整個腔調約 48 小時的錄音訓練的。這可能被質疑「雖然 TTS 只用 1 小時，但對齊器其實看過幾十小時」。實驗一把 MFA 也限制成只用 1 小時訓練子集，檢驗在真正只有 1 小時資料的情況下，方法是否仍然有效。", { indent: true }));
C.push(H2("3.2 做法"));
C.push(N("只用 1 小時訓練子集（四縣 431 句、海陸 404 句）訓練 MFA，發音詞典與訓練方式和主實驗完全相同；訓練只需約 15 分鐘（中央處理器即可）。", "x4"));
C.push(N("用這個 1 小時 MFA 對齊訓練集與驗證集，轉成音框時長標籤。", "x4"));
C.push(N("用新的標籤重新訓練 +AD 與 XAD，各三個種子，其餘設定與論文完全相同。為了不覆蓋主實驗的時長檔，另外建立獨立的資料目錄。", "x4"));
C.push(N("和論文的 Base、+AD、XAD 及 48 小時 Base 並列評估。", "x4"));
C.push(H2("3.3 時長標籤品質"));
C.push(P("先比較兩種 MFA 對同一批訓練句切出的結果。這裡的「邊界」是相鄰兩個音素的交界時間點；母音時長是結束邊界減去開始邊界。需要注意，這裡沒有人工標註的正確答案，比較的是「1 小時 MFA 和原本的 MFA 差多少」。", { indent: true }));
C.push(tcap("表 4 1 小時 MFA 與主實驗 MFA 的時長標籤比較"));
C.push(table(["項目", "四縣", "海陸"], [
  ["音素邊界相差 ≤ 10 毫秒的比例", "67.4％", "70.8％"],
  ["音素邊界相差 ≤ 20 毫秒的比例", "86.3％", "89.0％"],
  ["母音時長相關係數 r", "0.934", "0.955"],
  ["母音時長平均絕對差", "13.5 毫秒", "10.9 毫秒"],
  ["全部音素時長平均絕對差", "17.5 毫秒", "14.7 毫秒"],
  ["時長標籤可用比例（訓練+驗證）", "99.7％（693／695 句）", "98.7％（681／690 句）"],
], [4270, 2400, 2400], { zebra: true, align: AlignmentType.CENTER }));
C.push(P("約九成的邊界相差不到 2 個音框，母音時長平均只差約 1 個音框。可以這麼接近的原因是：強制對齊事先知道每句話的音素順序，只需要找出邊界位置，比語音辨識簡單得多；1 小時裡每個常見音素都出現數百次以上；而且 MFA 使用參數較少的統計模型，加上錄音條件一致。", { indent: true }));
C.push(H2("3.4 合成結果"));
C.push(...figure("fig1_x4.png", 586, 264, "圖 1 實驗一的入聲命中率（＊ 表示時長標籤來自只用 1 小時訓練的 MFA）"));
C.push(tcap("表 5 換成 1 小時 MFA 後的變化（百分點，括號為 95％ 信賴區間）"));
C.push(table(["比較", "四縣", "海陸"], [
  ["+AD＊ 對 +AD", "−1.5（−3.5～+0.5）", "−0.2（−2.9～+2.3）"],
  ["**XAD＊ 對 XAD**", "**−2.3（−4.3～−0.3）**", "**+0.2（−1.9～+2.5）**"],
  ["XAD＊ 對 Base", "+8.0（+4.8～+11.5）", "+13.0（+9.0～+17.1）"],
  ["XAD＊ 對 XAD 的 MCD 差", "−0.015 dB（−0.024～−0.006）", "−0.030 dB（−0.047～−0.013）"],
  ["XAD＊ 對 Base 的聲調辨識率差", "+8.9（+7.8～+10.0）", "+4.6（+3.5～+5.6）"],
], [3270, 2900, 2900], { zebra: true, align: AlignmentType.CENTER }));
C.push(H2("3.5 小結"));
C.push(B("**海陸完全沒有損失：**XAD＊ 與 XAD 的差為 +0.2，區間含 0。"));
C.push(B("**四縣小幅下降：**少 2.3 點（區間不含 0），但仍比 Base 高 8.0 點，保留約八成進步（8.0 ÷ 10.3 ≈ 78％）。四縣 1 小時 MFA 的標籤品質本來就略差於海陸（r = 0.934 對 0.955），下降幅度也較大，兩者相符。"));
C.push(B("**音質與聲調不受影響：**MCD 兩腔都沒有變差，聲調辨識仍顯著進步。"));
C.push(B("**結論：**從頭到尾只用 1 小時目標腔資料（包括對齊器），XAD 仍然有效，補上了論文最大的限制。"));
C.push(note("補充說明：22 kHz 音檔在前處理時，句首與句尾的靜音是依主實驗 MFA 的邊界裁切的，因此推算裁切起點時沿用主實驗的對齊結果；句子內部的音素時長則完全來自 1 小時 MFA。所有系統使用同一批音檔，比較仍然公平。"));

// 四、實驗二
C.push(H1("四、實驗二：VITS 架構上的對齊監督時長"));
C.push(H2("4.1 為什麼選 VITS"));
C.push(P("VITS（Variational Inference with adversarial learning for end-to-end Text-to-Speech）是廣泛使用的端對端架構，直接由文字產生波形。它和 Matcha-TTS 前半段相同，都由時長預測器決定每個音素的長度，所以 XAD 可以直接套用；後半段則不同：Matcha-TTS 先產生梅爾頻譜，再交給用大量語音預先訓練好的 HiFi-GAN（High-Fidelity Generative Adversarial Network）聲碼器，VITS 的波形解碼器則必須只用這 1 小時資料自己學。", { indent: true }));
C.push(H2("4.2 實作"));
C.push(tcap("表 6 VITS 實驗設定"));
C.push(table(["項目", "設定"], [
  ["程式", "官方 VITS 原始碼，不修改原始碼，只在外面包一層"],
  ["文字前端", "與 Matcha-TTS 相同，共用 MFA 時長檔"],
  ["時長預測器", "與 Matcha-TTS 同結構的確定性時長預測器（卷積核 3、256 通道）；XAD 放大為卷積核 5、512 通道"],
  ["對齊監督時長", "訓練時用 MFA 時長取代 MAS 找出的對齊路徑"],
  ["訓練步數與種子", "1 小時模型固定 20 萬步、三個種子；步數在看到三種子結果前就固定"],
  ["正確性驗證", "推論結果與官方程式逐點相同（差值 0）；對齊搜尋與官方版本逐點相同"],
  ["評估", "與 Matcha-TTS 完全相同的流程、測試句與指標"],
  ["訓練時間", "每步約 0.15 秒，每個 1 小時模型約 8 小時（單張 RTX 5070 Ti）"],
], [2270, 6800], { zebra: true }));
C.push(H2("4.3 試跑（單一種子）"));
C.push(P("先以四縣、種子 1234 試跑 VITS Base 與 VITS +AD，在第 5 萬步與第 10 萬步各評估一次。", { indent: true }));
C.push(tcap("表 7 VITS 試跑結果（四縣，單一種子）"));
C.push(table(["系統", "步數", "入聲命中率（%）", "舒聲命中率（%）", "入聲 ρ"], [
  ["VITS Base", "5 萬", "68.7", "62.1", "0.485"],
  ["VITS +AD", "5 萬", "69.0", "64.8", "0.508"],
  ["VITS Base", "10 萬", "69.7", "61.4", "0.530"],
  ["VITS +AD", "10 萬", "71.2", "64.4", "0.563"],
], [2270, 1400, 1800, 1800, 1800], { zebra: true, align: AlignmentType.CENTER }));
C.push(P("時長預測器本身確實學得更好：入聲母音的預測誤差由 39.0 毫秒降到 25.0 毫秒，預測值的相關係數由 0.371 升到 0.578。但合成語音只進步 +1.5（−4.0～+6.8），未達顯著。由於第 5 萬步到第 10 萬步兩個模型都仍在進步，正式實驗把步數固定為 20 萬步。", { indent: true }));
C.push(H2("4.4 正式結果（三個種子、20 萬步）"));
C.push(tcap("表 8 VITS +AD 的效果（四縣，三個種子平均）"));
C.push(table(["比較", "入聲命中率差", "MCD 差（dB）"], [
  ["VITS +AD 對 VITS Base", "+2.3（−0.4～+5.2）", "−0.089（−0.108～−0.070）"],
  ["Matcha +AD 對 Matcha Base（參考）", "+6.0（+2.8～+9.1）", "−0.119（−0.133～−0.104）"],
  ["VITS Base 對 Matcha Base", "−2.7（−6.2～+0.8）", "+0.084（+0.052～+0.116）"],
], [3570, 2750, 2750], { zebra: true, align: AlignmentType.CENTER }));
C.push(P("VITS Base 與 VITS +AD 的入聲命中率分別為 68.3％ 與 70.6％，入聲 ρ 由 0.502 升到 0.574，舒聲命中率由 61.9％ 升到 64.8％。對齊監督時長在 VITS 上方向相同，音質也顯著改善，但入聲命中率的進步只有 Matcha-TTS 的約四成，且未達顯著。", { indent: true }));

// 五、實驗三
C.push(H1("五、實驗三：VITS 架構上的 XAD"));
C.push(H2("5.1 系統定義"));
C.push(tcap("表 9 VITS 上的四種系統（與 Matcha-TTS 的定義相同）"));
C.push(table(["系統", "時長來源", "初始化", "時長預測器"], [
  ["Base", "MAS（模型自己找）", "從零", "原尺寸"],
  ["+AD", "MFA", "從零", "原尺寸"],
  ["X", "MAS", "另一腔 48 小時 VITS", "原尺寸"],
  ["XAD", "MFA", "另一腔 48 小時 VITS", "放大"],
], [1500, 2700, 2870, 2000], { zebra: true, align: AlignmentType.CENTER }));
C.push(P("跨腔初始化所需的來源模型，是用另一腔約 48 小時資料訓練 50 萬步的 VITS：四縣的系統從海陸模型初始化，海陸的系統從四縣模型初始化。依官方 VITS 的做法，來源模型略過超過 11.6 秒（1,000 音框）的長句，以避免顯示卡記憶體不足；以海陸來源模型為例，略過 19,196 句中的 1,677 句（8.7％）。目標腔 1 小時的訓練資料則完全沒有略過。", { indent: true }));
C.push(H2("5.2 結果"));
C.push(...figure("fig2_arch_hit.png", 586, 234, "圖 2 兩種架構、兩個腔調的入聲命中率（三個種子平均）"));
C.push(tcap("表 10 VITS 四種系統的結果（三個種子平均）"));
C.push(table(["腔調", "系統", "入聲命中率（%）", "舒聲命中率（%）", "入聲 ρ", "入聲命中率差（對 VITS Base）", "MCD 差（dB）"], [
  ["四縣", "Base", "68.3", "61.9", "0.502", "—", "—"],
  ["四縣", "+AD", "70.6", "64.8", "0.574", "+2.3（−0.4～+5.2）", "−0.089 ✓"],
  ["四縣", "X", "70.7", "62.2", "0.571", "+2.4（−0.9～+5.5）", "−0.095 ✓"],
  ["四縣", "XAD", "**74.8**", "**68.4**", "**0.629**", "**+6.5（+3.2～+9.6）**", "−0.076 ✓"],
  ["海陸", "Base", "**75.3**", "54.5", "0.641", "—", "—"],
  ["海陸", "+AD", "67.6", "52.9", "0.596", "**−7.7（−11.3～−4.0）**", "−0.020"],
  ["海陸", "X", "72.9", "55.5", "0.660", "−2.4（−6.1～+1.3）", "−0.222 ✓"],
  ["海陸", "XAD", "73.8", "**57.9**", "**0.673**", "−1.5（−5.9～+2.6）", "−0.217 ✓"],
], [700, 800, 1250, 1250, 900, 2470, 1700], { zebra: true, align: AlignmentType.CENTER, size: 18 }));
C.push(note("✓ 表示 95％ 信賴區間不含 0。四縣 XAD 比 +AD 高 +4.2（+1.3～+6.9）；海陸 XAD 比 +AD 高 +6.1（+2.5～+9.8）。"));
C.push(...figure("fig3_forest.png", 586, 208, "圖 3 與各自 Base 相比的入聲命中率差（線段為 95％ 信賴區間，未碰到虛線 0 才算顯著）"));
C.push(...figure("fig4_mcd.png", 586, 221, "圖 4 與各自 Base 相比的 MCD 差（越低越好）"));
C.push(H2("5.3 兩種架構的比較"));
C.push(B("**時長：**Matcha-TTS 的 XAD 在兩個腔調都大幅改善入聲命中率（+10.3、+12.8）。VITS 的 XAD 只在四縣有效（+6.5），在海陸沒有改善。"));
C.push(B("**音質：**跨腔初始化在 Matcha-TTS 上大幅改善 MCD（四縣 −0.313、海陸 −0.252 dB）。在 VITS 上，四縣只有 −0.095 dB，海陸則有 −0.222 dB。"));
C.push(B("**整體水準：**海陸 VITS Base（75.3％）顯著優於 Matcha Base（69.4％，差 +5.9）；但兩個腔調的 VITS XAD 都明顯低於 Matcha XAD（四縣 −6.5、海陸 −8.4）。海陸 VITS XAD 的音質則與 Matcha XAD 相當（MCD 差 −0.033，未達顯著）。"));
C.push(H2("5.4 偏差分析：為什麼海陸只加對齊監督時長會變差"));
C.push(P("一個可能的原因是系統性偏差，例如母音整體被合成得太短，導致許多音節落在 ±30 毫秒之外。因此計算每個 VITS 系統在波形上的入聲母音平均偏差。", { indent: true }));
C.push(tcap("表 11 VITS 系統的平均偏差與命中率（三個種子平均；負值表示合成得太短）"));
C.push(table(["系統", "四縣入聲偏差", "四縣舒聲偏差", "四縣入聲命中率", "海陸入聲偏差", "海陸舒聲偏差", "海陸入聲命中率"], [
  ["Base", "−12.2", "−11.3", "68.3", "−10.1", "−17.2", "**75.3**"],
  ["+AD", "−12.6", "−5.7", "70.6", "**+1.0**", "−0.7", "**67.6**"],
  ["X", "−7.8", "−5.4", "70.7", "−2.1", "−3.0", "72.9"],
  ["XAD", "**−15.2**", "−7.9", "**74.8**", "+2.3", "−0.9", "73.8"],
], [1170, 1300, 1300, 1350, 1300, 1300, 1350], { zebra: true, align: AlignmentType.CENTER, size: 18 }));
C.push(note("偏差單位為毫秒，入聲命中率單位為 %。"));
C.push(...figure("fig5_bias.png", 470, 260, "圖 5 VITS 各系統的平均偏差與入聲命中率"));
C.push(P("偏差大小和命中率高低沒有對應關係：海陸 +AD 的平均偏差幾乎是 0，命中率卻最低；海陸 Base 母音整體偏短約 10 毫秒，命中率反而最高；四縣 XAD 偏短最多，命中率卻是四縣最高。命中率取決於每個字的誤差有多分散，而不是整體往哪邊偏。海陸 +AD 的入聲 ρ 由 0.641 降到 0.596，也顯示「哪些母音長、哪些短」的判斷變差了。因此，這不是調整整體快慢就能修正的問題，而是 MFA 時長監督在海陸 VITS 上降低了時長預測的精準度。", { indent: true }));
C.push(P("另外，海陸 +AD 的三個種子在驗證集上選到的取整方式不一致（種子 1234 選累加取整，另兩個選無條件進位），偏差因此在 −10 與 +8 毫秒之間變動，代表這個系統的兩種取整方式本來就很接近，不影響上述結論。", { indent: true }));
C.push(H2("5.5 可能原因（推測，尚未驗證）"));
C.push(B("**VITS 的訓練目標與 MFA 的切法不一定相容：**VITS 使用 MAS 時，對齊是模型為了讓聲學特徵最吻合錄音而自己找出的；直接套用 MFA 的邊界，可能與 VITS 內部的聲學表示不一致，而 Matcha-TTS 較能吸收這種差異。"));
C.push(B("**海陸的 MFA 本來就比較難對齊：**論文中海陸對齊失敗而排除的句子為 143 句，四縣只有 42 句；保留下來的句子也可能有較多小錯誤，而 VITS 可能比 Matcha-TTS 對這些錯誤更敏感。"));
C.push(B("**VITS 長時間微調：**VITS 在 1 小時資料上微調 20 萬步，可能把來源模型學到的聲學知識洗掉，這可以解釋四縣跨腔初始化的音質好處較小。"));

// 六、綜合討論
C.push(H1("六、綜合討論"));
C.push(P("綜合三組實驗，可以分成「穩定成立」與「因腔調或架構而異」兩類結論：", { indent: true }));
C.push(B("**穩定成立：**XAD 在 Matcha-TTS 上兩個腔調都大幅改善入聲時長，而且連對齊器都只用 1 小時時依然成立；在 VITS 上，跨腔初始化穩定改善音質，XAD 也穩定優於只加對齊監督時長。"));
C.push(B("**因腔調或架構而異：**換到 VITS 後，MFA 時長監督的效果取決於腔調（四縣小幅改善、海陸反而變差），原因是時長預測的精準度，而非整體偏差。"));
C.push(P("碩士論文中可以這樣陳述：「XAD 的時長改善在 Matcha-TTS 上穩定成立；換到端對端的 VITS，跨腔初始化仍穩定改善音質，但 MFA 時長監督的效果因腔調而異，時長改善未能穩定推廣。」負面結果經過分析，本身也能呈現對方法與結果的批判性理解。", { indent: true }));
C.push(H2("目前的限制"));
C.push(B("每個腔調只有 100 句測試句，信賴區間寬約 ±3～4 個百分點。"));
C.push(B("評估所用的 MFA 是用真人錄音訓練的，用來量合成語音時可能較不準確；目前沒有人工標註的邊界可供比對。"));
C.push(B("聽測需等待指導教授確認倫理相關程序，目前只有客觀指標。"));
C.push(B("VITS 的 XAD 包含放大的時長預測器，但沒有單獨測試「對齊監督時長+放大時長預測器」的效果。"));

// 七、文獻調查
C.push(H1("七、文獻調查：語音時長建模方法"));
C.push(P("與其以「端對端或非端對端」區分，目前文獻更常依「時長如何被表示與學習」來分類。整理如下：", { indent: true }));
C.push(tcap("表 12 時長建模的主要類別"));
C.push(table(["類別", "時長怎麼處理", "代表研究（發表處）", "需要的資料量"], [
  ["1. 顯式時長、確定性預測", "每個音素預測一個長度，以迴歸學習", "FastSpeech 2；Matcha-TTS；本研究的 XAD", "小（1 小時可行）"],
  ["2. 顯式時長、機率式預測", "每個音素預測長度的分布再取樣", "VITS 的隨機時長預測器；流匹配時長[1]（Interspeech 2024）；MaskGIT 時長[2]（Interspeech 2024）", "小到中"],
  ["3. 可微分或自學對齊", "不用外部對齊器，訓練時讓模型自己學出時長", "對齊學習框架[3]（ICASSP 2022）；Parallel Tacotron 2[4]；StyleTTS 2[5]（NeurIPS 2023）", "小到中"],
  ["4. 只預測總長度", "只決定整句長度，每個字的時間由模型內部分配", "E2 TTS[6](SLT 2024);F5-TTS[7](ACL 2025);DiTTo-TTS[8](ICLR 2025);MaskGCT[9](ICLR 2025)", "非常大（約 8 萬至 10 萬小時）"],
  ["5. 自迴歸語言模型", "逐一產生聲音片段，時長是隱含的", "VALL-E[10]（IEEE/ACM TASLP 2025）；可控制時長的 IndexTTS2[11]（AAAI）", "非常大"],
  ["混合：稀疏對齊", "只用強制對齊提供大概位置（錨點），其餘由模型調整", "MegaTTS 3[12]（2025 預印本）", "大"],
], [1800, 2300, 3570, 1400], { zebra: true, size: 18 }));
C.push(note("發表處全名：Interspeech（Annual Conference of the International Speech Communication Association）、ICASSP（IEEE International Conference on Acoustics, Speech and Signal Processing）、NeurIPS（Conference on Neural Information Processing Systems）、SLT（IEEE Spoken Language Technology Workshop）、ACL（Annual Meeting of the Association for Computational Linguistics）、ICLR（International Conference on Learning Representations）、TASLP（IEEE/ACM Transactions on Audio, Speech, and Language Processing）、AAAI（AAAI Conference on Artificial Intelligence）。"));
C.push(H2("與本研究最相關的發現"));
C.push(B("**機率式時長對朗讀語料幫助有限：**與 Matcha-TTS 同一團隊的研究[1]指出，機率式時長對自發性口語與 VITS 有幫助，但對 Matcha-TTS 的朗讀語料沒有顯著差異。本研究的客語語料屬於朗讀語料。"));
C.push(B("**稀疏對齊支持實驗三的觀察：**MegaTTS 3[12]主張只用強制對齊提供大概位置，比直接套用強制對齊更不受時長錯誤影響，與「MFA 時長硬套在海陸 VITS 上反而變差」的發現相呼應。"));
C.push(B("**最佳時長策略因語言而異：**印度語言的研究[13]比較多種時長預測策略，發現最佳策略取決於語言與任務，與本研究「四縣有效、海陸無效」的結果一致。"));
C.push(B("**大型模型在新語言上的時長控制不可靠：**F5-TTS 用於印度語言時會系統性低估時長，需要補上各語言的時長統計[14]。這顯示專門研究入聲時長仍有其價值。"));
C.push(B("**大型模型的少量資料微調：**已有研究用約 10 小時微調 F5-TTS 到新語言[15]，或只訓練少量附加參數做語言轉移[16]；但 1 小時是否足夠，目前沒有文獻。"));

// 八、後續方向
C.push(H1("八、後續方向評估與規劃"));
C.push(H2("8.1 三個候選方向"));
C.push(tcap("表 13 後續方向比較"));
C.push(table(["", "A. 把 MFA 當成提示", "B. 大型預訓練模型微調", "C. 機率式時長"], [
  ["做法", "MAS 照樣自己找對齊，但在 MFA 邊界附近加分，讓對齊盡量靠近 MFA 但可以自己微調", "用 1 小時客語微調 F5-TTS 等大型預訓練模型", "把時長預測器換成流匹配等機率式模型"],
  ["預期效果", "中等，主要在 VITS", "可能最大，也可能失敗", "最小，可能變差"],
  ["風險", "低", "高", "低"],
  ["顯示卡運算時間", "約 2 至 4 天", "約 1 週以上，另需工程處理", "約 2 天"],
  ["與論文主軸", "高：直接回應實驗三的發現", "中：另開研究線", "低"],
  ["新穎性", "高：本研究自己的新方法", "中", "低：已有研究做過"],
  ["建議", "**最適合做**", "留待之後（RAG 展示版本）", "**不建議做**"],
], [1470, 2600, 2500, 2500], { size: 18 }));
C.push(P("**不建議做方向 C 的原因：**文獻已顯示機率式時長對 Matcha-TTS 的朗讀語料沒有顯著改善[1]；而且入聲命中率是拿合成語音與同一句真人錄音比較，機率式時長多出的隨機變化只會讓結果離那一句錄音更遠，直接預測平均值的確定性預測器在理論上就是較好的選擇。", { indent: true }));
C.push(H2("8.2 方向 A 的規劃"));
C.push(P("目前的 +AD 是直接用 MFA 時長取代 MAS。方向 A 改成在 MAS 的對齊分數中加入一項以 MFA 邊界為中心的先驗，權重在驗證集上選擇。權重為 0 時等於原本的 Base，權重很大時接近 +AD，因此在驗證集上最差約等於兩者中較好的一個，最好的情況則兼具兩者優點。", { indent: true }));
C.push(N("在 VITS 的對齊程式中加入 MFA 先驗（只需修改對齊分數的計算，評估流程不變）。", "a"));
C.push(N("先在問題最明顯的海陸 VITS 上做三個種子，並在驗證集上選擇先驗權重，約 1 天即可得到初步結果。", "a"));
C.push(N("若有效，再做四縣，並視情況套用到 Matcha-TTS。", "a"));
C.push(N("成功標準：海陸 VITS 的入聲命中率至少不低於 VITS Base，且保留跨腔初始化在音質上的好處。", "a"));
C.push(H2("8.3 方向 B 的概要"));
C.push(P("方向 B 以 F5-TTS[7]為主要候選，原因是程式碼與預訓練權重公開、官方程式支援微調到新語言，且已有低資源語言的微調研究。它屬於「只預測總長度」的類別，約以 10 萬小時普通話與英文預訓練。主要步驟為：", { indent: true }));
C.push(N("以客語拼音（含聲調數字）作為新的文字符號加入字表，避免模型以普通話讀音處理客語漢字。", "b"));
C.push(N("從預訓練權重開始，以全部參數微調，以及只訓練少量附加參數（例如低秩適應，LoRA，Low-Rank Adaptation）兩種方式微調。", "b"));
C.push(N("以訓練集統計客語每個字的平均時長，用於估計整句總長度，避免新語言時長被低估。", "b"));
C.push(N("合成相同的 100 句測試句，轉成 22.05 kHz 後以現有流程評估。", "b"));
C.push(P("主要風險：1 小時沒有前例；預訓練資料以普通話為主，沒有入聲；F5-TTS 預訓練權重為非商業授權，且由中國團隊開發。非中國的替代選擇（例如 XTTS-v2、StyleTTS 2）各有授權或語言支援上的限制。建議在碩士論文主線完成後，作為 RAG 語音助理的音質展示版本再進行。", { indent: true }));
C.push(H2("8.4 客語語音助理（RAG）規劃"));
C.push(P("所有實驗完成後，開始建置以檢索增強生成（RAG，Retrieval-Augmented Generation）為核心的客語語音助理：", { indent: true }));
C.push(N("**語音輸入：**以自動語音辨識（ASR，Automatic Speech Recognition）把使用者的問題轉成文字（目前評估已使用 Omnilingual ASR）。", "rag"));
C.push(N("**檢索：**從客語知識庫檢索相關資料，候選來源為教育部臺灣客家語常用詞辭典與客家委員會公開資料。", "rag"));
C.push(N("**生成回答：**由大型語言模型（LLM，Large Language Model）根據檢索結果生成回答。", "rag"));
C.push(N("**客語語音輸出：**把回答的漢字轉成客語拼音，再以 XAD 合成。大型語言模型不擅長書寫客語漢字，需要接上現有的發音詞典，這是系統主要的技術難點。", "rag"));
C.push(H2("8.5 建議時程"));
C.push(table(["階段", "內容", "預估時間"], [
  ["1", "方向 A：海陸 VITS 三個種子與驗證集調整權重", "約 2 天"],
  ["2", "方向 A：四縣 VITS，必要時套用到 Matcha-TTS", "約 2 至 3 天"],
  ["3", "整理 VITS 與方向 A 的結果，寫入碩士論文", "約 1 週"],
  ["4", "RAG 語音助理：知識庫蒐集、拼音轉換、系統整合", "約 2 至 3 週"],
  ["5", "（選做）方向 B：F5-TTS 小規模可行性測試", "約 1 週"],
], [900, 5970, 2200], { zebra: true }));

// 附錄
C.push(H1("附錄 A：時間軸"));
C.push(table(["日期", "工作"], [
  ["11 月 6 日", "ICEIC 2027 投稿；開始實驗一（1 小時 MFA 訓練約 15 分鐘）"],
  ["11 月 7 日", "實驗一完成（兩個腔調 × 兩種系統 × 三個種子）"],
  ["11 月 7 日至 8 日", "實驗二試跑：VITS Base 與 VITS +AD（四縣，單一種子，10 萬步）"],
  ["11 月 8 日至 10 日", "實驗二正式版：四縣三個種子、20 萬步"],
  ["11 月 10 日至 14 日", "實驗三（四縣）：海陸 48 小時來源模型、VITS XAD 與 X 各三個種子"],
  ["11 月 14 日至 18 日", "實驗三（海陸）：海陸 VITS Base、四縣 48 小時來源模型、海陸 VITS XAD"],
  ["11 月 18 日至 21 日", "實驗三（海陸）拆解：海陸 VITS +AD 與 X 各三個種子"],
  ["11 月 21 日至 22 日", "偏差分析、文獻調查、後續方向評估"],
], [2300, 6770], { zebra: true }));
C.push(H1("附錄 B：實驗程式"));
C.push(P("所有程式都在 repo 的 experiments/duration_c 資料夾。", { indent: false }));
C.push(table(["程式", "用途"], [
  ["x4_aligner_1h.sh", "實驗一主程式：訓練 1 小時 MFA、產生時長、訓練與評估"],
  ["x4_prepare.py、x4_compare.py", "實驗一的資料準備與標籤品質比較"],
  ["v_train.py", "VITS 訓練（對齊監督時長、跨腔初始化、時長預測器大小、自動續訓）"],
  ["hakka_c/vits_model.py、hakka_c/vits_io.py", "VITS 包裝與評估介面（不修改官方原始碼）"],
  ["v0_vits_pilot.sh", "實驗二：VITS Base 與 VITS +AD 的訓練與評估"],
  ["v1_vits_xad.sh", "實驗三：48 小時來源模型、VITS X 與 XAD 的訓練與評估"],
  ["c2_mfa2durations.py", "MFA 對齊結果轉成音框時長（新增指定對齊結果與裁切推算來源的選項）"],
], [3600, 5470], { zebra: true }));
C.push(H1("附錄 C：遇到的問題與解決方式"));
C.push(table(["問題", "原因", "解決方式"], [
  ["實驗一有 538 句時長無法使用", "音檔是依主實驗 MFA 邊界裁切的，卻用 1 小時 MFA 推算裁切位置", "裁切位置改用主實驗的對齊結果推算；可用句數恢復到 98％ 以上"],
  ["48 小時 VITS 訓練每步 1.41 秒，比 1 小時資料慢約 9 倍", "每個批次長度都不同，顯示卡程式庫對每種新形狀重新測速", "把批次長度補到固定倍數；速度恢復到每步約 0.14 至 0.16 秒"],
  ["NVML（NVIDIA Management Library）初始化失敗、訓練中斷", "顯示卡驅動程式在背景自動更新，版本與執行中的版本不一致", "重新開機，並暫停 NVIDIA 套件的自動更新"],
  ["48 小時 VITS 顯示卡記憶體不足", "長句集中在同一批次", "依官方做法略過超過 11.6 秒的長句；減少記憶體碎片；中斷時自動從存檔續訓"],
], [2700, 3170, 3200], { zebra: true, size: 18 }));

// 參考文獻
C.push(H1("參考文獻"));
const refs = [
  "S. Mehta, H. Lameris, R. Punmiya, J. Beskow, É. Székely, and G. E. Henter, “Should you use a probabilistic duration model in TTS? Probably! Especially for spontaneous speech,” in Proc. Interspeech, 2024.",
  "S. E. Eskimez et al., “Total-duration-aware duration modeling for text-to-speech systems,” in Proc. Interspeech, 2024.",
  "R. Badlani et al., “One TTS alignment to rule them all,” in Proc. IEEE ICASSP, 2022, pp. 6092–6096.",
  "I. Elias et al., “Parallel Tacotron 2: A non-autoregressive neural TTS model with differentiable duration modeling,” in Proc. Interspeech, 2021.",
  "Y. A. Li et al., “StyleTTS 2: Towards human-level text-to-speech through style diffusion and adversarial training with large speech language models,” in Proc. NeurIPS, 2023.",
  "S. E. Eskimez et al., “E2 TTS: Embarrassingly easy fully non-autoregressive zero-shot TTS,” in Proc. IEEE SLT, 2024.",
  "Y. Chen et al., “F5-TTS: A fairytaler that fakes fluent and faithful speech with flow matching,” in Proc. ACL, 2025, pp. 6255–6271.",
  "K. Lee et al., “DiTTo-TTS: Diffusion transformers for scalable text-to-speech without domain-specific factors,” in Proc. ICLR, 2025.",
  "Y. Wang et al., “MaskGCT: Zero-shot text-to-speech with masked generative codec transformer,” in Proc. ICLR, 2025.",
  "C. Wang et al., “Neural codec language models are zero-shot text to speech synthesizers,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 33, 2025.",
  "S. Zhou et al., “IndexTTS2: A breakthrough in emotionally expressive and duration-controlled auto-regressive zero-shot text-to-speech,” in Proc. AAAI.",
  "Z. Jiang et al., “MegaTTS 3: Sparse alignment enhanced latent diffusion transformer for zero-shot speech synthesis,” arXiv:2502.18924, 2025.",
  "I. Pandey, P. Gaikwad, A. Parulekar, and G. Ramakrishnan, “Technical report: Impact of duration prediction on speaker-specific TTS for Indian languages,” arXiv:2507.16875, 2025.",
  "“Cross-lingual TTS for Indian languages: A benchmark and inference-time duration estimation strategies,” arXiv:2610.14826, 2026.",
  "“Phir Hera Fairy: An English fairytaler is a strong faker of fluent speech in low-resource Indian languages,” arXiv:2505.20693, 2025.",
  "Kwon et al., parameter-efficient fine-tuning of F5-TTS for low-resource languages, in Proc. Interspeech, 2025.",
];
refs.forEach((r, i) => C.push(new Paragraph({ spacing: { after: 60, line: 280 }, indent: { left: 480, hanging: 480 },
  children: runs(`[${i + 1}]\t${r}`, { size: 19 }), tabStops: [{ type: "left", position: 480 }] })));
C.push(note("註：[11] 的出版年與卷期、[14] 至 [16] 的完整作者名單，待引用前於原文確認。"));

// ---------- 文件 ----------
const doc = new Document({
  creator: "Chien-Chih Fang",
  title: "ICEIC 2027 投稿後研究進度報告",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, italics: false, font: FONT, color: NAVY },
        paragraph: { spacing: { before: 360, after: 160 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, italics: false, font: FONT, color: NAVY },
        paragraph: { spacing: { before: 240, after: 100 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 440, hanging: 280 } } } }] },
      ...["steps", "q", "x4", "a", "b", "rag"].map((ref) => ({ reference: ref, levels: [{ level: 0, format: LevelFormat.DECIMAL,
        text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 440, hanging: 320 } } } }] })),
    ],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1418, bottom: 1418, left: 1418, right: 1418 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "ICEIC 2027 投稿後研究進度報告", size: 16, color: MUTED, font: FONT })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
      new TextRun({ text: "第 ", size: 18, color: MUTED, font: FONT }),
      new TextRun({ children: [PageNumber.CURRENT], size: 18, color: MUTED, font: FONT }),
      new TextRun({ text: " 頁／共 ", size: 18, color: MUTED, font: FONT }),
      new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 18, color: MUTED, font: FONT }),
      new TextRun({ text: " 頁", size: 18, color: MUTED, font: FONT }),
    ] })] }) },
    children: C,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT); });
