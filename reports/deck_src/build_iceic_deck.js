const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require("/root/.claude/skills/synced/9c2c4fb7-9911-4e7c-a212-1bd5071b37ce_e3c3c317-72b7-40a6-8ca4-a5e39d1971db/pptx/scripts/apply_theme.js");

const OUT = process.argv[2] || "XAD-TTS_ICEIC2027.pptx";
const FONT = "Microsoft JhengHei";
const THEME = {
  name: "XAD-TTS Hakka",
  headFontFace: FONT,
  bodyFontFace: FONT,
  colors: {
    dk1: "1B2333", lt1: "FFFFFF", dk2: "1F2A44", lt2: "F2F4F7",
    accent1: "B5262C", accent2: "3A7D44", accent3: "8C96A3",
    accent4: "4E6A9E", accent5: "C98A1B", accent6: "6B4C8A",
    hlink: "4E6A9E", folHlink: "6B4C8A",
  },
};
// 系統顏色(圖表只能用 hex,與主題一致)
const COL = { base: "8C96A3", x: "6B4C8A", ad: "3A7D44", addp: "2A8C8C", xad_nodp: "C98A1B", xad: "B5262C", full: "4E6A9E", nat: "1F2A44" };
const GRID = "E3E6EB";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "XAD-TTS:一小時客語語音合成";
pres.author = "Chien-Chih Fang";
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
const C = pres.SchemeColor;

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.9, w: 11.7, h: 1.9, fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.05, w: 11.7, h: 0.9, fontSize: 20, color: C.background2, valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "meta", type: "body", x: 0.8, y: 5.6, w: 11.7, h: 1.1, fontSize: 14, color: C.accent3, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.4, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.3, w: 12.1, h: 0.8, fontSize: 32, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "lead", type: "body", x: 0.6, y: 1.1, w: 12.1, h: 0.55, fontSize: 17, color: C.accent1, valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "XAD-TTS|ICEIC 2027", options: { x: 0.6, y: 7.02, w: 6, h: 0.3, fontSize: 10, color: C.accent3, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.02, w: 0.5, h: 0.3, fontSize: 10, color: C.accent3, align: "right" },
});
pres.defineSlideMaster({
  title: "CLOSING_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.4, w: 12.1, h: 0.9, fontSize: 34, bold: true, color: C.background1, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
});

// ---------- helpers ----------
async function iconPng(Icon, color = "#FFFFFF") {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Icon, { color, size: 256 }));
  const buf = await sharp(Buffer.from(svg)).resize(256, 256).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
async function iconCircle(slide, Icon, x, y, d, fill, name) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill }, objectName: `${name} circle` });
  const p = d * 0.26;
  slide.addImage({ data: await iconPng(Icon), x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p, altText: name, objectName: `${name} icon` });
}
function card(slide, x, y, w, h, name, fill = C.background2) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: fill }, objectName: name });
}
function bullets(items, size = 15, color = C.text1) {
  return items.map((t, i) => {
    const runs = Array.isArray(t) ? t : [{ text: t }];
    // 段落屬性(項目符號、段後距)只放在第一個 run;後面的 run 帶段落屬性會讓 pptxgenjs 多寫一個 pPr,項目符號就不見了
    return runs.map((r, j) => ({
      text: r.text,
      options: { ...(j === 0 ? { bullet: { indent: 16 }, paraSpaceAfter: 9 } : {}), fontSize: size, color, bold: !!r.bold,
                 ...(j === runs.length - 1 && i < items.length - 1 ? { breakLine: true } : {}), ...(r.options || {}) },
    }));
  }).flat();
}
function arrow(slide, x, y, w, h, name, color = C.text2) {
  // 負的寬高會產生無效圖形:一律由左到右 / 由上到下畫,方向相反時改用起點箭頭
  const rev = w < 0 || h < 0;
  const opt = rev ? { beginArrowType: "triangle" } : { endArrowType: "triangle" };
  slide.addShape(pres.shapes.LINE, { x: w < 0 ? x + w : x, y: h < 0 ? y + h : y, w: Math.abs(w), h: Math.abs(h),
                                     line: { color, width: 2, ...opt }, objectName: name });
}
function label(slide, text, x, y, w, h, opts = {}) {
  slide.addText(text, { x, y, w, h, isTextBox: true, margin: 0, fontSize: 14, color: C.text1, valign: "middle", ...opts });
}
function chartBase(title, extra = {}) {
  return {
    showTitle: true, title, titleFontFace: "+mn-lt", titleFontSize: 14, titleColor: THEME.colors.dk2,
    catAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 12, catAxisLabelColor: "1B2333",
    valAxisLabelFontFace: "+mn-lt", valAxisLabelFontSize: 11, valAxisLabelColor: "5A6372",
    valGridLine: { color: GRID, size: 0.75 }, catGridLine: { style: "none" },
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontFace: "+mn-lt", dataLabelFontSize: 11, dataLabelColor: "1B2333",
    dataLabelFormatCode: "0.0", valAxisLabelFormatCode: "0",
    showLegend: true, legendPos: "b", legendFontFace: "+mn-lt", legendFontSize: 12, legendColor: "1B2333",
    barGapWidthPct: 60, ...extra,
  };
}

(async () => {
  const S = {};
  // ===== 1. 封面 =====
  pres.addSection({ title: "開場" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "開場" });
  s.addText("XAD-TTS:一小時客語語音合成的跨腔初始化與對齊監督時長", { placeholder: "title" });
  s.addText("ICEIC 2027 投稿內容說明:研究方法與主要結果", { placeholder: "body" });
  s.addText([
    { text: "Chien-Chih Fang、I Elizabeth Cha*、Shin-Shin Kao", options: { breakLine: true } },
    { text: "中原大學 智慧運算與大數據碩士學位學程/應用數學系" },
  ], { placeholder: "meta" });
  s.addNotes("這份簡報說明投稿 ICEIC 2027 的論文內容。論文名稱是 Cross-Variety Initialization and Alignment-Supervised Durations for One-Hour Hakka Speech Synthesis。核心是提出 XAD-TTS 這個訓練做法,讓只有一小時目標腔語音的客語 TTS,時長(特別是入聲)更準確。接下來依序說明:結論、研究動機、方法、實驗設定、結果、穩健性檢查與限制。");

  // ===== 2. 一頁結論 =====
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "開場" });
  s.addText("一頁看懂主要結果", { placeholder: "title" });
  s.addText("只用 1 小時目標腔語音,XAD-TTS 讓入聲時長準確度接近 48 小時資料的模型", { placeholder: "lead" });
  const stats = [
    [fa.FaClock, C.accent1, "81.3% · 82.2%", "入聲母音時長在真人 ±30 ms 內的比例(四縣 · 海陸)", "Base 只有 71.0% · 69.4%"],
    [fa.FaDatabase, C.accent4, "≈ 48 小時", "1 小時資料即接近完整資料 Base 的入聲時長準確度", "完整資料 Base:81.1% · 83.0%"],
    [fa.FaWaveSquare, C.accent2, "−0.33 · −0.24 dB", "梅爾倒頻譜失真(MCD)降低", "三個訓練種子方向一致"],
    [fa.FaMusic, C.accent6, "+8.8 · +5.3", "聲調分類準確率提升(百分點)", "接近真人錄音的分類準確率"],
  ];
  for (let i = 0; i < 4; i++) {
    const [Icon, col, big, lab, sub] = stats[i];
    const x = 0.6 + (i % 2) * 6.2, y = 1.95 + Math.floor(i / 2) * 2.55;
    card(s, x, y, 5.9, 2.3, `stat card ${i + 1}`);
    await iconCircle(s, Icon, x + 0.35, y + 0.4, 0.85, col, `stat ${i + 1}`);
    label(s, big, x + 1.45, y + 0.3, 4.3, 0.85, { fontSize: 32, bold: true, color: col });
    label(s, lab, x + 1.45, y + 1.15, 4.3, 0.6, { fontSize: 15, color: C.text1, valign: "top" });
    label(s, sub, x + 1.45, y + 1.75, 4.3, 0.4, { fontSize: 12, color: C.accent3, valign: "top" });
  }
  s.addNotes("一頁總結。第一,入聲母音的時長命中率(合成語音的母音核時長和真人差在 ±30 毫秒內的比例)四縣從 71.0% 提高到 81.3%,海陸從 69.4% 到 82.2%,三個訓練種子平均。第二,這個數字已經接近用 48 小時(海陸 47.2 小時)完整資料訓練的 Base(81.1%、83.0%),但要注意 XAD 也用了另一腔的來源模型。第三,MCD 降低 0.33 和 0.24 dB。第四,聲調分類準確率提高 8.8 和 5.3 個百分點。");

  // ===== 3. 為什麼時長重要 =====
  pres.addSection({ title: "研究動機" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "研究動機" });
  s.addText("為什麼時長預測這麼重要", { placeholder: "title" });
  s.addText("時長預測器決定每個音要佔幾個音框,解碼器只能照著這個時間配置生成", { placeholder: "lead" });
  const flow = ["輸入符號\n(聲母、母音、韻尾)", "時長預測器\n每個符號幾個音框", "依時長展開\n特徵序列", "解碼器\n生成梅爾頻譜", "聲碼器\n輸出語音"];
  for (let i = 0; i < 5; i++) {
    const x = 0.6 + i * 2.5, hot = i === 1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.95, w: 2.05, h: 1.15, rectRadius: 0.1, fill: { color: hot ? C.accent1 : C.background2 }, line: { color: hot ? C.accent1 : C.background2 }, objectName: `pipeline step ${i + 1}` });
    label(s, flow[i], x, 1.95, 2.05, 1.15, { align: "center", fontSize: 14, bold: hot, color: hot ? C.background1 : C.text1 });
    if (i < 4) arrow(s, x + 2.1, 2.525, 0.35, 0, `pipeline arrow ${i + 1}`);
  }
  const why = [
    [fa.FaRulerHorizontal, C.accent4, "節奏與音段長度", "預測時長決定展開後的長度,直接影響合成語音的節奏與每個音段的長短。"],
    [fa.FaMusic, C.accent6, "聲調", "聲調主要靠母音上的音高曲線辨認;母音長度決定曲線有多少時間展開。"],
    [fa.FaStopCircle, C.accent1, "入聲(checked syllables)", "以不除阻的 -b、-d、-g 結尾,母音通常較短;母音長度是分辨入聲的線索之一。"],
  ];
  for (let i = 0; i < 3; i++) {
    const [Icon, col, h, t] = why[i];
    const x = 0.6 + i * 4.13;
    card(s, x, 3.55, 3.83, 3.1, `reason card ${i + 1}`);
    await iconCircle(s, Icon, x + 0.3, 3.8, 0.75, col, `reason ${i + 1}`);
    label(s, h, x + 1.2, 3.8, 2.5, 0.75, { fontSize: 18, bold: true, color: C.text2 });
    label(s, t, x + 0.3, 4.75, 3.25, 1.7, { fontSize: 15, valign: "top" });
  }
  s.addNotes("Matcha-TTS 這類非自迴歸 TTS 先由時長預測器決定每個輸入符號分到幾個音框,再依這個時間配置生成語音。所以時長預測錯了,後面的解碼器無法完全修正。時長影響三件事:整體節奏;聲調(聲調主要靠母音上的音高曲線,母音長度決定曲線能展開多久);以及客語入聲,入聲字以不除阻的塞音 -b -d -g 結尾,母音通常較短,所以母音長度是辨識入聲的線索之一。");

  // ===== 4. 一小時資料的困難 =====
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "研究動機" });
  s.addText("只有一小時資料時,問題出在哪", { placeholder: "title" });
  s.addText("模型自己學的對齊不可靠;但客語有兩種可以借用的外部資源", { placeholder: "lead" });
  card(s, 0.6, 1.95, 5.9, 4.75, "problem card");
  await iconCircle(s, fa.FaExclamationTriangle, 0.95, 2.25, 0.75, C.accent1, "problem");
  label(s, "原本的做法(Matcha-TTS / MAS)", 1.9, 2.25, 4.4, 0.75, { fontSize: 18, bold: true, color: C.text2 });
  s.addText(bullets([
    "訓練時,單調對齊搜尋(MAS)自己找出「符號 ↔ 音框」的對齊",
    "時長預測器學著重現這些對齊",
    "對齊是由正在訓練的模型、用同一份小資料估出來的",
    [{ text: "只有 1 小時語音:", bold: true }, { text: "對齊和聲學模型都學不穩,對齊的錯誤會傳給時長預測器" }],
  ], 15), { x: 0.95, y: 3.25, w: 5.3, h: 3.3, isTextBox: true, valign: "top", margin: 0 });
  const res = [
    [fa.FaLanguage, C.accent6, "相近的腔調", "四縣與海陸共用許多子音和母音;另一腔有 47–48 小時資料,可以當起點(跨腔初始化)。"],
    [fa.FaAlignLeft, C.accent2, "強制對齊器(MFA)", "從錄音與文字得到音素邊界,時長不依賴 TTS 模型本身;FastSpeech 2 也用這種時長。"],
  ];
  for (let i = 0; i < 2; i++) {
    const [Icon, col, h, t] = res[i];
    const y = 1.95 + i * 2.45;
    card(s, 6.8, y, 5.9, 2.3, `resource card ${i + 1}`);
    await iconCircle(s, Icon, 7.15, y + 0.3, 0.75, col, `resource ${i + 1}`);
    label(s, h, 8.1, y + 0.3, 4.4, 0.75, { fontSize: 18, bold: true, color: C.text2 });
    label(s, t, 8.1, y + 1.1, 4.4, 1.05, { fontSize: 15, valign: "top" });
  }
  s.addNotes("Matcha-TTS 訓練時用 MAS 讓模型自己找符號和音框的對齊,時長預測器再去模仿這些對齊。問題是對齊是正在訓練的模型自己估出來的,資料只有一小時時,對齊和聲學模型都不穩,錯誤會傳給時長預測器。客語有兩個可以借用的資源:一是相近腔調(四縣、海陸共用許多子音母音,另一腔有 47–48 小時資料);二是強制對齊器 MFA,它從錄音和文字給出不依賴 TTS 模型的音素時長。XAD-TTS 就是把這兩個資源結合起來。");

  // ===== 5. 方法總覽 =====
  pres.addSection({ title: "方法" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "方法" });
  s.addText("XAD-TTS 方法總覽", { placeholder: "title" });
  s.addText("X = 跨腔初始化(Cross-variety initialization)、AD = 對齊監督時長(Alignment-supervised Durations)", { placeholder: "lead" });
  const box = (name, text, x, y, w, h, fill, color = C.text1, bold = false) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: fill }, line: { color: fill }, objectName: name });
    label(s, text, x, y, w, h, { align: "center", fontSize: 14, color, bold });
  };
  box("source model", "來源腔模型\n(另一腔 47.2 / 48.0 h)", 0.6, 2.05, 2.9, 1.15, C.background2);
  box("target model", "目標腔 Matcha-TTS\n+ 加大的時長預測器", 4.95, 1.95, 3.4, 1.35, C.text2, C.background1, true);
  box("mfa", "MFA 強制對齊\n→ 音素時長標籤", 9.8, 2.05, 2.9, 1.15, C.background2);
  box("target data", "目標腔訓練資料\n(1 h 語音與文字)", 9.8, 3.75, 2.9, 1.15, C.background2);
  box("test text", "測試文字", 0.6, 5.4, 2.9, 1.15, C.background2);
  box("synth", "合成語音", 4.95, 5.4, 3.4, 1.15, C.background2);
  box("eval", "評估:時長、MCD、\n聲調、ASR\n(對照真人測試語音)", 9.8, 5.4, 2.9, 1.15, C.background2);
  arrow(s, 3.55, 2.62, 1.35, 0, "init arrow", C.accent6);
  label(s, "X 跨腔初始化", 3.5, 1.75, 1.5, 0.75, { fontSize: 13, bold: true, color: C.accent6, align: "center" });
  arrow(s, 9.75, 2.62, -1.35, 0, "duration arrow", C.accent2);
  label(s, "AD 時長監督", 8.4, 1.75, 1.4, 0.75, { fontSize: 13, bold: true, color: C.accent2, align: "center" });
  arrow(s, 11.25, 3.75, 0, -0.5, "data to mfa arrow");
  s.addShape(pres.shapes.LINE, { x: 6.65, y: 4.32, w: 3.1, h: 0, line: { color: C.text2, width: 2 }, objectName: "data to model line" });
  arrow(s, 6.65, 4.32, 0, -0.97, "data to model arrow");
  label(s, "模型訓練", 6.8, 3.6, 1.5, 0.5, { fontSize: 13, color: C.text2 });
  arrow(s, 6.65, 4.45, 0, 0.95, "inference arrow");
  label(s, "訓練後推論\n(不需要 MFA)", 6.8, 4.55, 2.2, 0.8, { fontSize: 13, color: C.text2 });
  arrow(s, 3.55, 5.95, 1.35, 0, "text arrow");
  arrow(s, 8.4, 5.95, 1.35, 0, "eval arrow");
  s.addNotes("XAD-TTS 是 cross-variety initialization with alignment-supervised durations 的縮寫。訓練時有兩條外部資訊進入目標腔模型:左邊,目標模型從另一腔的完整資料模型複製參數當起點(X);右邊,用 MFA 對 1 小時目標腔資料做強制對齊,得到的音素時長取代 MAS,用來監督加大後的時長預測器(AD)。模型本身仍用這 1 小時的語音與文字訓練。訓練完成後合成時不需要 MFA;MFA 只在評估時再用來量測合成語音的時長。");

  // ===== 6. 三個成分 =====
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "方法" });
  s.addText("三個成分的做法", { placeholder: "title" });
  s.addText("跨腔初始化、對齊監督時長、加大的時長預測器", { placeholder: "lead" });
  const comps = [
    [fa.FaExchangeAlt, C.accent6, "X 跨腔初始化", [
      "四縣目標用 47.2 h 海陸模型、海陸目標用 48.0 h 四縣模型當起點",
      "同名同形狀的參數直接複製;語者嵌入與加大的預測器重新初始化",
      "不凍結參數,最佳化器重新開始",
      "合成時改用目標腔的 mel 正規化統計量"]],
    [fa.FaAlignLeft, C.accent2, "AD 對齊監督時長", [
      "每個腔調自訓 MFA 聲學模型與發音詞典,對齊成功率 > 99%",
      "音素區間換成音框數(hop 256、22.05 kHz,約 11.6 ms)",
      "靜音分給停頓符號或 blank,每個符號至少 1 個音框",
      "取代 MAS:當時長損失的目標,也決定特徵展開"]],
    [fa.FaExpandArrowsAlt, C.accent1, "DP 加大時長預測器", [
      "卷積核 3 → 5、通道 256 → 512",
      "兩層卷積可看到的相鄰符號從 5 個變成 9 個",
      "參數量 20.9 M → 22.4 M",
      "取整方式在驗證集上二選一,所有系統(含 Base)同一程序"]],
  ];
  for (let i = 0; i < 3; i++) {
    const [Icon, col, h, items] = comps[i];
    const x = 0.6 + i * 4.13;
    card(s, x, 1.95, 3.83, 4.75, `component card ${i + 1}`);
    await iconCircle(s, Icon, x + 0.3, 2.2, 0.75, col, `component ${i + 1}`);
    label(s, h, x + 1.2, 2.2, 2.5, 0.75, { fontSize: 18, bold: true, color: col });
    s.addText(bullets(items, 14), { x: x + 0.3, y: 3.15, w: 3.3, h: 3.4, isTextBox: true, valign: "top", margin: 0 });
  }
  s.addNotes("X:來源模型是另一腔用 Base 方式在全部訓練資料上從零訓練的模型。四縣目標用海陸 47.2 小時模型,海陸目標用四縣 48.0 小時模型。名稱和形狀相同的參數直接複製,語者嵌入和加大的時長預測器重新初始化,不凍結,最佳化器重新開始;因為複製過來的 mel 統計量屬於來源腔,合成時改用目標腔的統計量。AD:每個腔調自訓 MFA,對齊成功率超過 99%,失敗的四縣 42 句、海陸 143 句排除。音素區間換成音框數,靜音分給停頓符號或 blank,每個符號至少一個音框。這些時長取代 MAS。DP:時長預測器卷積核 3→5、通道 256→512,參數量從 20.9M 變 22.4M。另外,取整方式(無條件進位或累加取整)在驗證集上選,所有系統都用同一個程序,所以不是 XAD 獨有的優勢。");

  // ===== 7. 實驗設定 =====
  pres.addSection({ title: "實驗設計" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "實驗設計" });
  s.addText("實驗設定", { placeholder: "title" });
  s.addText("同一份資料、同樣步數、三個訓練種子,只改變要比較的成分", { placeholder: "lead" });
  card(s, 0.6, 1.95, 4.6, 4.75, "data card");
  await iconCircle(s, fa.FaDatabase, 0.9, 2.2, 0.75, C.accent4, "data");
  label(s, "資料:HAT 語料", 1.85, 2.2, 3.2, 0.75, { fontSize: 18, bold: true, color: C.text2 });
  s.addText(bullets([
    "四縣、海陸各一位專業女聲、一位專業男聲",
    "依提示卡切分 train / val / test,測試文字不出現在同腔訓練",
    "1 h ⊂ 10 h 巢狀子集,男女時長各半",
    "每腔 100 句測試句(每位語者 50 句)",
  ], 14), { x: 0.9, y: 3.15, w: 4.1, h: 3.4, isTextBox: true, valign: "top", margin: 0 });
  s.addTable([
    [{ text: "系統", options: { bold: true, color: "FFFFFF", fill: { color: THEME.colors.dk2 } } },
     { text: "時長來源", options: { bold: true, color: "FFFFFF", fill: { color: THEME.colors.dk2 } } },
     { text: "初始化", options: { bold: true, color: "FFFFFF", fill: { color: THEME.colors.dk2 } } },
     { text: "時長預測器", options: { bold: true, color: "FFFFFF", fill: { color: THEME.colors.dk2 } } }],
    ["Base(原始 Matcha-TTS)", "MAS", "從零", "原尺寸"],
    ["+AD", "MFA", "從零", "原尺寸"],
    [{ text: "XAD(完整方法)", options: { bold: true, color: THEME.colors.accent1 } }, "MFA", "跨腔初始化", "加大"],
    ["完整資料 Base(參考)", "MAS", "從零,48.0 / 47.2 h", "原尺寸"],
  ], { x: 5.5, y: 1.95, w: 7.2, colW: [2.6, 1.2, 1.9, 1.5], fontSize: 14, fontFace: FONT, color: THEME.colors.dk1,
       border: { type: "solid", pt: 0.75, color: GRID }, rowH: 0.5, valign: "middle", objectName: "systems table" });
  card(s, 5.5, 4.75, 7.2, 1.95, "training card");
  s.addText(bullets([
    "Adam(學習率 10⁻⁴、batch 16),固定 40,000 步,評估第 40,000 步的模型",
    "三個訓練種子(1234、123、2026);完整資料 Base 只有一個種子",
    "一次訓練約 2 小時(單張 RTX 5070 Ti);合成 10 步、temperature 0.667",
  ], 14), { x: 5.8, y: 4.95, w: 6.7, h: 1.6, isTextBox: true, valign: "middle", margin: 0 });
  s.addNotes("資料使用客委會 HAT 語料的 TTS 部分,四縣和海陸各有一位專業女聲和男聲。驗證和測試集依提示卡切分,所以測試文字不會出現在同腔的訓練資料。1 小時和 10 小時子集是巢狀的,男女各半。主要比較三個系統:Base 是原始 Matcha-TTS;+AD 只把時長換成 MFA;XAD 是完整方法。另外用完整資料(四縣 48.0、海陸 47.2 小時)訓練的 Base 當高資源參考。全部固定 40,000 步、三個種子,所以差異來自方法本身,不是訓練長短。");

  // ===== 8. 評估方式 =====
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "實驗設計" });
  s.addText("四種評估方式", { placeholder: "title" });
  s.addText("不只看頻譜距離,也直接量合成語音的時長、聲調與可辨識度", { placeholder: "lead" });
  const evals = [
    [fa.FaClock, C.accent1, "音段時長(主要指標)", "合成語音再用同一個 MFA 對齊,比較每個母音核與真人錄音的時長;報告誤差在 ±30 ms(約 2.6 個音框)內的比例,以及入聲時長的 Pearson ρ。"],
    [fa.FaWaveSquare, C.accent2, "頻譜失真 MCD", "WORLD 抽 24 階梅爾倒頻譜(不含代表音量的 c0),DTW 對齊後計算;每句合成 5 次取平均。"],
    [fa.FaMusic, C.accent6, "客觀聲調分類", "只用真人訓練語音訓練的梯度提升分類器,看 F0 曲線、時長、有聲比例與前後音節 F0;以真人測試錄音的準確率當參考。"],
    [fa.FaCommentDots, C.accent4, "語音辨識(ASR)檢查", "Omnilingual ASR(1B)分別轉寫真人與合成語音,比較 CER、入聲韻尾 -p/-t/-k 保留率、聲調記號一致率;copy-synthesis 當參考。"],
  ];
  for (let i = 0; i < 4; i++) {
    const [Icon, col, h, t] = evals[i];
    const x = 0.6 + (i % 2) * 6.2, y = 1.95 + Math.floor(i / 2) * 2.45;
    card(s, x, y, 5.9, 2.3, `evaluation card ${i + 1}`);
    await iconCircle(s, Icon, x + 0.3, y + 0.3, 0.75, col, `evaluation ${i + 1}`);
    label(s, h, x + 1.25, y + 0.3, 4.45, 0.75, { fontSize: 18, bold: true, color: C.text2 });
    label(s, t, x + 1.25, y + 1.05, 4.45, 1.15, { fontSize: 13.5, valign: "top" });
  }
  s.addNotes("時長是主要指標:把合成語音再用同一個 MFA 對齊,比較每個母音核和真人錄音同一個母音核的長度,誤差在 ±30 毫秒內就算命中,另外報告入聲時長的 Pearson 相關。MCD 用 WORLD 的 24 階梅爾倒頻譜、不含 c0、DTW 對齊,每句五次生成平均。聲調用只看聲學特徵的分類器。ASR 用 Omnilingual ASR 分別轉寫真人和合成語音,比較兩份轉寫的一致性。所有 95% 信賴區間都是以句子重抽 5,000 次得到。");

  // ===== 9. 前導實驗:只做跨腔初始化 =====
  pres.addSection({ title: "結果" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "結果" });
  s.addText("前導實驗:只做跨腔初始化", { placeholder: "title" });
  s.addText("驗證損失在 1 小時改善最多,但單看 MCD 結果不一致", { placeholder: "lead" });
  const hdr = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: THEME.colors.dk2 } } });
  s.addTable([
    [hdr("目標腔"), hdr("符號"), hdr("從零"), hdr("初始化"), hdr("Δ [95% CI]")],
    ["四縣", "joint", "6.36", "5.99", "−0.36 [−0.39, −0.34]"],
    ["四縣", "separate", "6.17", "6.08", "−0.09 [−0.12, −0.06]"],
    ["海陸", "joint", "6.46", "6.59", { text: "+0.13 [+0.09, +0.18]", options: { color: THEME.colors.accent1, bold: true } }],
    ["海陸", "separate", "6.48", "6.22", "−0.26 [−0.30, −0.22]"],
  ], { x: 0.6, y: 2.0, w: 7.2, colW: [1.1, 1.3, 1.1, 1.2, 2.5], fontSize: 14, fontFace: FONT, color: THEME.colors.dk1,
       border: { type: "solid", pt: 0.75, color: GRID }, rowH: 0.55, align: "center", valign: "middle", objectName: "table I" });
  label(s, "MCD(dB),1 h、一個種子、取驗證損失最低的 checkpoint", 0.6, 4.85, 7.2, 0.4, { fontSize: 12, color: C.accent3 });
  card(s, 0.6, 5.4, 7.2, 1.25, "how to read card");
  await iconCircle(s, fa.FaInfo, 0.85, 5.62, 0.8, C.accent4, "how to read");
  label(s, "讀法:Δ = 初始化 − 從零,負值代表初始化較好;95% CI 不含 0 才算有差。joint = 聲調黏在母音上,separate = 聲調是獨立輸入。", 1.85, 5.5, 5.8, 1.05, { fontSize: 14 });
  card(s, 8.1, 2.0, 4.6, 4.65, "init findings card");
  s.addText(bullets([
    [{ text: "16 / 20", bold: true }, { text: " 個設定的驗證損失下降,1 h 下降最多(0.08–0.13)" }],
    [{ text: "4 個條件中 3 個", bold: true }, { text: " MCD 下降;海陸 joint 反而上升,主要來自男聲" }],
    [{ text: "後來發現:", bold: true }, { text: "初始化模型的最佳 checkpoint 都在前 2,000 步,太早" }],
    [{ text: "改用主實驗規則(第 40,000 步)後,", bold: true }, { text: "只做初始化就讓海陸 MCD 下降 0.25 dB" }],
  ], 15), { x: 8.4, y: 2.25, w: 4.1, h: 4.2, isTextBox: true, valign: "top", margin: 0 });
  s.addNotes("這是 TAAI 階段的前導實驗:只做跨腔初始化。在五個資料量的掃描中,20 個設定有 16 個驗證損失下降,1 小時最多。但用 MCD 看,四個條件只有三個下降,海陸 joint 上升 0.13 dB,主要來自男聲。後來在成分分析中發現,這個表選的是驗證損失最低的 checkpoint,而初始化模型的最低點都落在第 1,000 或 2,000 步,太早了;改用主實驗規則評估第 40,000 步的模型,只做初始化就讓海陸 MCD 下降 0.25 dB。所以初始化單獨用時效果有,但要搭配對齊時長才在時長上最好。");

  // ===== 10. 主結果:入聲時長 =====
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "結果" });
  s.addText("主結果:入聲時長準確度", { placeholder: "title" });
  s.addText("XAD 讓誤差在 ±30 ms 內的入聲母音大幅增加,接近完整資料 Base", { placeholder: "lead" });
  s.addChart(pres.charts.BAR, [
    { name: "Base(1 h)", labels: ["四縣", "海陸"], values: [71.0, 69.4] },
    { name: "+AD(1 h)", labels: ["四縣", "海陸"], values: [77.0, 77.7] },
    { name: "XAD(1 h)", labels: ["四縣", "海陸"], values: [81.3, 82.2] },
    { name: "完整資料 Base", labels: ["四縣", "海陸"], values: [81.1, 83.0] },
  ], { x: 0.6, y: 1.9, w: 7.4, h: 4.85, barDir: "col", barGrouping: "clustered", chartColors: [COL.base, COL.ad, COL.xad, COL.full],
       valAxisMinVal: 60, valAxisMaxVal: 90, valAxisMajorUnit: 10, objectName: "checked accuracy chart",
       ...chartBase("入聲母音時長誤差在 ±30 ms 內的比例(%)") });
  card(s, 8.3, 1.9, 4.4, 4.85, "main findings card");
  s.addText(bullets([
    [{ text: "XAD 比 Base:", bold: true }, { text: "+10.3 [6.9, 13.9](四縣)、+12.8 [9.1, 16.5](海陸)" }],
    [{ text: "超出容許範圍的入聲母音:", bold: true }, { text: "29.0% → 18.7%、30.6% → 17.8%(減少約 36%、42%)" }],
    [{ text: "舒聲也進步:", bold: true }, { text: "+10.3、+11.0 個百分點" }],
    [{ text: "時長相關 ρ:", bold: true }, { text: "0.54 → 0.74、0.58 → 0.74(完整資料 0.77、0.80)" }],
  ], 15), { x: 8.6, y: 2.15, w: 3.9, h: 4.4, isTextBox: true, valign: "top", margin: 0 });
  s.addNotes("這是論文最重要的結果。灰色是 Base,綠色 +AD,紅色 XAD,藍色是完整資料 Base。只把時長換成 MFA(+AD)就提升 6.0 和 8.3 個百分點;XAD 再多 4.3 和 4.5,達到 81.3% 和 82.2%,和完整資料 Base 的 81.1% 和 83.0% 很接近。換個角度,超出 ±30 ms 的入聲母音從 29.0% 降到 18.7%(四縣)、30.6% 降到 17.8%(海陸),約少了 36% 和 42%。舒聲字也有相近的進步,所以不只入聲變好。ρ 提高代表合成時長更能跟上真人「哪些母音長、哪些短」的變化,但仍低於完整資料模型。±20 和 ±50 ms 的排名在 1 小時時也相同。");

  // ===== 11. 頻譜與聲調 =====
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "結果" });
  s.addText("主結果:頻譜失真與聲調", { placeholder: "title" });
  s.addText("XAD 的 MCD 降幅是 +AD 的 2–3 倍,聲調分類準確率接近真人錄音", { placeholder: "lead" });
  s.addChart(pres.charts.BAR, [
    { name: "+AD(1 h)", labels: ["四縣", "海陸"], values: [-0.12, -0.10] },
    { name: "XAD(1 h)", labels: ["四縣", "海陸"], values: [-0.33, -0.24] },
  ], { x: 0.6, y: 1.9, w: 5.9, h: 4.25, barDir: "col", barGrouping: "clustered", chartColors: [COL.ad, COL.xad],
       valAxisMinVal: -0.4, valAxisMaxVal: 0, valAxisMajorUnit: 0.1, objectName: "mcd chart",
       ...chartBase("與 Base 的 MCD 差(dB,越低越好)", { dataLabelFormatCode: "0.00", valAxisLabelFormatCode: "0.0" }) });
  s.addChart(pres.charts.BAR, [
    { name: "Base", labels: ["四縣", "海陸"], values: [64.3, 62.1] },
    { name: "+AD", labels: ["四縣", "海陸"], values: [69.0, 64.2] },
    { name: "XAD", labels: ["四縣", "海陸"], values: [73.0, 67.3] },
    { name: "真人錄音", labels: ["四縣", "海陸"], values: [71.5, 67.1] },
  ], { x: 6.8, y: 1.9, w: 5.9, h: 4.25, barDir: "col", barGrouping: "clustered", chartColors: [COL.base, COL.ad, COL.xad, COL.nat],
       valAxisMinVal: 50, valAxisMaxVal: 80, valAxisMajorUnit: 10, objectName: "tone chart",
       ...chartBase("聲調分類準確率(%)") });
  label(s, "聲調:XAD 比 Base +8.8 [7.7, 9.9]、+5.3 [4.4, 6.2] 百分點。分類器也用到母音時長,部分提升可能來自時長變準;準確率接近真人不代表比真人更自然。",
        0.6, 6.25, 12.1, 0.6, { fontSize: 13, color: C.text1 });
  s.addNotes("左圖是 MCD:+AD 降 0.12 和 0.10 dB,XAD 降 0.33 和 0.24 dB,是 +AD 的二到三倍,而且三個種子都有下降。右圖是聲調分類準確率:XAD 比 Base 高 8.8 和 5.3 個百分點,數值接近甚至略高於真人錄音。要誠實說明兩點:第一,分類器的特徵也包含母音時長和前後音節 F0,所以聲調提升有一部分可能來自時長變準;第二,準確率接近真人不代表合成的聲調比真人更自然,只代表比較規律、容易分類。");

  // ===== 12. 成分分析 =====
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "結果" });
  s.addText("成分分析:每個成分貢獻多少", { placeholder: "title" });
  s.addText("兩個主要成分各自有效且互補;加大預測器的貢獻較小", { placeholder: "lead" });
  s.addChart(pres.charts.BAR, [
    { name: "四縣", labels: ["Base", "X", "+AD", "AD+DP", "X+AD", "XAD"], values: [71.0, 76.1, 77.0, 79.4, 79.4, 81.3] },
    { name: "海陸", labels: ["Base", "X", "+AD", "AD+DP", "X+AD", "XAD"], values: [69.4, 78.1, 77.7, 74.7, 80.5, 82.2] },
  ], { x: 0.6, y: 1.9, w: 7.4, h: 4.85, barDir: "col", barGrouping: "clustered", chartColors: [COL.nat, COL.xad_nodp],
       valAxisMinVal: 60, valAxisMaxVal: 90, valAxisMajorUnit: 10, objectName: "ablation chart",
       ...chartBase("入聲母音時長誤差在 ±30 ms 內的比例(%),三種子平均") });
  card(s, 8.3, 1.9, 4.4, 4.85, "ablation findings card");
  s.addText(bullets([
    [{ text: "單獨都有效:", bold: true }, { text: "X +5.1 / +8.7、+AD +6.0 / +8.3(四縣 / 海陸)" }],
    [{ text: "互補:", bold: true }, { text: "X+AD 比 +AD 多 2.4、2.8,比 X 多 3.3、2.4,CI 都不含 0" }],
    [{ text: "MCD 與聲調主要來自 X:", bold: true }, { text: "X 降 0.31 / 0.25 dB,+AD 只降 0.12 / 0.10 dB" }],
    [{ text: "加大預測器較小:", bold: true }, { text: "加在 X+AD 上 +2.0 / +1.7(海陸不顯著);沒有初始化時不一致" }],
  ], 15), { x: 8.6, y: 2.15, w: 3.9, h: 4.4, isTextBox: true, valign: "top", margin: 0 });
  s.addNotes("為了拆開 XAD 的三個成分,另外訓練了兩個系統:X(只做跨腔初始化、時長仍用 MAS)和 AD+DP(對齊時長加大預測器、沒有初始化)。X+AD 是 XAD 去掉加大預測器。結果:兩個主要成分單獨都有效;合在一起比任何一個單獨都好,所以互補,這正是提出 XAD-TTS 的理由。MCD 和聲調的改善主要來自初始化。加大預測器加在 X+AD 上,四縣多 2.0(顯著)、海陸多 1.7(不顯著),MCD 沒有更低;沒有初始化時效果不一致(四縣 +2.4、海陸 −3.0)。這部分照實寫在論文裡。");

  // ===== 13. 資料量 =====
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "結果" });
  s.addText("資料量的影響:1 小時 vs 10 小時", { placeholder: "title" });
  s.addText("入聲 ±30 ms 命中率:資料越少,XAD 的幫助越大;10 小時時差距縮小", { placeholder: "lead" });
  const sizeChart = (name, vals, x, ref) => s.addChart(pres.charts.BAR, [
    { name: "Base", labels: ["1 h", "10 h"], values: vals[0] },
    { name: "+AD", labels: ["1 h", "10 h"], values: vals[1] },
    { name: "XAD", labels: ["1 h", "10 h"], values: vals[2] },
  ], { x, y: 1.9, w: 5.9, h: 4.25, barDir: "col", barGrouping: "clustered", chartColors: [COL.base, COL.ad, COL.xad],
       valAxisMinVal: 60, valAxisMaxVal: 90, valAxisMajorUnit: 10, objectName: `${name} size chart`,
       ...chartBase(`${name}(%;完整資料 Base:${ref})`) });
  sizeChart("四縣", [[71.0, 78.8], [77.0, 79.4], [81.3, 80.8]], 0.6, "81.1");
  sizeChart("海陸", [[69.4, 78.1], [77.7, 81.0], [82.2, 81.6]], 6.8, "83.0");
  label(s, "10 h 時 XAD 比 Base 多 2.0 / 3.5 個百分點,MCD 降 0.24 / 0.16 dB;+AD 的 MCD 不再下降(+0.04 / +0.03 dB)。1 h 與完整資料的 95% CI 約 ±4 個百分點。",
        0.6, 6.25, 12.1, 0.6, { fontSize: 13 });
  s.addNotes("這頁看資料量。1 小時時 XAD 比 Base 多 10 到 13 個百分點;到 10 小時,Base 自己也學得比較準,差距縮小到 2.0 和 3.5。XAD 從 1 小時到 10 小時幾乎持平,四縣 81.3 到 80.8 的微小下降在信賴區間(約 ±4)之內,代表這個指標已經接近上限。10 小時時 XAD 仍降 MCD 0.24 和 0.16 dB,但 +AD 的 MCD 不再下降。結論是這個方法最適合資料極少的情況。完整資料 Base 只有一個種子,只當參考線。");

  // ===== 14. ASR =====
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "結果" });
  s.addText("語音辨識檢查(ASR)", { placeholder: "title" });
  s.addText("整體 CER 差不多,但入聲韻尾與聲調記號和真人更一致", { placeholder: "lead" });
  s.addTable([
    [hdr("系統"), hdr("CER (%) ↓"), hdr("入聲保留 (%) ↑"), hdr("Gap ↑"), hdr("聲調記號 (%) ↑")],
    ["Base(1 h)", "22.2", "77.8", "−4.1", "73.9"],
    ["+AD(1 h)", "21.8", "81.5", "−1.5", "76.7"],
    [{ text: "XAD(1 h)", options: { bold: true, color: THEME.colors.accent1 } }, "21.7",
     { text: "82.6", options: { bold: true } }, { text: "−0.5", options: { bold: true } }, "77.1"],
    ["完整資料 Base", "19.6", "80.1", "−3.7", "77.2"],
    [{ text: "Copy-synthesis(參考)", options: { color: THEME.colors.accent3 } }, { text: "9.8", options: { color: THEME.colors.accent3 } },
     { text: "89.4", options: { color: THEME.colors.accent3 } }, { text: "+0.2", options: { color: THEME.colors.accent3 } }, { text: "88.9", options: { color: THEME.colors.accent3 } }],
  ], { x: 0.6, y: 1.95, w: 7.4, colW: [2.4, 1.15, 1.45, 1.0, 1.4], fontSize: 14, fontFace: FONT, color: THEME.colors.dk1,
       border: { type: "solid", pt: 0.75, color: GRID }, rowH: 0.55, align: "center", valign: "middle", objectName: "asr table" });
  label(s, "兩腔合併 180 句;Gap = 入聲字與其他字的字母相似度差,越接近 0 表示入聲越不吃虧", 0.6, 5.4, 7.4, 0.6, { fontSize: 12, color: C.accent3 });
  card(s, 8.3, 1.95, 4.4, 2.9, "asr findings card");
  s.addText(bullets([
    [{ text: "CER:", bold: true }, { text: "−0.5 [−2.7, +1.9],沒有明顯改善" }],
    [{ text: "入聲韻尾保留:", bold: true }, { text: "+4.8 [1.6, 8.0] 個百分點" }],
    [{ text: "入聲劣勢:", bold: true }, { text: "−4.1 → −0.5(+3.6 [1.2, 6.0])" }],
    [{ text: "聲調記號:", bold: true }, { text: "+3.2 [1.9, 4.4] 個百分點" }],
    [{ text: "只有 +AD 就縮小劣勢,完整資料 Base 沒有:", bold: true }, { text: "改善與對齊時長有關,不只是資料量" }],
  ], 14), { x: 8.6, y: 2.15, w: 3.9, h: 2.6, isTextBox: true, valign: "top", margin: 0 });
  card(s, 8.3, 5.05, 4.4, 1.7, "copy synthesis card");
  await iconCircle(s, fa.FaInfo, 8.55, 5.3, 0.7, C.accent4, "copy synthesis");
  label(s, "Copy-synthesis:真人 mel 直接過同一個聲碼器,代表聲學模型與時長都完美時的參考;XAD 和它仍有差距。", 9.45, 5.2, 3.1, 1.4, { fontSize: 13 });
  s.addNotes("ASR 是另一種評估,不直接用 MFA 的時長量測或聲調分類器。用 Omnilingual ASR 分別轉寫真人和合成語音,比較兩份轉寫。20 句因為真人錄音的辨識結果幾乎不是拉丁字母而排除(四縣 12、海陸 8),剩 180 句,所有系統相同。整體 CER 沒有明顯差異;但入聲韻尾 -p/-t/-k 的保留率提升 4.8 個百分點,入聲字相對其他字的劣勢從 −4.1 縮小到 −0.5,聲調記號一致率提升 3.2。只有 +AD 就縮小了劣勢,而完整資料 Base 沒有,代表入聲的改善和對齊時長有關。和 copy-synthesis(真人 mel 直接過聲碼器)相比還有差距,原因需要進一步研究。");

  // ===== 15. 穩健性 =====
  pres.addSection({ title: "檢查與限制" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "檢查與限制" });
  s.addText("穩健性檢查", { placeholder: "title" });
  s.addText("兩個最可能被審稿人質疑的點,都用額外實驗確認結論不變", { placeholder: "lead" });
  const checks = [
    [fa.FaSearch, C.accent4, "跨腔文本重疊", "四縣 3 句、海陸 10 句的文字也出現在另一腔訓練資料(來源模型可能看過)", "排除後 XAD 增益變動 ≤ 0.2 個百分點,MCD ≤ 0.005 dB"],
    [fa.FaShieldAlt, C.accent2, "MFA 看過測試錄音", "原本 MFA 用全部錄音訓練;改用只以訓練集訓練的 MFA 重新對齊真人與合成語音", "真人時長平均差 5.8 / 5.3 ms(r = 0.98 / 0.99);XAD 增益 10.0 / 11.3(原 10.3 / 12.8),排名不變"],
    [fa.FaSlidersH, C.accent6, "容許範圍的選擇", "主指標用 ±30 ms;另外試 ±20、±40、±50 ms", "1 h 時 Base、+AD、XAD 的排名在 ±20 與 ±50 ms 下相同"],
  ];
  for (let i = 0; i < 3; i++) {
    const [Icon, col, h, what, out] = checks[i];
    const x = 0.6 + i * 4.13;
    card(s, x, 1.95, 3.83, 4.75, `check card ${i + 1}`);
    await iconCircle(s, Icon, x + 0.3, 2.2, 0.75, col, `check ${i + 1}`);
    label(s, h, x + 1.2, 2.2, 2.5, 0.75, { fontSize: 18, bold: true, color: C.text2 });
    label(s, what, x + 0.3, 3.15, 3.25, 1.25, { fontSize: 14, valign: "top" });
    label(s, "結果", x + 0.3, 4.5, 3.25, 0.35, { fontSize: 14, bold: true, color: col });
    label(s, out, x + 0.3, 4.9, 3.25, 1.6, { fontSize: 15, valign: "top", bold: true, color: C.text1 });
  }
  s.addNotes("第一,跨腔文本重疊:主實驗的 100 句測試句中,四縣 3 句、海陸 10 句的文字也出現在另一腔的訓練資料,來源模型可能看過。排除這些句子後,XAD 的入聲增益最多變動 0.2 個百分點,MCD 最多 0.005 dB,所以保留全部句子。第二,原本的 MFA 用了包含測試錄音的全部錄音訓練。改用只以訓練集訓練的 MFA 重新對齊後,真人母音時長平均只差 5.8 和 5.3 毫秒,相關 0.98 和 0.99;XAD 比 Base 的增益是 10.0 和 11.3,原本是 10.3 和 12.8,排名不變。第三,容許範圍改成 ±20 或 ±50 毫秒,1 小時的排名也相同。");

  // ===== 16. 限制 =====
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "檢查與限制" });
  s.addText("限制與誠實揭露", { placeholder: "title" });
  s.addText("論文中照實寫出的限制,以及口試或審稿時可能被問到的點", { placeholder: "lead" });
  const lims = [
    [fa.FaHeadphones, "尚未做聽測", "目前結果都是客觀指標;聽測列為下一步"],
    [fa.FaSeedling, "完整資料參考只有一個種子", "48.0 / 47.2 h 的 Base 只當參考線,不做統計比較"],
    [fa.FaAlignLeft, "MFA 同時用於訓練與評估", "對齊誤差可能兩邊都受影響;所有系統用同一個 MFA,比較仍公平"],
    [fa.FaRobot, "評估工具不完美", "聲調分類器對真人只有 71.5% / 67.1%;ASR 也有辨識錯誤"],
    [fa.FaExpandArrowsAlt, "加大預測器貢獻小", "只在搭配初始化時有小幅幫助,海陸不顯著"],
    [fa.FaHistory, "前導實驗的 checkpoint 太早", "Table I 的初始化模型最佳點在前 2,000 步,低估初始化效果"],
  ];
  for (let i = 0; i < 6; i++) {
    const [Icon, h, t] = lims[i];
    const x = 0.6 + (i % 2) * 6.2, y = 1.95 + Math.floor(i / 2) * 1.6;
    card(s, x, y, 5.9, 1.4, `limitation card ${i + 1}`);
    await iconCircle(s, Icon, x + 0.3, y + 0.32, 0.75, C.accent3, `limitation ${i + 1}`);
    label(s, h, x + 1.25, y + 0.15, 4.5, 0.5, { fontSize: 16, bold: true, color: C.text2 });
    label(s, t, x + 1.25, y + 0.65, 4.5, 0.65, { fontSize: 14, valign: "top" });
  }
  s.addNotes("限制部分。沒有聽測,所以目前都是客觀指標。完整資料參考只有一個種子,只能當參考線。MFA 同時用來產生訓練標籤和評估時長,對齊誤差可能兩邊都有影響,但所有系統都用同一個 MFA,比較是公平的,而且已經用只以訓練集訓練的 MFA 確認過。聲調分類器和 ASR 都不是完美的裁判。加大的時長預測器貢獻小,只在搭配初始化時有幫助。前導實驗 Table I 選的 checkpoint 太早,低估了初始化的效果。");

  // ===== 17. 結論 =====
  pres.addSection({ title: "結論" });
  s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: "結論" });
  s.addText("結論與下一步", { placeholder: "title" });
  const concl = [
    [fa.FaCheckCircle, "1 小時資料,入聲時長接近 48 小時", "XAD 的入聲 ±30 ms 命中率達 81.3% / 82.2%,MCD 與聲調分類也同步改善"],
    [fa.FaLayerGroup, "兩個主要成分互補", "跨腔初始化與對齊時長單獨都有效,結合後最好;合成時不需要 MFA"],
    [fa.FaShieldAlt, "結論經得起檢查", "排除重疊句、改用只以訓練集訓練的 MFA,排名與增益都不變"],
  ];
  for (let i = 0; i < 3; i++) {
    const [Icon, h, t] = concl[i];
    const y = 1.6 + i * 1.3;
    await iconCircle(s, Icon, 0.6, y + 0.1, 0.8, C.accent1, `conclusion ${i + 1}`);
    label(s, h, 1.65, y, 6.4, 0.5, { fontSize: 20, bold: true, color: C.background1 });
    label(s, t, 1.65, y + 0.5, 6.4, 0.7, { fontSize: 14, color: C.background2, valign: "top" });
  }
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.5, y: 1.6, w: 4.2, h: 3.0, rectRadius: 0.12, fill: { color: C.background1, transparency: 88 }, line: { color: C.accent3, width: 1 }, objectName: "next steps card" });
  label(s, "下一步", 8.8, 1.8, 3.6, 0.5, { fontSize: 20, bold: true, color: C.background1 });
  s.addText(bullets([
    "聽測(配對比較),補上主觀評估",
    "投稿:老師以通訊作者身分上傳 EDAS(截止 11 月 6 日)",
    "ICASSP 若先公開,定稿時可引用",
  ], 15, C.background2), { x: 8.8, y: 2.4, w: 3.7, h: 2.05, isTextBox: true, valign: "top", margin: 0 });
  label(s, "Chien-Chih Fang · I Elizabeth Cha* · Shin-Shin Kao|中原大學", 0.6, 6.55, 12.1, 0.4, { fontSize: 12, color: C.accent3 });
  s.addNotes("總結三點:一、只用一小時目標腔資料,XAD 讓入聲時長準確度接近 48 小時模型,MCD 和聲調也一起改善。二、成分分析顯示跨腔初始化和對齊時長單獨都有效、而且互補;合成時不需要 MFA。三、排除重疊句和改用只以訓練集訓練的 MFA 之後,結論都不變。下一步是補聽測,並由老師上傳 EDAS 投稿。");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  // pptxgenjs 會在同一段落的第 2 個以後的 run 前面再寫一個 <a:pPr>(不合規範,而且會把項目符號蓋成 buNone);
  // 段落屬性只該出現在段落開頭,所以把 </a:r> 後面緊接的 pPr 刪掉。
  const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
  const zip = await JSZip.loadAsync(require("fs").readFileSync(OUT));
  let fixed = 0;
  for (const name of Object.keys(zip.files).filter((n) => /^ppt\/slides\/slide\d+\.xml$/.test(n))) {
    const xml = await zip.file(name).async("string");
    const out = xml.replace(/(<\/a:r>)<a:pPr\b[^>]*?(?:\/>|>[\s\S]*?<\/a:pPr>)/g, (m, r) => { fixed++; return r; });
    zip.file(name, out);
  }
  require("fs").writeFileSync(OUT, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
  console.log("removed mid-paragraph pPr:", fixed);
  console.log("wrote", OUT);
})().catch((e) => { console.error(e); process.exit(1); });
