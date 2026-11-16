// ICEIC 2027 投稿後的實驗進度報告(2026 年 11 月):X4、V0、V1。
// 樣式沿用 build_iceic_deck.js(同一組主題色與字型);標點一律全形、不用斜體、每頁有頁碼、縮寫第一次出現時寫出全名。
//   NODE_PATH=<pptxgenjs 所在的 node_modules> node build_progress_202611.js out.pptx
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require(process.env.APPLY_THEME || "/root/.claude/skills/synced/9c2c4fb7-9911-4e7c-a212-1bd5071b37ce_e3c3c317-72b7-40a6-8ca4-a5e39d1971db/pptx/scripts/apply_theme.js");

const OUT = process.argv[2] || "progress_202611.pptx";
const FONT = "Microsoft JhengHei";
const THEME = {
  name: "Hakka TTS Progress",
  headFontFace: FONT,
  bodyFontFace: FONT,
  colors: {
    dk1: "1B2333", lt1: "FFFFFF", dk2: "1F2A44", lt2: "F2F4F7",
    accent1: "B5262C", accent2: "3A7D44", accent3: "8C96A3",
    accent4: "4E6A9E", accent5: "C98A1B", accent6: "6B4C8A",
    hlink: "4E6A9E", folHlink: "6B4C8A",
  },
};
const HEX = { sixian: "4E6A9E", hailu: "3A7D44", matcha: "4E6A9E", vits: "C98A1B", ok: "3A7D44", run: "C98A1B", plan: "8C96A3",
              red: "B5262C", text: "1B2333", muted: "5A6372", grid: "E3E6EB", navy: "1F2A44" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "ICEIC 2027 投稿後實驗進度報告";
pres.author = "Chien-Chih Fang";
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
const C = pres.SchemeColor;
const NUM = { x: 12.2, y: 7.02, w: 0.5, h: 0.3, fontSize: 11, align: "right" };

// ---------- 版面 ----------
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.6, w: 11.7, h: 1.5, fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.3, w: 11.7, h: 1.0, fontSize: 20, color: C.background2, valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "meta", type: "body", x: 0.8, y: 5.2, w: 11.7, h: 1.5, fontSize: 14, color: C.accent3, valign: "top", margin: 0 }, text: "" } },
  ],
  slideNumber: { ...NUM, color: C.accent3 },
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.4, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.3, w: 12.1, h: 0.8, fontSize: 30, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "lead", type: "body", x: 0.6, y: 1.1, w: 12.1, h: 0.55, fontSize: 17, color: C.accent1, valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "客語語音合成進度報告｜2026 年 11 月", options: { x: 0.6, y: 7.02, w: 6, h: 0.3, fontSize: 11, color: C.accent3, margin: 0 } } },
  ],
  slideNumber: { ...NUM, color: C.accent3 },
});

// ---------- 小工具 ----------
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
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: fill }, line: { color: fill }, objectName: name });
}
function label(slide, text, x, y, w, h, opts = {}) {
  slide.addText(text, { x, y, w, h, isTextBox: true, margin: 0, fontSize: 14, color: C.text1, valign: "middle", ...opts });
}
function arrow(slide, x, y, w, h, name, color = C.text2) {
  slide.addShape(pres.shapes.LINE, { x, y, w, h, line: { color, width: 2, endArrowType: "triangle" }, objectName: name });
}
function chip(slide, text, x, y, w, color, name) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.32, rectRadius: 0.08, fill: { color }, line: { color }, objectName: name });
  label(slide, text, x, y, w, 0.32, { fontSize: 12, bold: true, color: C.background1, align: "center" });
}
const head = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: HEX.navy } } });
const TABLE = { fontFace: FONT, color: HEX.text, border: { type: "solid", pt: 0.75, color: HEX.grid }, valign: "middle", margin: 0.05 };
function chartOpts(title, extra = {}) {
  return {
    showTitle: true, title, titleFontFace: "+mn-lt", titleFontSize: 14, titleColor: HEX.navy,
    catAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 12, catAxisLabelColor: HEX.text,
    valAxisLabelFontFace: "+mn-lt", valAxisLabelFontSize: 11, valAxisLabelColor: HEX.muted,
    valGridLine: { color: HEX.grid, size: 0.75 }, catGridLine: { style: "none" },
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontFace: "+mn-lt", dataLabelFontSize: 11, dataLabelColor: HEX.text,
    dataLabelFormatCode: "0.0", valAxisLabelFormatCode: "0",
    showLegend: true, legendPos: "b", legendFontFace: "+mn-lt", legendFontSize: 12, legendColor: HEX.text,
    barGapWidthPct: 55, ...extra,
  };
}
// 森林圖:每列一個點與 95% 信賴區間(用圖形畫,PowerPoint 長條圖沒有可用的誤差線)
function forest(slide, rows, { x, y, w, rowH, lo, hi, labelW, valueW, name }) {
  const px = x + labelW, pw = w - labelW - valueW;
  const sx = (v) => px + ((v - lo) / (hi - lo)) * pw;
  const h = rows.length * rowH;
  for (let t = Math.ceil(lo / 5) * 5; t <= hi; t += 5) {
    slide.addShape(pres.shapes.LINE, { x: sx(t), y, w: 0, h, line: { color: t === 0 ? HEX.muted : HEX.grid, width: t === 0 ? 1.25 : 0.75, dashType: t === 0 ? "dash" : "solid" }, objectName: `${name} grid ${t}` });
    label(slide, (t > 0 ? "+" : t < 0 ? "−" : "") + Math.abs(t), sx(t) - 0.3, y + h + 0.05, 0.6, 0.28, { fontSize: 11, color: HEX.muted, align: "center" });
  }
  rows.forEach((r, i) => {
    const cy = y + (i + 0.5) * rowH;
    label(slide, r.label, x, cy - rowH / 2, labelW - 0.1, rowH, { fontSize: 13, bold: !!r.bold, color: r.color });
    if (r.ci) {
      slide.addShape(pres.shapes.LINE, { x: sx(r.ci[0]), y: cy, w: sx(r.ci[1]) - sx(r.ci[0]), h: 0, line: { color: r.color, width: 2.5 }, objectName: `${name} ci ${i + 1}` });
    }
    const d = 0.17;
    slide.addShape(pres.shapes.OVAL, { x: sx(r.v) - d / 2, y: cy - d / 2, w: d, h: d, fill: { color: r.color }, line: { color: r.color }, objectName: `${name} point ${i + 1}` });
    label(slide, r.text, px + pw + 0.1, cy - rowH / 2, valueW - 0.1, rowH, { fontSize: 12, color: HEX.text, bold: !!r.bold });
  });
}

(async () => {
  let s;
  // ===== 1. 封面 =====
  pres.addSection({ title: "封面" });
  s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "封面" });
  s.addText("ICEIC 2027 投稿後實驗進度報告", { placeholder: "title" });
  s.addText("客語一小時語音合成：對齊器也只用一小時、換成端對端架構後，方法是否仍然有效", { placeholder: "body" });
  s.addText([
    { text: "ICEIC：International Conference on Electronics, Information, and Communication", options: { breakLine: true } },
    { text: "Chien-Chih Fang｜中原大學 智慧運算與大數據碩士學位學程", options: { breakLine: true } },
    { text: "2026 年 11 月 16 日" },
  ], { placeholder: "meta" });
  s.addNotes("這份報告整理 ICEIC 2027 論文投稿之後做的後續實驗。主要回答兩個問題：第一，如果連強制對齊器都只能用一小時的目標腔資料訓練，方法還有沒有效；第二，把方法搬到另一種語音合成架構 VITS，效果如何，以及兩種架構差多少。");

  // ===== 2. 進度總覽 =====
  pres.addSection({ title: "總覽" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "總覽" });
  s.addText("投稿後的實驗進度", { placeholder: "title" });
  s.addText("三項完成、一項進行中、一項規劃中", { placeholder: "lead" });
  card(s, 0.6, 1.8, 12.1, 1.15, "definitions card");
  s.addText([
    { text: "名詞：", options: { bold: true, color: HEX.navy } },
    { text: "文字轉語音（TTS，Text-to-Speech）；XAD = 跨腔初始化（X，cross-variety initialization）+ 對齊監督時長（AD，alignment-supervised duration）+ 放大的時長預測器（DP，duration predictor）", options: { breakLine: true } },
    { text: "指標：", options: { bold: true, color: HEX.navy } },
    { text: "入聲命中率 = 合成語音的入聲母音時長，與真人錄音相差在 ±30 毫秒內的比例（越高越好）" },
  ], { x: 0.85, y: 1.85, w: 11.6, h: 1.05, isTextBox: true, margin: 0, fontSize: 13, color: C.text1, valign: "middle", paraSpaceAfter: 4 });
  // 時間軸
  const steps = [
    { d: "11／6～11／7", id: "X4", t: "對齊器只用 1 小時", q: "蒙特婁強制對齊器（MFA，Montreal Forced Aligner）也只用 1 小時資料訓練", r: "XAD 仍比 Base 高 8.0（四縣）/13.0（海陸）點", st: "完成", c: HEX.ok },
    { d: "11／7～11／10", id: "V0", t: "VITS 試跑", q: "VITS（Variational Inference with adversarial learning for end-to-end TTS）上 +AD 有沒有用", r: "+2.3 點，方向相同但未達顯著", st: "完成", c: HEX.ok },
    { d: "11／10～11／14", id: "V1", t: "VITS 上的 XAD（四縣）", q: "跨腔初始化與 XAD 搬到 VITS 是否仍有效", r: "+6.5 點，顯著；約為 Matcha-TTS 的六成", st: "完成", c: HEX.ok },
    { d: "11／14～11／17", id: "V1", t: "VITS 上的 XAD（海陸）", q: "另一個腔調的結果是否一致", r: "預計 11／17 完成", st: "進行中", c: HEX.run },
    { d: "之後", id: "RAG", t: "客語語音助理", q: "檢索增強生成（RAG，Retrieval-Augmented Generation）結合客語語音合成", r: "實驗全部完成後開始", st: "規劃中", c: HEX.plan },
  ];
  const cw = 2.3, gap = 0.15, ty = 3.3;
  s.addShape(pres.shapes.LINE, { x: 0.6 + cw / 2, y: ty + 0.17, w: 4 * (cw + gap), h: 0, line: { color: HEX.grid, width: 3 }, objectName: "timeline line" });
  steps.forEach((st, i) => {
    const x = 0.6 + i * (cw + gap);
    s.addShape(pres.shapes.OVAL, { x: x + cw / 2 - 0.17, y: ty, w: 0.34, h: 0.34, fill: { color: st.c }, line: { color: "FFFFFF", width: 2 }, objectName: `timeline dot ${i + 1}` });
    label(s, st.d, x, ty - 0.33, cw, 0.3, { fontSize: 12, color: HEX.muted, align: "center" });
    const cy = ty + 0.5;
    card(s, x, cy, cw, 3.05, `step card ${i + 1}`);
    chip(s, st.st, x + 0.15, cy + 0.15, 0.85, st.c, `status ${i + 1}`);
    label(s, st.id, x + 1.1, cy + 0.15, cw - 1.25, 0.32, { fontSize: 13, bold: true, color: HEX.muted, align: "right" });
    label(s, st.t, x + 0.15, cy + 0.55, cw - 0.3, 0.5, { fontSize: 15, bold: true, color: C.text2, valign: "top" });
    label(s, st.q, x + 0.15, cy + 1.05, cw - 0.3, 1.15, { fontSize: 11, color: HEX.muted, valign: "top" });
    label(s, st.r, x + 0.15, cy + 2.2, cw - 0.3, 0.75, { fontSize: 13, bold: true, color: st.c === HEX.plan ? HEX.muted : st.c, valign: "top" });
  });
  s.addNotes("先看整體進度。這段時間做了四組實驗、規劃一項。X4 回答「對齊器也只能用一小時資料」時方法是否仍有效，結論是有效。V0 和 V1 把方法搬到端對端架構 VITS：只有對齊監督時長時效果方向相同但不顯著；完整的 XAD 在四縣顯著有效。海陸的 VITS 實驗正在跑，預計 11 月 17 日完成。之後開始 RAG 客語語音助理。");

  // ===== 3. X4 標籤品質 =====
  pres.addSection({ title: "X4" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "X4" });
  s.addText("X4：只用 1 小時訓練的 MFA，時長標籤品質", { placeholder: "title" });
  s.addText("和主實驗的 MFA（用全部錄音訓練）比較同一批訓練句的音素邊界", { placeholder: "lead" });
  s.addChart(pres.charts.BAR, [
    { name: "四縣", labels: ["邊界差 ≤ 10 毫秒", "邊界差 ≤ 20 毫秒"], values: [67.4, 86.3] },
    { name: "海陸", labels: ["邊界差 ≤ 10 毫秒", "邊界差 ≤ 20 毫秒"], values: [70.8, 89.0] },
  ], { x: 0.6, y: 1.85, w: 6.6, h: 4.85, barDir: "col", chartColors: [HEX.sixian, HEX.hailu], valAxisMinVal: 0, valAxisMaxVal: 100,
       ...chartOpts("兩種 MFA 的音素邊界一致比例（％）"), objectName: "boundary agreement chart" });
  const qs = [
    [fa.FaChartLine, C.accent4, "母音時長相關係數 r", "0.934", "0.955", ""],
    [fa.FaRulerHorizontal, C.accent2, "母音時長平均絕對差", "13.5", "10.9", "毫秒（約 1 個音框 = 11.6 毫秒）"],
    [fa.FaCheckCircle, C.accent6, "時長標籤可用比例", "99.7％", "98.7％", "訓練＋驗證：693／695 句、681／690 句"],
  ];
  for (let i = 0; i < 3; i++) {
    const [Icon, col, lab, a, b, unit] = qs[i];
    const y = 1.85 + i * 1.67;
    card(s, 7.5, y, 5.2, 1.5, `quality card ${i + 1}`);
    await iconCircle(s, Icon, 7.75, y + 0.38, 0.72, col, `quality ${i + 1}`);
    label(s, lab, 8.7, y + 0.12, 3.9, 0.4, { fontSize: 14, bold: true, color: C.text2 });
    label(s, [{ text: "四縣 ", options: { fontSize: 12, color: HEX.muted } }, { text: a, options: { fontSize: 24, bold: true, color: HEX.sixian } },
              { text: "    海陸 ", options: { fontSize: 12, color: HEX.muted } }, { text: b, options: { fontSize: 24, bold: true, color: HEX.hailu } }],
          8.7, y + 0.5, 3.9, 0.6);
    if (unit) label(s, unit, 8.7, y + 1.08, 3.9, 0.32, { fontSize: 11, color: HEX.muted });
  }
  s.addNotes("X4 第一步：主實驗的 MFA 用整個腔調的錄音訓練，這裡改成只用一小時的訓練子集訓練 MFA，再和主實驗的 MFA 比較同一批訓練句的對齊結果。兩腔約七成的音素邊界差在 10 毫秒內、將近九成在 20 毫秒內；母音時長相關係數四縣 0.934、海陸 0.955，平均絕對差約一個音框。轉成時長標籤時，幾乎所有句子都可用。所以一小時訓練的 MFA 標籤品質已經相當接近。");

  // ===== 4. X4 合成結果 =====
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "X4" });
  s.addText("X4：對齊器只用 1 小時，XAD 的進步大多保留", { placeholder: "title" });
  s.addText("三個訓練種子平均；＊ 表示時長標籤來自只用 1 小時訓練的 MFA", { placeholder: "lead" });
  const sysL = ["Base", "+AD", "+AD＊", "XAD", "XAD＊", "48 小時 Base"];
  s.addChart(pres.charts.BAR, [
    { name: "四縣", labels: sysL, values: [71.0, 77.0, 75.4, 81.3, 79.1, 81.1] },
    { name: "海陸", labels: sysL, values: [69.4, 77.7, 77.4, 82.2, 82.4, 83.0] },
  ], { x: 0.6, y: 1.85, w: 6.3, h: 4.85, barDir: "col", chartColors: [HEX.sixian, HEX.hailu], valAxisMinVal: 50, valAxisMaxVal: 90,
       ...chartOpts("入聲命中率（％）", { barGapWidthPct: 45, dataLabelFontSize: 10 }), objectName: "x4 hit rate chart" });
  label(s, "換成 1 小時 MFA 後的變化（百分點，括號為 95％ 信賴區間）", 7.2, 1.85, 5.5, 0.4, { fontSize: 12, bold: true, color: C.text2 });
  s.addTable([
    [head("比較"), head("四縣"), head("海陸")],
    ["+AD＊ 對 +AD", "−1.5（−3.5～+0.5）", "−0.2（−2.9～+2.3）"],
    [{ text: "XAD＊ 對 XAD", options: { bold: true } }, { text: "−2.3（−4.3～−0.3）", options: { bold: true, color: HEX.red } }, { text: "+0.2（−1.9～+2.5）", options: { bold: true } }],
    ["XAD＊ 對 Base", "+8.0（+4.8～+11.5）", "+13.0（+9.0～+17.1）"],
    ["XAD＊ 聲調辨識 對 Base", "+8.9", "+4.6"],
  ], { x: 7.2, y: 2.3, w: 5.5, colW: [1.7, 1.9, 1.9], fontSize: 11, rowH: 0.45, ...TABLE, objectName: "x4 diff table" });
  card(s, 7.2, 4.95, 5.5, 1.75, "x4 takeaway card");
  s.addText([
    { text: "海陸：完全沒有損失", options: { bold: true, color: HEX.hailu, breakLine: true } },
    { text: "四縣：少 2.3 點，仍保留約八成進步", options: { bold: true, color: HEX.sixian, breakLine: true } },
    { text: "梅爾倒頻譜失真（MCD，Mel-Cepstral Distortion）兩腔都沒有變差", options: { color: C.text1 } },
  ], { x: 7.45, y: 5.05, w: 5.05, h: 1.55, isTextBox: true, margin: 0, fontSize: 14, valign: "middle", paraSpaceAfter: 6 });
  s.addNotes("X4 第二步：用一小時 MFA 的時長標籤，重新訓練 +AD 和 XAD，各三個種子。圖中有星號的是 X4 版本。海陸的 XAD 星號版和原本幾乎一樣（+0.2，區間含 0）；四縣少了 2.3 點，區間不含 0，是小但顯著的下降，不過仍比 Base 高 8.0 點，保留大約八成的進步。四縣的標籤品質（r = 0.934）本來就比海陸（0.955）差一點，下降幅度也比較大，這個對應是合理的。MCD 兩腔都沒有變差，聲調辨識也仍然進步。結論：即使從頭到尾只有一小時目標腔資料，方法仍然有效。");

  // ===== 5. VITS 實驗設計 =====
  pres.addSection({ title: "VITS" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "VITS" });
  s.addText("V0／V1：把方法搬到 VITS 架構", { placeholder: "title" });
  s.addText("VITS 是端對端架構：直接由文字產生波形，不使用另外訓練好的聲碼器", { placeholder: "lead" });
  const pipe = (rowY, name, boxes, col) => {
    label(s, name, 0.6, rowY, 1.2, 0.95, { fontSize: 14, bold: true, color: col });
    let x = 1.85;
    boxes.forEach(([t, w, hot], i) => {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: rowY, w, h: 0.95, rectRadius: 0.08, fill: { color: hot ? col : C.background2 }, line: { color: hot ? col : C.background2 }, objectName: `${name} box ${i + 1}` });
      label(s, t, x, rowY, w, 0.95, { fontSize: 12, align: "center", bold: !!hot, color: hot ? C.background1 : C.text1 });
      if (i < boxes.length - 1) arrow(s, x + w + 0.03, rowY + 0.475, 0.19, 0, `${name} arrow ${i + 1}`);
      x += w + 0.25;
    });
  };
  pipe(1.9, "Matcha-TTS", [["文字", 0.6], ["編碼器\n+ 時長預測器", 1.15], ["解碼器\n產生梅爾頻譜", 1.15], ["預訓練聲碼器\nHiFi-GAN", 1.2, true], ["波形", 0.6]], HEX.matcha);
  pipe(3.05, "VITS", [["文字", 0.6], ["編碼器\n+ 時長預測器", 1.15], ["流模型 + 波形解碼器\n只用 1 小時資料學", 2.6, true], ["波形", 0.6]], HEX.vits);
  label(s, "HiFi-GAN：High-Fidelity Generative Adversarial Network，用大量語音預先訓練的聲碼器", 1.85, 4.05, 5.7, 0.35, { fontSize: 11, color: HEX.muted });
  s.addTable([
    [head("系統"), head("時長來源"), head("初始化"), head("時長預測器")],
    ["Base", "單調對齊搜尋（MAS，Monotonic Alignment Search）", "從零", "原尺寸"],
    ["+AD", "MFA", "從零", "原尺寸"],
    ["X", "MAS", "另一腔 48 小時模型", "原尺寸"],
    [{ text: "XAD", options: { bold: true, color: HEX.red } }, "MFA", "另一腔 48 小時模型", "放大"],
  ], { x: 0.6, y: 4.55, w: 7.0, colW: [0.8, 3.1, 1.9, 1.2], fontSize: 12, rowH: 0.42, ...TABLE, objectName: "vits systems table" });
  card(s, 7.9, 1.9, 4.8, 4.8, "vits settings card");
  label(s, "訓練設定（兩個架構盡量一致）", 8.15, 2.05, 4.3, 0.4, { fontSize: 15, bold: true, color: C.text2 });
  const sets = [
    ["程式", "官方 VITS 原始碼，不修改，只在外面包一層"],
    ["文字前端", "與 Matcha-TTS 相同，共用 MFA 時長檔"],
    ["時長預測器", "與 Matcha-TTS 同結構；XAD 放大為卷積核 5、512 通道"],
    ["訓練", "1 小時模型固定 20 萬步、3 個種子（事先決定）"],
    ["來源模型", "48 小時、50 萬步；依官方做法略過 8.7％ 超過 11.6 秒的長句"],
    ["驗證", "推論結果與官方程式逐點相同（差值 0）"],
    ["時間", "每個 1 小時模型約 8 小時（RTX 5070 Ti）"],
  ];
  sets.forEach(([k, v], i) => {
    const y = 2.55 + i * 0.58;
    label(s, k, 8.15, y, 1.15, 0.55, { fontSize: 12, bold: true, color: HEX.vits, valign: "top" });
    label(s, v, 9.3, y, 3.25, 0.55, { fontSize: 12, valign: "top" });
  });
  s.addNotes("VITS 和 Matcha-TTS 最大的差別在後半段：Matcha-TTS 先產生梅爾頻譜，再交給用大量語音預先訓練好的 HiFi-GAN 聲碼器；VITS 是端對端，波形解碼器要用這一小時資料自己學。為了公平，文字前端、MFA 時長檔、時長預測器結構都和 Matcha-TTS 相同，四種系統的定義也一樣。VITS 用官方原始碼，只在外面包一層；已驗證推論結果和官方程式逐點相同。訓練步數 20 萬步、三個種子，事先固定，不是看了測試結果才挑。48 小時來源模型依官方做法略過 8.7％ 超過 11.6 秒的長句，避免顯示卡記憶體不足。");

  // ===== 6. VITS 結果 =====
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "VITS" });
  s.addText("VITS 結果：XAD 同樣有效，但幅度較小", { placeholder: "title" });
  s.addText("四縣 1 小時、三個訓練種子平均", { placeholder: "lead" });
  const cfg = ["Base", "+AD", "X", "XAD"];
  s.addChart(pres.charts.BAR, [
    { name: "Matcha-TTS", labels: cfg, values: [71.0, 77.0, 76.1, 81.3] },
    { name: "VITS", labels: cfg, values: [68.3, 70.6, 70.7, 74.8] },
  ], { x: 0.6, y: 1.85, w: 5.8, h: 4.85, barDir: "col", chartColors: [HEX.matcha, HEX.vits], valAxisMinVal: 50, valAxisMaxVal: 90,
       ...chartOpts("入聲命中率（％）"), objectName: "architecture hit rate chart" });
  label(s, "與各自 Base 的入聲命中率差（百分點，95％ 信賴區間）", 6.7, 1.85, 6.0, 0.4, { fontSize: 13, bold: true, color: C.text2 });
  forest(s, [
    { label: "Matcha +AD", v: 6.0, ci: [2.8, 9.1], text: "+6.0（+2.8～+9.1）", color: HEX.matcha },
    { label: "Matcha XAD", v: 10.3, ci: [6.9, 13.9], text: "+10.3（+6.9～+13.9）", color: HEX.matcha, bold: true },
    { label: "VITS +AD", v: 2.3, ci: [-0.4, 5.2], text: "+2.3（−0.4～+5.2）", color: HEX.vits },
    { label: "VITS X", v: 2.4, ci: [-0.9, 5.5], text: "+2.4（−0.9～+5.5）", color: HEX.vits },
    { label: "VITS XAD", v: 6.5, ci: [3.2, 9.6], text: "+6.5（+3.2～+9.6）", color: HEX.vits, bold: true },
  ], { x: 6.7, y: 2.35, w: 6.0, rowH: 0.5, lo: -5, hi: 15, labelW: 1.3, valueW: 2.3, name: "forest" });
  card(s, 6.7, 5.35, 6.0, 1.35, "vits takeaway card");
  s.addText([
    { text: "VITS 的 +AD、X 單獨使用都接近顯著；合在一起（XAD）+6.5，顯著", options: { breakLine: true } },
    { text: "XAD 比只加 AD 再高 +4.2（+1.3～+6.9）：兩者互補，與 Matcha-TTS 相同", options: { bold: true, color: HEX.vits } },
  ], { x: 6.95, y: 5.45, w: 5.55, h: 1.15, isTextBox: true, margin: 0, fontSize: 13, color: C.text1, valign: "middle", paraSpaceAfter: 6 });
  s.addNotes("左圖是兩個架構四種設定的入聲命中率，右圖是相對於各自 Base 的進步與 95％ 信賴區間（線段不碰到虛線 0，就是顯著）。Matcha-TTS 的 X 只有論文表 III 的平均值，所以不畫區間。VITS 上，+AD 和 X 單獨都只有 +2.3、+2.4，區間剛好碰到 0；合起來的 XAD 是 +6.5，顯著，而且比只加 AD 再高 4.2 點，也顯著。這和 Matcha-TTS 論文裡兩個成分互補的發現一致。不過幅度只有 Matcha-TTS（+10.3）的六成左右。");

  // ===== 7. 架構比較 =====
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "VITS" });
  s.addText("兩個架構的比較", { placeholder: "title" });
  s.addText("時長的改善可以推廣到 VITS；音質的改善主要出現在 Matcha-TTS", { placeholder: "lead" });
  s.addTable([
    [head("四縣 1 小時"), head("Matcha Base"), head("Matcha XAD"), head("VITS Base"), head("VITS XAD")],
    ["入聲命中率（％）", "71.0", { text: "81.3", options: { bold: true } }, "68.3", { text: "74.8", options: { bold: true } }],
    ["舒聲命中率（％）", "63.7", { text: "74.0", options: { bold: true } }, "61.9", { text: "68.4", options: { bold: true } }],
    ["入聲時長相關 ρ", "0.535", { text: "0.738", options: { bold: true } }, "0.502", { text: "0.629", options: { bold: true } }],
    ["MCD 與 Matcha Base 差（dB）", "—", { text: "−0.332", options: { bold: true } }, "+0.084", "+0.009"],
  ], { x: 0.6, y: 1.85, w: 6.6, colW: [2.2, 1.1, 1.1, 1.1, 1.1], fontSize: 12, rowH: 0.5, ...TABLE, align: "center", objectName: "architecture table" });
  const tk = [
    [fa.FaClock, C.accent2, "時長：XAD 在兩個架構都有效", "入聲命中率 +10.3（Matcha-TTS）與 +6.5（VITS）"],
    [fa.FaWaveSquare, C.accent1, "音質：跨腔初始化只在 Matcha-TTS 有明顯好處", "X 的 MCD 改善 −0.313 dB 對 −0.095 dB"],
    [fa.FaLightbulb, C.accent5, "推測原因（待驗證）", "VITS 要用 1 小時資料自己學波形，長時間微調可能洗掉來源模型的聲學知識"],
  ];
  for (let i = 0; i < 3; i++) {
    const [Icon, col, h, t] = tk[i];
    const y = 4.55 + i * 0.75;
    await iconCircle(s, Icon, 0.6, y + 0.07, 0.55, col, `takeaway ${i + 1}`);
    label(s, h, 1.35, y, 5.85, 0.35, { fontSize: 14, bold: true, color: C.text2 });
    label(s, t, 1.35, y + 0.34, 5.85, 0.35, { fontSize: 12, color: HEX.muted });
  }
  s.addChart(pres.charts.BAR, [
    { name: "Matcha-TTS", labels: ["+AD", "X", "XAD"], values: [-0.119, -0.313, -0.332] },
    { name: "VITS", labels: ["+AD", "X", "XAD"], values: [-0.089, -0.095, -0.076] },
  ], { x: 7.5, y: 1.85, w: 5.2, h: 4.85, barDir: "col", chartColors: [HEX.matcha, HEX.vits], valAxisMinVal: -0.4, valAxisMaxVal: 0,
       ...chartOpts("MCD 與各自 Base 的差（dB，越低越好）", { dataLabelFormatCode: "0.00", valAxisLabelFormatCode: "0.0" }), objectName: "mcd chart" });
  s.addNotes("表格把兩個架構的 Base 和 XAD 放在一起比。時長方面，XAD 在兩個架構都有效，但 VITS 的 XAD（74.8）仍比 Matcha-TTS 的 XAD（81.3）低 6.5 點。右圖是音質：Matcha-TTS 加上跨腔初始化後 MCD 改善約 0.31 dB，VITS 只有約 0.1 dB，三種設定都差不多。我的推測是 Matcha-TTS 用預訓練的 HiFi-GAN 聲碼器，而 VITS 的波形解碼器只能用一小時資料學，而且又微調了 20 萬步，可能把來源模型的聲學知識洗掉了；這還需要實驗驗證。");

  // ===== 8. 下一步 =====
  pres.addSection({ title: "下一步" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "下一步" });
  s.addText("下一步", { placeholder: "title" });
  s.addText("先完成海陸的 VITS 實驗，再開始客語語音助理", { placeholder: "lead" });
  card(s, 0.6, 1.85, 5.9, 4.85, "hailu progress card");
  label(s, "海陸 VITS 實驗進度（截至 11／16）", 0.9, 2.0, 5.3, 0.45, { fontSize: 16, bold: true, color: C.text2 });
  const prog = [
    ["海陸 VITS Base × 3 種子", 1.0, "完成", HEX.ok],
    ["四縣 48 小時來源模型（50 萬步）", 0.7, "70％", HEX.run],
    ["海陸 VITS XAD × 3 種子", 0.0, "等待中", HEX.plan],
  ];
  prog.forEach(([t, p, txt, col], i) => {
    const y = 2.7 + i * 1.05;
    label(s, t, 0.9, y, 4.2, 0.35, { fontSize: 14 });
    label(s, txt, 5.0, y, 1.2, 0.35, { fontSize: 14, bold: true, color: col, align: "right" });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.9, y: y + 0.42, w: 5.3, h: 0.22, rectRadius: 0.05, fill: { color: HEX.grid }, line: { color: HEX.grid }, objectName: `progress track ${i + 1}` });
    if (p > 0) s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.9, y: y + 0.42, w: 5.3 * p, h: 0.22, rectRadius: 0.05, fill: { color: col }, line: { color: col }, objectName: `progress bar ${i + 1}` });
  });
  label(s, "預計 11／17 完成，結果補進四縣的比較表", 0.9, 5.95, 5.3, 0.5, { fontSize: 14, bold: true, color: HEX.run });
  card(s, 6.8, 1.85, 5.9, 4.85, "rag card");
  label(s, "客語語音助理（RAG）規劃", 7.1, 2.0, 5.3, 0.45, { fontSize: 16, bold: true, color: C.text2 });
  const rag = [
    [fa.FaMicrophone, "語音問題", "自動語音辨識（ASR，Automatic Speech Recognition）"],
    [fa.FaSearch, "檢索客語知識庫", "教育部客語辭典、客委會公開資料"],
    [fa.FaComments, "生成回答", "大型語言模型（LLM，Large Language Model）"],
    [fa.FaVolumeUp, "客語語音輸出", "漢字轉拼音後，用 XAD 合成"],
  ];
  for (let i = 0; i < rag.length; i++) {
    const [Icon, t, d] = rag[i];
    const y = 2.6 + i * 1.0;
    await iconCircle(s, Icon, 7.1, y, 0.6, C.accent4, `rag step ${i + 1}`);
    label(s, t, 7.9, y - 0.05, 4.6, 0.35, { fontSize: 14, bold: true, color: C.text2 });
    label(s, d, 7.9, y + 0.3, 4.6, 0.35, { fontSize: 12, color: HEX.muted });
    if (i < rag.length - 1) arrow(s, 7.4, y + 0.63, 0, 0.32, `rag arrow ${i + 1}`, C.accent3);
  }
  s.addNotes("海陸的 VITS 實驗正在進行：海陸 Base 三個種子已完成，四縣 48 小時來源模型跑到約七成，之後是海陸 XAD 三個種子，預計 11 月 17 日完成，結果會補進比較表。完成後開始客語語音助理：語音問題先經 ASR 轉成文字，檢索客語知識庫，再由大型語言模型生成回答，最後把漢字轉成拼音，用 XAD 合成客語語音。主要的技術難點是大型語言模型不擅長寫客語漢字，需要接上現有的發音詞典。");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})().catch((e) => { console.error(e); process.exit(1); });
