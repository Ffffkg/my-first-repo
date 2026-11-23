// 研究進度報告(2026 年 11 月 23 日):VITS 海陸結果、入聲時長誤差來源分析(S1~S4)。
// 樣式沿用 build_progress_202611.js;標點全形、不用斜體、每頁有頁碼、縮寫第一次出現時寫出全名。
//   NODE_PATH=<pptxgenjs 所在的 node_modules> node build_progress_20261123.js out.pptx
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require(process.env.APPLY_THEME || "/root/.claude/skills/synced/9c2c4fb7-9911-4e7c-a212-1bd5071b37ce_e3c3c317-72b7-40a6-8ca4-a5e39d1971db/pptx/scripts/apply_theme.js");

const OUT = process.argv[2] || "progress_20261123.pptx";
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
pres.title = "研究進度報告 2026-11-23";
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
    { text: { text: "客語語音合成進度報告｜2026 年 11 月 23 日", options: { x: 0.6, y: 7.02, w: 6, h: 0.3, fontSize: 11, color: C.accent3, margin: 0 } } },
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
function note2(slide, text) {
  label(slide, text, 0.6, 6.45, 12.1, 0.45, { fontSize: 10, color: HEX.muted, valign: "top" });
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
  s.addText("研究進度報告", { placeholder: "title" });
  s.addText("VITS 海陸結果，以及入聲時長誤差的來源分析", { placeholder: "body" });
  s.addText([
    { text: "Chien-Chih Fang｜中原大學 智慧運算與大數據碩士學位學程", options: { breakLine: true } },
    { text: "報告期間：2026 年 11 月 16 日至 11 月 23 日" },
  ], { placeholder: "meta" });
  s.addNotes("這次報告兩件事：第一，把 VITS 的實驗補上海陸，看 XAD 換到另一種架構後在兩個腔調是否一致；第二，分析 XAD 剩下的入聲時長誤差從哪裡來，並測試雙層時長預測器能不能改善。");

  // ===== 2. VITS 海陸結果 =====
  pres.addSection({ title: "VITS" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "VITS" });
  s.addText("VITS 架構：海陸結果與四縣不一致", { placeholder: "title" });
  s.addText("XAD 在四縣有效、在海陸無效；跨腔初始化在兩腔都改善音質", { placeholder: "lead" });
  const cfg = ["Base", "+AD", "X", "XAD"];
  s.addChart(pres.charts.BAR, [
    { name: "四縣", labels: cfg, values: [68.3, 70.6, 70.7, 74.8] },
    { name: "海陸", labels: cfg, values: [75.3, 67.6, 72.9, 73.8] },
  ], { x: 0.6, y: 1.75, w: 5.6, h: 4.5, barDir: "col", chartColors: [HEX.sixian, HEX.hailu], valAxisMinVal: 50, valAxisMaxVal: 90,
       ...chartOpts("VITS 入聲命中率（％）"), objectName: "vits hit chart" });
  label(s, "與 VITS Base 相比（三個種子；括號為 95％ 信賴區間）", 6.5, 1.75, 6.2, 0.4, { fontSize: 13, bold: true, color: C.text2 });
  s.addTable([
    [head("系統"), head("四縣入聲命中率差"), head("海陸入聲命中率差"), head("MCD 差（四縣／海陸）")],
    ["+AD", "+2.3（−0.4～+5.2）", { text: "−7.7（−11.3～−4.0）", options: { bold: true, color: HEX.red } }, "−0.09/−0.02"],
    ["X", "+2.4（−0.9～+5.5）", "−2.4（−6.1～+1.3）", { text: "−0.10/−0.22", options: { bold: true } }],
    [{ text: "XAD", options: { bold: true } }, { text: "+6.5（+3.2～+9.6）", options: { bold: true, color: HEX.ok } }, "−1.5（−5.9～+2.6）", { text: "−0.08/−0.22", options: { bold: true } }],
  ], { x: 6.5, y: 2.2, w: 6.2, colW: [0.75, 1.8, 1.9, 1.75], fontSize: 11, rowH: 0.45, ...TABLE, align: "center", objectName: "vits table" });
  card(s, 6.5, 4.25, 6.2, 2.0, "vits takeaway");
  s.addText([
    { text: "海陸只加 AD 反而變差，但不是整體偏長或偏短：", options: { bold: true, color: C.text2, breakLine: true } },
    { text: "平均偏差只有 +1.0 毫秒，入聲時長相關係數 ρ 卻從 0.641 降到 0.596，代表個別音節的長短判斷變差", options: { breakLine: true } },
    { text: "Matcha-TTS 上 XAD 在兩腔都有效（+10.3、+12.8），結論仍然穩定", options: { bold: true, color: HEX.ok } },
  ], { x: 6.75, y: 4.35, w: 5.7, h: 1.8, isTextBox: true, margin: 0, fontSize: 13, color: C.text1, valign: "middle", paraSpaceAfter: 6 });
  note2(s, "VITS：Variational Inference with adversarial learning for end-to-end Text-to-Speech；XAD：跨腔初始化（X）+ 對齊監督時長（AD）+ 放大的時長預測器；MCD：梅爾倒頻譜失真（Mel-Cepstral Distortion，越低越好）");
  s.addNotes("左圖是 VITS 四種系統的入聲命中率。右表是和 VITS Base 相比的差。四縣只有完整的 XAD 顯著有效；海陸則是 XAD 沒有改善，只加 AD 甚至顯著變差 7.7 點。檢查偏差後發現，海陸 +AD 的平均偏差幾乎是 0，問題是時長相關係數下降，也就是個別音節的長短判斷變差。跨腔初始化在兩腔都顯著改善音質。Matcha-TTS 上的主要結論不受影響。");

  // ===== 3. S1 誤差來源診斷 =====
  pres.addSection({ title: "誤差來源" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "誤差來源" });
  s.addText("XAD＊ 剩下的入聲誤差從哪裡來", { placeholder: "title" });
  s.addText("如果整個音節的長度完全正確，入聲命中率最多可提高 7～11 點", { placeholder: "lead" });
  s.addChart(pres.charts.BAR, [
    { name: "四縣", labels: ["目前", "（a）音節總長", "（b）母音比例"], values: [76.0, 87.2, 92.3] },
    { name: "海陸", labels: ["目前", "（a）音節總長", "（b）母音比例"], values: [80.8, 88.1, 91.6] },
  ], { x: 0.6, y: 1.75, w: 6.6, h: 4.5, barDir: "col", chartColors: [HEX.sixian, HEX.hailu], valAxisMinVal: 60, valAxisMaxVal: 100,
       ...chartOpts("入聲母音 ±30 毫秒命中率（％，驗證集）"), objectName: "s1 chart" });
  const s1 = [
    [fa.FaRulerHorizontal, C.accent4, "（a） 音節總長的上限", "+11.2 / +7.4", "四縣／海陸，信賴區間都不含 0"],
    [fa.FaChartPie, C.accent6, "誤差分解：音節總長／母音比例", "31％／62％", "海陸 30％／55％"],
    [fa.FaExchangeAlt, C.accent2, "入聲與舒聲相反", "入聲重比例，舒聲重總長", "舒聲 （a） 可提高 21～24 點"],
  ];
  for (let i = 0; i < 3; i++) {
    const [Icon, col, lab, big, sub] = s1[i];
    const y = 1.75 + i * 1.55;
    card(s, 7.5, y, 5.2, 1.4, `s1 card ${i + 1}`);
    await iconCircle(s, Icon, 7.75, y + 0.35, 0.7, col, `s1 ${i + 1}`);
    label(s, lab, 8.65, y + 0.1, 3.9, 0.35, { fontSize: 13, bold: true, color: C.text2 });
    label(s, big, 8.65, y + 0.45, 3.9, 0.5, { fontSize: 20, bold: true, color: col });
    label(s, sub, 8.65, y + 0.95, 3.9, 0.3, { fontSize: 11, color: HEX.muted });
  }
  note2(s, "XAD＊：時長標籤來自只用 1 小時資料訓練的強制對齊器（MFA，Montreal Forced Aligner）；（a）、（b） 用到真人時長，只是上限診斷，不是模型成績");
  s.addNotes("用 XAD＊ 合成驗證集，每個音節量音節總長和母音長度，再做兩種假設性的調整。（a） 把音節總長換成真人的，入聲命中率四縣可以多 11.2 點、海陸多 7.4 點，通過我們事先訂的門檻，所以值得嘗試先預測音節總長的雙層模型。（b） 把母音比例換成真人的，提高更多，但這個算法直接用到真人母音長度，下一頁會說明要打折扣。");

  // ===== 4. S2 標籤可靠度、S3 音節邊界 =====
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "誤差來源" });
  s.addText("音節總長的標籤本身就不穩定", { placeholder: "title" });
  s.addText("母音前後的邊界很穩；音節兩端（子音與子音、子音與靜音的交界）最難切", { placeholder: "lead" });
  const bl = ["聲母 → 母音", "母音 → 塞音韻尾", "入聲音節起點", "入聲音節終點"];
  s.addChart(pres.charts.BAR, [
    { name: "四縣", labels: bl, values: [99.3, 90.9, 65.3, 60.8] },
    { name: "海陸", labels: bl, values: [97.8, 95.3, 74.0, 56.7] },
  ], { x: 0.6, y: 1.75, w: 6.3, h: 4.5, barDir: "col", chartColors: [HEX.sixian, HEX.hailu], valAxisMinVal: 0, valAxisMaxVal: 100,
       ...chartOpts("兩套 MFA 邊界相差 ≤ 20 毫秒的比例（％）"), objectName: "s2 chart" });
  label(s, "入聲標籤：兩套 MFA 的一致程度", 7.2, 1.75, 5.5, 0.4, { fontSize: 13, bold: true, color: C.text2 });
  s.addTable([
    [head("標籤"), head("相關係數（四縣／海陸）"), head("平均差（四縣／海陸）")],
    ["母音長度", "0.852／0.943", "12.8／8.2 毫秒"],
    [{ text: "音節總長", options: { bold: true } }, "0.827／0.844", { text: "38.7／33.2 毫秒", options: { bold: true, color: HEX.red } }],
    ["母音占音節比例", "0.773／0.846", "0.081／0.062"],
  ], { x: 7.2, y: 2.2, w: 5.5, colW: [1.5, 2.0, 2.0], fontSize: 12, rowH: 0.42, ...TABLE, align: "center", objectName: "s2 table" });
  card(s, 7.2, 4.15, 5.5, 2.1, "s3 card");
  s.addText([
    { text: "合成與真人的音節總長誤差（約 35 毫秒）和兩套 MFA 之間的差距相當", options: { bold: true, color: C.text2, breakLine: true } },
    { text: "音節邊界無法只從音素序列推出（規則正確率 93％／96.5％，鼻音可屬前或屬後音節），改由文字（拼音）提供，訓練與測試句全部涵蓋" },
  ], { x: 7.45, y: 4.25, w: 5.0, h: 1.9, isTextBox: true, margin: 0, fontSize: 13, color: C.text1, valign: "middle", paraSpaceAfter: 8 });
  s.addNotes("在同一批真人錄音上比較兩套 MFA 切出的邊界。和母音相鄰的邊界非常穩定，包括原本擔心的入聲母音到塞音韻尾；不穩定的是音節兩端。所以母音長度是最可靠的標籤，音節總長最不可靠，兩套 MFA 之間就差 33 到 39 毫秒，和合成語音的誤差差不多大。另外，音節邊界不能只從音素序列推出，因為單一鼻音可能是前一個音節的韻尾或後一個音節的聲母，所以改由文字提供音節邊界。");

  // ===== 5. S4 雙層時長預測器 =====
  pres.addSection({ title: "雙層模型" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "雙層模型" });
  s.addText("雙層時長預測器：沒有改善", { placeholder: "title" });
  s.addText("先預測音節總長、再分配給各音素；其他條件與 XAD＊ 完全相同（三個種子，測試集）", { placeholder: "lead" });
  const sys = ["XAD＊", "音節總長當提示", "總長 × 比例"];
  s.addChart(pres.charts.BAR, [
    { name: "四縣", labels: sys, values: [79.1, 79.4, 78.9] },
    { name: "海陸", labels: sys, values: [82.4, 82.5, 76.0] },
  ], { x: 0.6, y: 1.75, w: 5.6, h: 4.5, barDir: "col", chartColors: [HEX.sixian, HEX.hailu], valAxisMinVal: 60, valAxisMaxVal: 90,
       ...chartOpts("入聲命中率（％）"), objectName: "s4 chart" });
  label(s, "與 XAD＊ 的入聲命中率差（百分點，95％ 信賴區間）", 6.5, 1.75, 6.2, 0.4, { fontSize: 13, bold: true, color: C.text2 });
  forest(s, [
    { label: "四縣 提示", v: 0.3, ci: [-1.2, 1.8], text: "+0.3（−1.2～+1.8）", color: HEX.sixian },
    { label: "四縣 總長×比例", v: -0.1, ci: [-3.3, 3.2], text: "−0.1（−3.3～+3.2）", color: HEX.sixian },
    { label: "海陸 提示", v: 0.1, ci: [-1.3, 1.5], text: "+0.1（−1.3～+1.5）", color: HEX.hailu },
    { label: "海陸 總長×比例", v: -6.4, ci: [-9.7, -3.0], text: "−6.4（−9.7～−3.0）", color: HEX.red, bold: true },
  ], { x: 6.5, y: 2.2, w: 6.2, rowH: 0.45, lo: -10, hi: 5, labelW: 1.75, valueW: 1.95, name: "s4 forest" });
  label(s, "合成波形上的入聲音節總長誤差（毫秒）", 6.5, 4.55, 6.2, 0.35, { fontSize: 13, bold: true, color: C.text2 });
  s.addTable([
    [head(""), head("XAD＊"), head("音節總長當提示"), head("總長 × 比例")],
    ["四縣", "37.8", "36.4", "40.1"],
    ["海陸", "39.7", "39.9", { text: "43.7", options: { bold: true, color: HEX.red } }],
  ], { x: 6.5, y: 4.95, w: 6.2, colW: [1.1, 1.5, 1.8, 1.8], fontSize: 12, rowH: 0.4, ...TABLE, align: "center", objectName: "s4 table" });
  s.addNotes("我們試了兩種雙層時長預測器。第一種只把預測的音節總長當作提示特徵，第二種把音節總長乘上音節內比例。兩種在兩個腔調都沒有提高入聲命中率；總長乘比例在海陸還顯著變差 6.4 點。右下表顯示，就算明確預測音節總長，音節總長的誤差也沒有下降，仍然在 36 到 44 毫秒，和兩套 MFA 之間的差距差不多。");

  // ===== 6. 結論與下一步 =====
  pres.addSection({ title: "結論" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "結論" });
  s.addText("結論與下一步", { placeholder: "title" });
  s.addText("剩下的入聲時長誤差主要受限於標籤與量測精度，不是時長預測器的結構", { placeholder: "lead" });
  const chain = [
    ["誤差來源", "音節總長正確的話，最多可提高 7～11 點"],
    ["標籤可靠度", "音節總長的標籤雜訊和誤差差不多大"],
    ["音節邊界", "必須由文字（拼音）提供"],
    ["雙層模型", "明確預測音節總長也無法降低誤差"],
  ];
  chain.forEach(([h, t], i) => {
    const x = 0.6 + i * 3.08;
    card(s, x, 1.85, 2.8, 2.0, `chain ${i + 1}`);
    label(s, `${i + 1}. ${h}`, x + 0.2, 1.95, 2.4, 0.5, { fontSize: 15, bold: true, color: C.text2 });
    label(s, t, x + 0.2, 2.5, 2.4, 1.2, { fontSize: 13, valign: "top" });
    if (i < 3) arrow(s, x + 2.82, 2.85, 0.22, 0, `chain arrow ${i + 1}`, C.accent3);
  });
  s.addTable([
    [head("1 小時目標腔資料"), head("四縣入聲命中率"), head("海陸入聲命中率")],
    ["Base", "71.0％", "69.4％"],
    [{ text: "XAD＊（連對齊器也只用 1 小時）", options: { bold: true } }, { text: "79.1％", options: { bold: true } }, { text: "82.4％", options: { bold: true } }],
    ["48 小時 Base（參考）", "81.1％", "83.0％"],
  ], { x: 0.6, y: 4.15, w: 6.4, colW: [3.2, 1.6, 1.6], fontSize: 12, rowH: 0.42, ...TABLE, align: "center", objectName: "summary table" });
  card(s, 7.3, 4.15, 5.4, 2.1, "next card");
  label(s, "下一步", 7.55, 4.25, 4.9, 0.4, { fontSize: 15, bold: true, color: C.text2 });
  s.addText([
    { text: "時長模型研究線收尾，寫入碩論", options: { bullet: true, breakLine: true } },
    { text: "倫理程序確認後進行聽測", options: { bullet: true, breakLine: true } },
    { text: "開始客語語音助理（RAG，Retrieval-Augmented Generation，檢索增強生成）", options: { bullet: true } },
  ], { x: 7.55, y: 4.7, w: 4.9, h: 1.45, isTextBox: true, margin: 0, fontSize: 13, color: C.text1, valign: "top", paraSpaceAfter: 6 });
  s.addNotes("四個分析串起來的結論是：在只有 1 小時資料時，XAD 剩下的入聲時長誤差，主要受限於標籤和量測的精度，不是時長預測器的結構能解決的。XAD＊ 連對齊器都只用 1 小時，入聲命中率已經接近 48 小時模型。接下來時長模型這條線收尾，等倫理程序確認後做聽測，並開始客語語音助理。");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})().catch((e) => { console.error(e); process.exit(1); });
