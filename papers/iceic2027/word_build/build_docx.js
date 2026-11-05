// blocks.json(tex2json.py 產生)→ IEEE A4 雙欄 Word 檔。版面數值取自 IEEEtran a4paper(上 19 mm、下 43 mm、左右 14.32 mm、欄距 4.22 mm)。
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, AlignmentType,
  BorderStyle, SectionType, LevelFormat, TabStopType, VerticalAlign, TableLayoutType,
} = require("docx");

const DIR = __dirname;
const OUT = process.argv[2] || path.join(DIR, "paper.docx");
const B = JSON.parse(fs.readFileSync(path.join(DIR, "blocks.json"), "utf8"));
const FONT = "Times New Roman";
const MM = 56.6929;                                  // twips / mm
const PAGE = { width: 11906, height: 16838 };
const MARGIN = { top: Math.round(19 * MM), bottom: Math.round(43 * MM), left: Math.round(14.32 * MM), right: Math.round(14.32 * MM) };
const TEXT_W = PAGE.width - MARGIN.left - MARGIN.right;
const COL_GAP = Math.round(4.22 * MM);
const COL_W = Math.floor((TEXT_W - COL_GAP) / 2);

const runs = (rs, base = {}) => rs.map((r) => r.br ? new TextRun({ text: "", break: 1 }) : new TextRun({
  text: r.text, font: r.mono ? "Courier New" : FONT, size: base.size || 20,
  bold: !!(r.b || base.bold), italics: !!(r.i || base.italics),
  superScript: !!r.sup, subScript: !!r.sub, smallCaps: !!(r.sc || base.smallCaps),
}));
const plain = (rs) => rs.map((r) => r.text).join("");

const body = [];
const front = [];
const emptyBorders = { top: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, bottom: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" } };

function figure(b) {
  const pdfSize = { "fig1.png": [246.479, 229.029], "fig2.png": [241.218, 133.154] }[b.img];
  const px = (pt) => Math.round((pt / 72) * 96);
  body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(DIR, b.img)),
      transformation: { width: px(pdfSize[0]), height: px(pdfSize[1]) },
      altText: { title: `Fig. ${b.num}`, description: plain(b.caption), name: `Fig. ${b.num}` } })] }));
  body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
    children: [new TextRun({ text: `Fig. ${b.num}.  `, font: FONT, size: 16 }), ...runs(b.caption, { size: 16 })] }));
}

function table(b) {
  body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 80 }, keepNext: true,
    children: [new TextRun({ text: `TABLE ${b.num}. `, font: FONT, size: 16 }), ...runs(b.caption, { size: 16, smallCaps: true })] }));
  const spec = b.spec.replace(/[^lcr]/g, "");
  const n = spec.length;
  // 欄寬:依各欄最長文字估算(8 pt Times 約 85 twips/字),超過一欄寬就等比縮小
  const len = Array(n).fill(3);
  for (const r of b.rows) {
    let c = 0;
    for (const cell of r.cells) {
      if (cell.span === 1) len[c] = Math.max(len[c], plain(cell.runs).length);
      c += cell.span;
    }
  }
  // 一律填滿一欄寬,依文字長度分配(另加每欄固定留白),避免數字在儲存格內斷行
  const raw = len.map((l, k) => l + (k === 0 ? 7 : 2.2));     // 第一欄是系統名稱,多留一點
  const total = raw.reduce((a, x) => a + x, 0);
  const w = raw.map((x) => Math.floor((x * COL_W) / total));
  const tw = w.reduce((a, x) => a + x, 0);
  const rule = (kind) => kind ? { style: BorderStyle.SINGLE, size: kind === "thick" ? 8 : 4, color: "000000" } : { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const rows = b.rows.map((r, ri) => {
    const keep = ri < b.rows.length - 1;
    let c = 0;
    const cells = r.cells.map((cell) => {
      const c0 = c, c1 = c + cell.span - 1;
      c += cell.span;
      const al = cell.align || spec[c0];
      const cmid = (r.cmid || []).some(([a, z]) => a - 1 <= c0 && c1 <= z - 1);
      return new TableCell({
        columnSpan: cell.span, width: { size: w.slice(c0, c1 + 1).reduce((a, x) => a + x, 0), type: WidthType.DXA },
        margins: { top: 15, bottom: 15, left: 50, right: 50 }, verticalAlign: VerticalAlign.CENTER,
        borders: { top: rule(r.top), bottom: rule(r.bottom || (cmid ? "thin" : null)), left: rule(null), right: rule(null) },
        children: [new Paragraph({ alignment: al === "l" ? AlignmentType.LEFT : al === "r" ? AlignmentType.RIGHT : AlignmentType.CENTER,
          spacing: { before: 0, after: 0 }, keepNext: keep, keepLines: true, children: runs(cell.runs, { size: 16 }) })],
      });
    });
    return new TableRow({ children: cells, cantSplit: true });
  });
  body.push(new Table({ rows, width: { size: tw, type: WidthType.DXA }, columnWidths: w, alignment: AlignmentType.CENTER, layout: TableLayoutType.FIXED }));
  body.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
}

for (const b of B) {
  switch (b.t) {
    case "title":
      front.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 280 }, children: runs(b.runs.map((r) => ({ ...r, text: r.text.replace(/-/g, "\u2011") })), { size: 48 }) }));
      break;
    case "authors": {
      const cw = Math.floor(TEXT_W / b.authors.length);
      const cells = b.authors.map((a) => new TableCell({
        width: { size: cw, type: WidthType.DXA }, borders: emptyBorders,
        children: [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20 }, children: runs(a.name, { size: 22 }) }),
          ...a.lines.map((l) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: runs(l, { size: 20 }) }))],
      }));
      front.push(new Table({ rows: [new TableRow({ children: cells })], width: { size: cw * b.authors.length, type: WidthType.DXA },
        columnWidths: Array(b.authors.length).fill(cw), borders: { insideHorizontal: emptyBorders.top, insideVertical: emptyBorders.top, ...emptyBorders } }));
      front.push(new Paragraph({ spacing: { after: 200 }, children: [] }));
      break;
    }
    case "abstract":
      body.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 80 }, indent: { firstLine: 180 },
        children: [new TextRun({ text: "Abstract", font: FONT, size: 18, bold: true, italics: true }), new TextRun({ text: "—", font: FONT, size: 18, bold: true }),
          ...runs(b.runs, { size: 18, bold: true })] }));
      break;
    case "keywords":
      body.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 160 }, indent: { firstLine: 180 },
        children: [new TextRun({ text: "Index Terms", font: FONT, size: 18, bold: true, italics: true }), new TextRun({ text: "—", font: FONT, size: 18, bold: true }),
          ...runs(b.runs, { size: 18, bold: true })] }));
      break;
    case "h1":
      body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 }, keepNext: true,
        children: [new TextRun({ text: b.text, font: FONT, size: 20, smallCaps: true })] }));
      break;
    case "h2":
      body.push(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { before: 120, after: 60 }, keepNext: true,
        children: [new TextRun({ text: b.text, font: FONT, size: 20, italics: true })] }));
      break;
    case "parhead":
      body.push(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { before: 60, after: 0 }, keepNext: true,
        children: [new TextRun({ text: b.text, font: FONT, size: 20, bold: true })] }));
      break;
    case "p":
      body.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 0 }, indent: { firstLine: 200 }, children: runs(b.runs) }));
      break;
    case "bullet": {
      // 粗體標題自成一行(靠左,不被左右對齊拉開),說明另起一段、與標題文字切齊
      const k = b.runs.findIndex((r) => r.br);
      const head = k < 0 ? b.runs : b.runs.slice(0, k), rest = k < 0 ? [] : b.runs.slice(k + 1);
      body.push(new Paragraph({ alignment: k < 0 ? AlignmentType.JUSTIFIED : AlignmentType.LEFT, numbering: { reference: "bullets", level: 0 },
        spacing: { after: 0 }, keepNext: rest.length > 0, children: runs(head) }));
      if (rest.length) body.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 0 }, indent: { left: 200 }, children: runs(rest) }));
      break;
    }
    case "figure": figure(b); break;
    case "table": table(b); break;
    case "refs":
      body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 }, keepNext: true,
        children: [new TextRun({ text: "REFERENCES", font: FONT, size: 20, smallCaps: true })] }));
      b.items.forEach((it, k) => body.push(new Paragraph({
        alignment: AlignmentType.JUSTIFIED, spacing: { after: 20 }, indent: { left: 340, hanging: 340 },
        tabStops: [{ type: TabStopType.LEFT, position: 340 }],
        children: [new TextRun({ text: `[${k + 1}]\t`, font: FONT, size: 16 }), ...runs(it, { size: 16 })] })));
      break;
  }
}

const page = { size: PAGE, margin: MARGIN };
const doc = new Document({
  creator: "Chien-Chih Fang",
  title: plain(B[0].runs),
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 200, hanging: 200 } } } }] }] },   // 項目符號切齊段落左緣
  sections: [
    { properties: { page }, children: front },
    { properties: { page, type: SectionType.CONTINUOUS, column: { count: 2, space: COL_GAP, equalWidth: true } }, children: body },
  ],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT); });
