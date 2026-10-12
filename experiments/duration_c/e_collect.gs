/**
 * 實驗 E2 線上聽測:把聽者作答收進 Google 試算表(Google Apps Script)。
 *
 * 設定一次就好:
 *   1. 開一個新的 Google 試算表(例如命名「客語聽測回收」)。
 *   2. 在「試算表裡」選單「擴充功能」→「Apps Script」;先把編輯器裡原本的
 *      function myFunction() { } 整段刪掉(要全部清空),再貼上這整個檔案,按「儲存」(磁碟圖示或 Ctrl+S)。
 *      若 Apps Script 不是從試算表開的(直接在 script.google.com 新建),請把試算表網址中
 *      /d/ 與 /edit 之間那一串填到下面的 SHEET_ID。
 *   3. 右上角「部署」→「新增部署作業」→ 齒輪選「網頁應用程式」:
 *        執行身分:我    誰可以存取:所有人
 *      按「部署」,依指示授權(會出現「Google 尚未驗證這個應用程式」→「進階」→「前往…」→「允許」)。
 *   4. 複製「網頁應用程式網址」(https://script.google.com/macros/s/……/exec),
 *      給 e_build_ab.py --submit-url 使用。
 *
 * 每份聽測(例如 sixian_base_vs_c2)會自動建立同名的工作表;每答一題就新增一列。
 * 分析時:在該工作表「檔案」→「下載」→「逗號分隔值 (.csv)」,放進一個資料夾,再跑 e_analyze.py。
 */
var SHEET_ID = '';   // 通常留空;只有 Apps Script 不是從試算表開的時候才需要填

var HEADER = ['received', 'listener', 'name', 'dialect', 'frequency', 'trial', 'q1', 'q2', 'seconds', 'test', 'ua'];

function safe_(v) {
  // 避免以 = + - @ 開頭的文字被試算表當成公式(數字不受影響)
  if (typeof v === 'string' && /^[=+\-@]/.test(v) && isNaN(Number(v))) return "'" + v;
  return v;
}

function doPost(e) {
  var lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    var d = JSON.parse(e.postData.contents);
    var name = String(d.test || 'listening').replace(/[^\w\-]/g, '_').slice(0, 90);
    var ss = SHEET_ID ? SpreadsheetApp.openById(SHEET_ID) : SpreadsheetApp.getActiveSpreadsheet();
    if (!ss) throw new Error('找不到試算表:請從試算表的「擴充功能 → Apps Script」開啟,或填寫 SHEET_ID');
    var sh = ss.getSheetByName(name) || ss.insertSheet(name);
    if (sh.getLastRow() === 0) sh.appendRow(HEADER);
    var rows = (d.rows || []).map(function (r) {
      return [new Date(), d.listener, d.name, d.dialect, d.frequency, r.trial, Number(r.q1), Number(r.q2),
              Number(r.sec), d.test, d.ua || ''].map(safe_);
    });
    if (rows.length) sh.getRange(sh.getLastRow() + 1, 1, rows.length, HEADER.length).setValues(rows);
    return ContentService.createTextOutput('ok');
  } finally {
    lock.releaseLock();
  }
}

function doGet() {
  var ss = SHEET_ID ? SpreadsheetApp.openById(SHEET_ID) : SpreadsheetApp.getActiveSpreadsheet();
  return ContentService.createTextOutput(ss ? '聽測回收服務正常運作中(試算表:' + ss.getName() + ')'
                                            : '★ 找不到試算表:請從試算表的「擴充功能 → Apps Script」開啟,或填寫 SHEET_ID');
}
