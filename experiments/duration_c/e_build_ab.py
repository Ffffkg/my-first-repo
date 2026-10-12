"""實驗 E2:兩個系統的成對比較聽測(CMOS / AB 偏好),產生可以直接寄給聽者的資料夾。

    python e_build_ab.py --a ~/hakka_tts/exp_c/eval/base_sixian_step40000 \
                         --b ~/hakka_tts/exp_c/eval/c2_sixian_step40000 \
                         --n 30 --out ~/hakka_tts/exp_e/sixian_base_vs_c2

輸出:
  <out>/listening_test/       ← 整個資料夾交給聽者(壓縮後寄出或放雲端)
      index.html              ← 用瀏覽器打開即可作答,完成後下載 CSV 回傳
      stimuli/tXX_1.wav, tXX_2.wav
  <out>/key.csv               ← 每題的「1/2 分別是哪個系統」,不要給聽者
  <out>/trials.json

設計(事先登記):
  * 句子:兩系統都有合成的測試句中,優先挑入聲 ≥ 2 個的句子,種子 20260716 抽 n 句;生成種子固定 1234。
  * 每題左右順序隨機;另外加入 n_identical 題「兩邊完全相同」的檢查題(期望回答 0),用來檢查聽者是否專心。
  * 響度:兩邊都正規化到相同 RMS(−23 dBFS),並限制峰值,避免「比較大聲就比較好」。
  * 聽者資料:會說的腔調可複選(四縣 / 海陸 / 其他),CSV 的 dialect 欄記成「四縣+海陸」這種形式。
  * 作答:7 點 CMOS,−3(1 好很多)… 0(一樣)… +3(2 好很多),外加「入聲字聽起來哪邊比較自然」的同一量表。
"""
import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402


def rms_normalize(x, target_dbfs=-23.0, peak=0.95):
    x = np.asarray(x, np.float64)
    rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    y = x * (10 ** (target_dbfs / 20) / rms)
    m = np.abs(y).max()
    return (y * (peak / m) if m > peak else y).astype(np.float32)


HTML = """<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>客語合成語音聽測</title>
<style>
body{font-family:system-ui,-apple-system,"Noto Sans TC",sans-serif;max-width:760px;margin:0 auto;padding:16px;line-height:1.6;background:#fafafa;color:#222}
.card{background:#fff;border:1px solid #ddd;border-radius:10px;padding:16px;margin:14px 0}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:8px 0}.lab{min-width:2em;font-weight:600}
.scale{display:flex;justify-content:space-between;gap:2px;margin:6px 0 2px}.scale label{flex:1;text-align:center;font-size:14px;padding:4px 0}
.scale input{display:block;margin:0 auto 2px;width:22px;height:22px}.hint{display:flex;justify-content:space-between;font-size:13px;color:#666}
input[type=checkbox]{width:20px;height:20px;vertical-align:middle}input[type=text]{font-size:16px;padding:4px 6px}select{font-size:16px}
audio{width:100%;max-width:420px}.err{color:#c62828;font-weight:600;min-height:1.4em}
button{font-size:17px;padding:10px 20px;border-radius:8px;border:1px solid #888;background:#fff;cursor:pointer;touch-action:manipulation}
button.primary{background:#1f5fbf;color:#fff;border-color:#1f5fbf}#prog{color:#666}
textarea{width:100%;height:9em;font-size:12px}
</style></head><body>
<h2>客語合成語音聽測</h2>
<div class="card err" id="nojs">如果按「開始」沒有反應:請改用 Chrome 或 Safari 開啟這個網頁,不要在 LINE、檔案預覽或雲端硬碟的預覽畫面裡開。</div>
<div class="card" id="resume" style="display:none"><p>偵測到上次沒做完的進度。</p>
<div class="row"><button class="primary" id="btn_resume" type="button">繼續上次進度</button><button id="btn_restart" type="button">重新開始</button></div></div>
<div class="card" id="intro">
<p>每一題有兩段<b>同一句話</b>的合成語音(1 與 2)。請戴耳機,在安靜的地方作答,每段可以重複聽。</p>
<p>請依兩個問題作答:<b>(一)整體哪一段比較自然</b>;<b>(二)句中的入聲字(-b/-d/-g 結尾的字)哪一段唸得比較自然、長短比較對</b>。分不出來就選 0。</p>
<div class="row"><span>你的代號(不用真名):</span><input type="text" id="lid" placeholder="例如 L01" size="10"></div>
<div class="row"><span>會說的腔調(可複選):</span>
<label><input type="checkbox" name="dia" value="四縣"> 四縣</label>
<label><input type="checkbox" name="dia" value="海陸"> 海陸</label>
<label><input type="checkbox" name="dia" value="其他"> 其他</label><input type="text" id="dia_other" placeholder="其他腔調,例如大埔" size="12"></div>
<div class="row"><span>客語使用頻率:</span><select id="freq"><option>每天</option><option>每週</option><option>偶爾</option><option>很少</option></select></div>
<div class="err" id="msg_intro"></div>
<button class="primary" id="btn_start" type="button">開始</button></div>
<div id="test" style="display:none"><div id="prog"></div><div class="card" id="trial"></div>
<div class="err" id="msg_trial"></div>
<div class="row"><button class="primary" id="btn_next" type="button">下一題</button></div></div>
<div class="card" id="done" style="display:none"><p>完成了,謝謝!請把結果傳給研究者,下面三種方式選一種就好:</p>
<div class="row"><button class="primary" id="btn_share" type="button">分享結果(手機:用 LINE / Email 傳)</button></div>
<div class="row"><button id="btn_download" type="button">下載結果 CSV</button><button id="btn_copy" type="button">複製結果文字</button></div>
<div class="err" id="msg_done"></div>
<p style="font-size:14px;color:#666">如果上面都不行,請長按下面的文字全選、複製,貼到 LINE 或 Email 傳給研究者。</p>
<textarea id="csv" readonly></textarea></div>
<script>
var TRIALS = __TRIALS__, KEY = 'hakka_ab___TESTID__';
var i = 0, t0 = 0, ans = [];
function $(id){return document.getElementById(id)}
function say(id, t){$(id).textContent = t}
function save(){try{localStorage.setItem(KEY, JSON.stringify({lid:$('lid').value, dia:dialects(), other:$('dia_other').value,
  freq:$('freq').value, i:i, ans:ans}))}catch(e){}}
function load(){try{return JSON.parse(localStorage.getItem(KEY) || 'null')}catch(e){return null}}
function clean(s){return String(s).replace(/[,"\\r\\n]/g, ' ').trim()}
function scale(name){var h='<div class="scale">';for(var v=-3;v<=3;v++){h+='<label><input type="radio" name="'+name+'" value="'+v+'">'+(v>0?'+'+v:v)+'</label>'}
  return h+'</div><div class="hint"><span>1 好很多</span><span>一樣</span><span>2 好很多</span></div>'}
function show(){var t=TRIALS[i];say('prog','第 '+(i+1)+' / '+TRIALS.length+' 題');say('msg_trial','');
  $('trial').innerHTML='<div class="row"><span class="lab">1</span><audio controls preload="auto" src="stimuli/'+t.id+'_1.wav"></audio></div>'+
  '<div class="row"><span class="lab">2</span><audio controls preload="auto" src="stimuli/'+t.id+'_2.wav"></audio></div>'+
  '<p>(一)整體哪一段比較自然?</p>'+scale('q1')+'<p>(二)入聲字哪一段比較自然?</p>'+scale('q2');
  var au=$('trial').querySelectorAll('audio');for(var k=0;k<au.length;k++){au[k].addEventListener('error',function(){
    say('msg_trial','音檔載入失敗:請確認 stimuli 資料夾和 index.html 放在一起(整個資料夾解壓縮後再開)。')})}
  window.scrollTo(0,0);t0=Date.now()}
function dialects(){var out=[],c=document.querySelectorAll('input[name=dia]:checked');
  for(var k=0;k<c.length;k++){out.push(c[k].value==='其他'?'其他:'+(clean($('dia_other').value)||'未填'):c[k].value)}return out.join('+')}
function begin(){$('intro').style.display='none';$('resume').style.display='none';$('test').style.display='block';
  if(i<TRIALS.length)show();else finish()}
function start(){if(!clean($('lid').value)){say('msg_intro','請先填代號');return}
  if(!dialects()){say('msg_intro','請勾選會說的腔調(可複選)');return}
  i=0;ans=[];save();begin()}
function val(n){var e=document.querySelector('input[name='+n+']:checked');return e?e.value:null}
function next(){var a=val('q1'),b=val('q2');if(a===null||b===null){say('msg_trial','兩題都要作答');return}
  ans.push({trial:TRIALS[i].id,q1:a,q2:b,sec:((Date.now()-t0)/1000).toFixed(1)});i++;save();
  if(i<TRIALS.length)show();else finish()}
function csv(){var id=clean($('lid').value),d=dialects(),f=$('freq').value,s='listener,dialect,frequency,trial,q1,q2,seconds\\n';
  for(var k=0;k<ans.length;k++){var r=ans[k];s+=[id,d,f,r.trial,r.q1,r.q2,r.sec].join(',')+'\\n'}return s}
function fname(){return 'listening_'+(clean($('lid').value).replace(/\\s+/g,'_')||'listener')+'.csv'}
function finish(){$('test').style.display='none';$('done').style.display='block';$('csv').value=csv();window.scrollTo(0,0)}
function download(){var a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\\ufeff'+csv()],{type:'text/csv'}));
  a.download=fname();document.body.appendChild(a);a.click();a.remove()}
function share(){try{var file=new File(['\\ufeff'+csv()],fname(),{type:'text/csv'});
  if(navigator.canShare&&navigator.canShare({files:[file]})){navigator.share({files:[file],title:fname()}).catch(function(){});return}
  if(navigator.share){navigator.share({title:fname(),text:csv()}).catch(function(){});return}}catch(e){}
  say('msg_done','這個瀏覽器不支援分享,請改用「下載」或「複製」。')}
function copy(){var t=$('csv');t.focus();t.select();var ok=false;try{ok=document.execCommand('copy')}catch(e){}
  if(navigator.clipboard){navigator.clipboard.writeText(t.value).then(function(){say('msg_done','已複製,請貼到 LINE 或 Email。')},
    function(){say('msg_done',ok?'已複製,請貼到 LINE 或 Email。':'無法自動複製,請長按文字框全選複製。')})}
  else say('msg_done',ok?'已複製,請貼到 LINE 或 Email。':'無法自動複製,請長按文字框全選複製。')}
$('nojs').style.display='none';
$('btn_start').addEventListener('click',start);$('btn_next').addEventListener('click',next);
$('btn_download').addEventListener('click',download);$('btn_share').addEventListener('click',share);$('btn_copy').addEventListener('click',copy);
var prev=load();
if(prev&&prev.ans&&prev.ans.length){$('resume').style.display='block';
  $('btn_resume').addEventListener('click',function(){$('lid').value=prev.lid||'';$('dia_other').value=prev.other||'';$('freq').value=prev.freq||'每天';
    var parts=(prev.dia||'').split('+');var c=document.querySelectorAll('input[name=dia]');
    for(var k=0;k<c.length;k++){for(var m=0;m<parts.length;m++){if(parts[m]===c[k].value||(c[k].value==='其他'&&parts[m].indexOf('其他')===0))c[k].checked=true}}
    ans=prev.ans;i=prev.i;begin()});
  $('btn_restart').addEventListener('click',function(){try{localStorage.removeItem(KEY)}catch(e){}$('resume').style.display='none'})}
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="系統 A 的 c_eval 輸出資料夾(含 wav/、bounds/)")
    ap.add_argument("--b", required=True, help="系統 B 的 c_eval 輸出資料夾")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--n-identical", type=int, default=2)
    ap.add_argument("--gen-seed", type=int, default=1234)
    ap.add_argument("--min-entering", type=int, default=2)
    args = ap.parse_args()

    import soundfile as sf
    A, B = Path(args.a).expanduser(), Path(args.b).expanduser()
    out = Path(args.out).expanduser()
    stems = sorted({p.name.rsplit("_g", 1)[0] for p in (A / "wav").glob(f"*_g{args.gen_seed}.wav")}
                   & {p.name.rsplit("_g", 1)[0] for p in (B / "wav").glob(f"*_g{args.gen_seed}.wav")})
    if not stems:
        sys.exit("★ 兩個資料夾沒有共同的合成音檔:先對兩個系統都跑 c_eval.py --synth")

    def n_et(stem):
        b = json.load(open(A / "bounds" / f"{stem}.json", encoding="utf-8"))
        return sum(1 for s in b["syllables"] if s["entering"])

    rng = random.Random(C.SPLIT_SEED)
    rich = [s for s in stems if n_et(s) >= args.min_entering]
    pool = rich if len(rich) >= args.n else stems
    chosen = sorted(rng.sample(pool, min(args.n, len(pool))))
    ident = rng.sample(chosen, min(args.n_identical, len(chosen)))
    print(f"共同句子 {len(stems)} 句,入聲 ≥ {args.min_entering} 的 {len(rich)} 句;抽 {len(chosen)} 句 + {len(ident)} 題相同檢查題")

    if out.exists():
        shutil.rmtree(out)
    stim = out / "listening_test" / "stimuli"
    stim.mkdir(parents=True)
    items = [(s, "AB") for s in chosen] + [(s, "same") for s in ident]
    rng.shuffle(items)
    key, trials = [], []
    for k, (stem, kind) in enumerate(items, 1):
        tid = f"t{k:02d}"
        a, sr = sf.read(A / "wav" / f"{stem}_g{args.gen_seed}.wav")
        b, _ = sf.read(B / "wav" / f"{stem}_g{args.gen_seed}.wav")
        if kind == "same":
            pair, sys1, sys2 = (a, a), "A", "A"
        elif rng.random() < 0.5:
            pair, sys1, sys2 = (a, b), "A", "B"
        else:
            pair, sys1, sys2 = (b, a), "B", "A"
        for j, x in enumerate(pair, 1):
            sf.write(stim / f"{tid}_{j}.wav", rms_normalize(x), sr)
        key.append({"trial": tid, "stem": stem, "kind": kind, "pos1": sys1, "pos2": sys2,
                    "n_entering": n_et(stem)})
        trials.append({"id": tid})
    html = HTML.replace("__TRIALS__", json.dumps(trials)).replace("__TESTID__", out.name)
    (out / "listening_test" / "index.html").write_text(html, encoding="utf-8")
    with open(out / "key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    json.dump({"A": str(A), "B": str(B), "gen_seed": args.gen_seed, "n": len(chosen), "n_identical": len(ident)},
              open(out / "trials.json", "w"), indent=2, ensure_ascii=False)
    print(f"→ 給聽者:{out / 'listening_test'}(整個資料夾,用瀏覽器開 index.html)")
    print(f"→ 自己保留:{out / 'key.csv'}(答案對照,不要給聽者)")
    print(f"   收回 CSV 後:python e_analyze.py --key {out / 'key.csv'} --responses <放 CSV 的資料夾>")


if __name__ == "__main__":
    main()
