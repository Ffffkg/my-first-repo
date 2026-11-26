# 給其他腳本 source:找出各系統三個種子的測試集評估資料夾(依各自在驗證集選好的取整方式 / 訓練步數)。
#   sys_dirs <系統> <腔調>   → 逗號分隔的資料夾(找不到的種子略過,並在 stderr 提示)
#   系統:base、c2、c6(Matcha-TTS,4 萬步)、vbase、vad、vx、vxad(VITS,20 萬步)、
#         raon、raon_oracle、raon_40k、raon_40k_oracle(Raon-OpenTTS,步數由 r_pick.py 在驗證集選)
# 需要的變數:HERE(本資料夾)、E(eval)、H(hakka_tts 根目錄)
_sfx() { [ "$1" = 1234 ] && echo "" || echo "_s$1"; }
sys_dirs() {
  local k=$1 d=$2 s st val dir pick tag oracle out=""
  for s in 1234 123 2026; do
    case $k in
      base|c2|c6) st=40000; val=c6val_${k}_${d}$(_sfx "$s") ;;
      vbase|vad|vx|vxad) st=200000; val=c6val_${k}_${d}$(_sfx "$s")_step200000 ;;
      raon*)
        oracle=""; [ "${k%_oracle}" != "$k" ] && oracle=_oracle
        tag=${k#raon}; tag=${tag%_oracle}
        pick=$H/exp_c/r/pick${tag}_${d}$(_sfx "$s").md
        [ -f "$pick" ] || { echo "  (沒有 $pick)" >&2; continue; }
        dir=raon${tag}_${d}$(_sfx "$s")_step$(tail -1 "$pick")$oracle
        [ -f "$E/$dir/syllables_wave.csv" ] && out+="$E/$dir," || echo "  (找不到 $dir)" >&2
        continue ;;
      *) echo "  (不認得的系統 $k)" >&2; return 1 ;;
    esac
    dir=${k}_${d}$(_sfx "$s")_step$st
    if [ -f "$H/exp_c/c3/$val/recs.pkl" ]; then
      pick=$(python "$HERE/c6_pick_rounding.py" --val "$H/exp_c/c3/$val/recs.pkl" 2>/dev/null | tail -1)
      [ "$pick" = cum ] && dir=${dir}_cum
    elif [ -f "$E/${dir}_cum/syllables_wave.csv" ]; then
      [ -f "$E/$dir/syllables_wave.csv" ] && echo "  (★ $dir 與 ${dir}_cum 都在、又沒有驗證集紀錄,取 ${dir}_cum)" >&2
      dir=${dir}_cum
    fi
    [ -f "$E/$dir/syllables_wave.csv" ] && out+="$E/$dir," || echo "  (找不到 $dir)" >&2
  done
  echo "${out%,}"
}
