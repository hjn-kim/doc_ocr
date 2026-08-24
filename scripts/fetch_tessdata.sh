#!/usr/bin/env bash
# tesseract 언어 파일(traineddata)을 홈 디렉터리에 받는다. sudo 가 필요 없다.
#
#   bash scripts/fetch_tessdata.sh            # ~/tessdata 에 받는다
#   bash scripts/fetch_tessdata.sh /some/dir  # 받을 곳을 직접 지정
#
# 받은 뒤 벤치마크에 경로를 알려준다.
#   python src/run_benchmark.py --device gpu:0 \
#     --tesseract-config "--oem 3 --psm 3 --tessdata-dir $HOME/tessdata"
set -euo pipefail

DEST="${1:-$HOME/tessdata}"
BEST="https://github.com/tesseract-ocr/tessdata_best/raw/main"
# tgl(타갈로그)만 tessdata_best 에 없다. 구버전 저장소에서 받는다.
LEGACY="https://github.com/tesseract-ocr/tessdata/raw/main"

mkdir -p "$DEST"
for lang in eng kor chi_sim rus vie uzb osd; do
    if [ -s "$DEST/$lang.traineddata" ]; then
        echo "이미 있음  $lang"
        continue
    fi
    echo "받는 중    $lang"
    curl -fL --retry 3 -o "$DEST/$lang.traineddata" "$BEST/$lang.traineddata"
done
if [ ! -s "$DEST/tgl.traineddata" ]; then
    echo "받는 중    tgl (구버전 저장소)"
    curl -fL --retry 3 -o "$DEST/tgl.traineddata" "$LEGACY/tgl.traineddata"
else
    echo "이미 있음  tgl"
fi

echo
echo "받은 곳: $DEST"
ls -1 "$DEST"
echo
echo "확인:  TESSDATA_PREFIX=$DEST tesseract --list-langs"
