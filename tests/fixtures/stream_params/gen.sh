#!/bin/sh
# Regenerates the Frame::StreamParams fixtures with black-box
# ffmpeg/libx265: independently encoded segments spliced byte-wise, each
# opening with an IDR picture, so every segment boundary activates a new
# SPS. bframes=2 keeps pictures in the reorder buffer across each
# boundary, and the second IRAP picture of each segment is a CRA picture
# with a RASL picture. Records, per output frame of the spliced stream, the size /
# pix_fmt ffprobe reports and the MD5 of the raw planes of a black-box
# decode of the segment alone (expected.txt: width,height,pix_fmt,md5).
# Run from this directory: sh gen.sh
set -e
enc() { # name pix_fmt size rate
  ffmpeg -v error -y -f lavfi -i "testsrc2=size=$3:rate=$4:duration=0.32" \
    -frames:v 8 -pix_fmt "$2" -c:v libx265 -preset ultrafast \
    -x265-params "log-level=none:bframes=2:keyint=5" -f hevc "$1.hevc"
}
# a -> b: size + bit depth; b -> c: identical shape (no change);
# c -> d: size + chroma format + depth; d -> e: conformance window only
# (same 64x48 coded size, 64x46 output); e -> f: frame rate only;
# f -> g: bit depth only.
enc seg_a yuv420p     64x48 25
enc seg_b yuv420p10le 48x32 25
enc seg_c yuv420p10le 48x32 25
enc seg_d yuv422p     64x48 25
enc seg_e yuv422p     64x46 25
enc seg_f yuv422p     64x46 50
enc seg_g yuv422p10le 64x46 50
cat seg_a.hevc seg_b.hevc seg_c.hevc seg_d.hevc seg_e.hevc seg_f.hevc seg_g.hevc > spliced7.hevc
: > md5.tmp
for s in a b c d e f g; do
  ffmpeg -v error -i seg_$s.hevc -c:v rawvideo -f framemd5 - | grep -v '^#' \
    | awk -F', *' '{print $6}' >> md5.tmp
done
ffprobe -v error -select_streams v:0 -show_entries frame=width,height,pix_fmt \
  -of csv=p=0 spliced7.hevc | cut -d, -f1-3 > shape.tmp
paste -d, shape.tmp md5.tmp > expected.txt
rm -f seg_*.hevc md5.tmp shape.tmp
