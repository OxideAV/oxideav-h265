#!/bin/sh
# Regenerates the output_params fixtures with black-box ffmpeg/libx265 and
# records what ffprobe reports for each in expected.txt. Run from this
# directory: sh gen.sh
set -e
enc() { # name pix_fmt size rate extra-x265-params
  ffmpeg -v error -y -f lavfi -i "testsrc2=size=$3:rate=$4:duration=0.2" \
    -pix_fmt "$2" -c:v libx265 -preset ultrafast \
    -x265-params "log-level=none:bframes=2:keyint=5:$5" -f hevc "$1.hevc"
}
enc yuv420p_70x46_25     yuv420p     70x46 25         "colorprim=bt709:transfer=bt709:colormatrix=bt709"
enc yuv422p_64x48_30     yuv422p     64x48 30000/1001 "range=full:colorprim=bt470bg:transfer=smpte170m:colormatrix=bt470bg"
enc yuv444p_48x32_24     yuv444p     48x32 24         ""
enc gray_64x48_25        gray        64x48 25         ""
enc yuv420p10_64x48_50   yuv420p10le 64x48 50         "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc"
enc yuv422p10_48x32_25   yuv422p10le 48x32 25         ""
enc yuv444p12_32x32_25   yuv444p12le 32x32 25         ""
enc gray10_32x32_25      gray10le    32x32 25         ""
# Mid-stream change: two coded video sequences of different shapes.
enc part_a yuv420p     64x48 25 ""
enc part_b yuv420p10le 48x32 25 ""
cat part_a.hevc part_b.hevc > change_64x48_8bit_then_48x32_10bit.hevc
rm -f part_a.hevc part_b.hevc
: > expected.txt
for f in yuv*.hevc gray*.hevc; do
  printf '%s ' "$f" >> expected.txt
  ffprobe -v error -select_streams v:0 -count_frames \
    -show_entries stream=width,height,pix_fmt,r_frame_rate,color_range,color_space,color_primaries,color_transfer,nb_read_frames \
    -of default=noprint_wrappers=1 "$f" | tr '\n' ' ' >> expected.txt
  echo >> expected.txt
done
