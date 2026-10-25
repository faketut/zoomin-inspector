#!/usr/bin/env bash
# Build video/out/zoomin-combined-demo.mp4 — the single submission video:
# narrated slide segments 1-4, a LIVE DEMO title card, the live UI recording
# (tightly re-cut from live/raw/*.webm, 1.32x), then slide segments 5-6.
# Total must stay <= 300s (competition cap); current build: 299.0s.
set -euo pipefail
cd "$(dirname "$0")/.."   # video/

# 1) Title card (1.6s, silent)
ffmpeg -y -v error -f lavfi -i color=c=black:s=1920x1080:r=30 \
  -f lavfi -i anullsrc=r=24000:cl=mono \
  -vf "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='LIVE DEMO — the real app, recorded live':fontcolor=white:fontsize=44:x=(w-text_w)/2:y=(h-text_h)/2" \
  -t 1.6 -c:v libx264 -pix_fmt yuv420p -c:a aac -b:a 128k live/gap.mp4

# 2) Live section: re-cut acts from the raw Playwright webm, speed 1.5x,
#    scale/pad to 1920x1080@30, silent audio to match the slide segments.
#    Trims follow the recorder's act timeline (~62s per act: load, upload,
#    inspect, 45s marker wait, hold, scroll, reload). Re-derive them from
#    scene changes if the recorder's waits change. Speed chosen so the
#    combined video stays under the 300s cap with the longer seg6 (2026-10-25).
W=$(ls -t live/raw/*.webm | head -1)
ffmpeg -y -v error -i "$W" -f lavfi -i anullsrc=r=24000:cl=mono -filter_complex "\
[0:v]trim=2:21,setpts=PTS-STARTPTS[a1];\
[0:v]trim=53.5:62.5,setpts=PTS-STARTPTS[a1b];\
[0:v]trim=64.5:83.5,setpts=PTS-STARTPTS[a2];\
[0:v]trim=115.6:124.6,setpts=PTS-STARTPTS[a2b];\
[0:v]trim=126.6:145.6,setpts=PTS-STARTPTS[a3];\
[0:v]trim=177:185.5,setpts=PTS-STARTPTS[a3b];\
[a1][a1b][a2][a2b][a3][a3b]concat=n=6:v=1:a=0,setpts=PTS/1.5,scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30[v]" \
  -map "[v]" -map 1:a -shortest -c:v libx264 -preset medium -crf 20 \
  -pix_fmt yuv420p -c:a aac -b:a 128k live/live_mid.mp4

# 3) Join everything (re-encode for uniform parameters). Encode to a temp
#    file first: a killed run must never leave a truncated mp4 at the
#    delivery path (learned 2026-10-24, when a restart did exactly that).
printf "file '%s'\n" "$PWD/segments/seg1.mp4" "$PWD/segments/seg2.mp4" \
  "$PWD/segments/seg3.mp4" "$PWD/segments/seg4.mp4" "$PWD/live/gap.mp4" \
  "$PWD/live/live_mid.mp4" "$PWD/segments/seg5.mp4" "$PWD/segments/seg6.mp4" \
  > live/combined_list.txt
ffmpeg -y -v error -f concat -safe 0 -i live/combined_list.txt \
  -c:v libx264 -preset fast -crf 21 -pix_fmt yuv420p -c:a aac -b:a 128k \
  live/combined_tmp.mp4
mv live/combined_tmp.mp4 out/zoomin-combined-demo.mp4
ffprobe -v error -show_entries format=duration -of csv=p=0 out/zoomin-combined-demo.mp4
