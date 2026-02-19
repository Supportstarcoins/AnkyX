# YouTube import

## Dependencies
- `yt-dlp`
- `ffmpeg`

Install (Linux):
```bash
sudo apt-get install ffmpeg
pip install yt-dlp
```

## How it works
- URL hash (`sha256`) is stored in `youtube_imports` to prevent duplicate imports.
- Clips are created by ffmpeg and stored in `data/media/youtube/<video_id>/clip_XXXX.mp4`.
- Clip metadata is stored in `youtube_clips` and can be attached to cards through `cards.media_json`.

## Debugging
- If ffmpeg is missing you get: `ffmpeg не найден...`.
- Verify DB rows:
```sql
SELECT video_id, url, sha256 FROM youtube_imports;
SELECT clip_id, path FROM youtube_clips;
```
