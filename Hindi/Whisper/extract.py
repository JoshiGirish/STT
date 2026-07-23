import ffmpeg

(
    ffmpeg
    .input("video.mp4")
    .filter("highpass", f=80)
    .filter("lowpass", f=7500)
    .filter("loudnorm")
    .output(
        "audio.wav",
        acodec="pcm_s16le",
        ac=1,
        ar=16000,
    )
    .overwrite_output()
    .run()
)