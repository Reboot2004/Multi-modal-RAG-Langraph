from ingestion.audio_video_loader import AudioVideoLoader


def run_one(path: str, source: str):
    loader = AudioVideoLoader()
    out = loader.parse(path, source_name=source)
    debug = out.get("debug", {})
    bench = debug.get("benchmark", {})
    print("source:", source)
    print("chunks:", len(out.get("chunks", [])))
    print("detected_language:", debug.get("detected_language"))
    print("decode_seconds:", bench.get("decode_seconds"))
    print("segments:", bench.get("segments"))
    print("---")


if __name__ == "__main__":
    run_one("data/test_media/audio_hi/Aditya.ogg", "Aditya.ogg")
    run_one("data/test_media/video_hi/Arnab Goswami interview by Devang Bhatt.webm", "Arnab.webm")
