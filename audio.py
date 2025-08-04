import json
import os
from gtts import gTTS
from pydub import AudioSegment
from pydub.effects import normalize, speedup


def combine_audio_files(file_paths, output_path):
    """
    Concatenate multiple audio files into a single output file.
    """
    combined_audio = AudioSegment.empty()
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                audio = AudioSegment.from_file(file_path)
                combined_audio += audio
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")
    combined_audio = normalize(combined_audio)
    combined_audio = speedup(combined_audio, playback_speed=1.3)
    file_ext = os.path.splitext(output_path)[1][1:]
    combined_audio.export(output_path, format=file_ext)


def create_audio_from_json(json_data, output_file="podcast.mp3"):
    """
    Convert JSON podcast script to audio with two alternating speakers
    and combine into one audio file.
    """
    podcast_data = json.loads(json_data) if isinstance(json_data, str) else json_data

    temp_dir = "temp_audio"
    audio_files = []
    os.makedirs(temp_dir, exist_ok=True)
    def text_to_audio(text, filename, voice_tld):
        if not text or not isinstance(text, str):
            print(f"Skipping empty or invalid text for {filename}")
            return None
        try:
            tts = gTTS(text=text, lang='en', tld=voice_tld)
            tts.save(filename)
            print(f"Generated: {filename} with voice {voice_tld}")
            return filename
        except Exception as e:
            print(f"Error generating audio for {filename}: {e}")
            return None
    def get_voice_tld(speaker_str):
        if "UK" in speaker_str:
            return "co.uk"
        elif "India" in speaker_str:
            return "co.in"
        else:
            return "com"
    sections = [
        "host_intro", "paper_overview", "key_insights", "methodology",
        "results", "real_world_applications", "limitations",
        "conclusion", "outro"
    ]

    for section_key in sections:
        print(f"Processing section: {section_key}...")
        section_data = podcast_data.get(section_key)

        if not section_data:
            print(f"Warning: Section '{section_key}' not found or empty in JSON data.")
            continue

        if section_key == "key_insights":
            if isinstance(section_data, list):
                for i, insight_block in enumerate(section_data):
                    if isinstance(insight_block, list):
                        for j, dialogue_item in enumerate(insight_block):
                            speaker = dialogue_item.get("speaker", "")
                            dialogue = dialogue_item.get("dialogue", "")
                            voice_tld = get_voice_tld(speaker)
                            filename = os.path.join(temp_dir, f"{section_key}_{i}_{j}.mp3")
                            audio_file = text_to_audio(dialogue, filename, voice_tld)
                            if audio_file:
                                audio_files.append(audio_file)
                    else:
                         print(f"Warning: Expected list for insight block {i} in '{section_key}', got {type(insight_block)}.")
            else:
                print(f"Warning: Expected list for '{section_key}', got {type(section_data)}.")

        elif isinstance(section_data, list):
            for i, dialogue_item in enumerate(section_data):
                 if isinstance(dialogue_item, dict):
                    speaker = dialogue_item.get("speaker", "")
                    dialogue = dialogue_item.get("dialogue", "")
                    voice_tld = get_voice_tld(speaker)
                    filename = os.path.join(temp_dir, f"{section_key}_{i}.mp3")
                    audio_file = text_to_audio(dialogue, filename, voice_tld)
                    if audio_file:
                        audio_files.append(audio_file)
                 else:
                     print(f"Warning: Expected dict for item {i} in '{section_key}', got {type(dialogue_item)}.")
        else:
            print(f"Warning: Expected list for section '{section_key}', got {type(section_data)}. Skipping.")

    print(f"\nCombining {len(audio_files)} audio files into {output_file}...")
    combine_audio_files(audio_files, output_file)

    print("Cleaning up temporary files...")
    if os.path.exists(temp_dir):
        try:
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
            print("Temporary directory removed.")
        except OSError as e:
            print(f"Error removing temporary directory {temp_dir}: {e}")
if __name__ == "__main__":
    try:
        with open('podcast_script.json', 'r') as f:
            podcast_json_data = json.load(f)
        create_audio_from_json(podcast_json_data, "ai4sg_podcast.mp3")
        print("Podcast audio created successfully.")
    except FileNotFoundError:
        print("Error: podcast_script.json not found.")
    except json.JSONDecodeError:
        print("Error: Could not decode JSON from podcast_script.json.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")