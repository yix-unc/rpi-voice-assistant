import os
import time
import struct
import sys
import yaml
import pvporcupine
from google.cloud import speech_v1 as speech

from queue import Queue
from threading import Thread

import audio
from voiceflow import Voiceflow
from gptflow import GptFlow

RATE = 16000
language_code = "en-US"  # a BCP-47 language tag

def load_config(config_file="config.yaml"):
    with open(config_file) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        return yaml.load(file, Loader=yaml.FullLoader)


def main():
    config = load_config()

    # Wakeword setup
    porcupine = pvporcupine.create(config["porcupine_access_key"], keywords=config["wakewords"])
    CHUNK = porcupine.frame_length  # 512 entries

    # Voiceflow setup
    vf = GptFlow(config["openai_key"])

    # Google ASR setup
    google_asr_client = speech.SpeechClient()
    google_asr_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=google_asr_config, interim_results=True
    )

    gap_filling_sentences = vf.thinking_words()

    speaker_queue = Queue(maxsize=0)
    def process_speaker_queue(q):
        while True:
            text = q.get()
            start = time.time()
            audio.speak(text)
            q.task_done()
    speaker_worker = Thread(target=process_speaker_queue, args=(speaker_queue,))
    speaker_worker.setDaemon(True)
    speaker_worker.start()
    
    gap_filling_sentences_id = 0
    with audio.MicrophoneStream(RATE, CHUNK) as stream:
        print("Starting voice assistant!")
        while True:
            if gap_filling_sentences_id >= len(gap_filling_sentences):
                gap_filling_sentences = vf.thinking_words()
                gap_filling_sentences_id = 0

            pcm = stream.get_sync_frame()
            if len(pcm) == 0:
                # Protects against empty frames
                continue
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            keyword_index = porcupine.process(pcm)

            if keyword_index >= 0:
                print("Wakeword Detected")
                audio.beep()
                end = False
                while not end:
                    stream.start_buf()  # Only start the stream buffer when we detect the wakeword
                    audio_generator = stream.generator()
                    requests = (
                        speech.StreamingRecognizeRequest(audio_content=content)
                        for content in audio_generator
                    )
                    start = time.time()

                    responses = google_asr_client.streaming_recognize(streaming_config, requests)
                    print("voice recognition time {}".format(time.time() - start))
                    print(responses)
                    # Now, put the transcription responses to use.
                    utterance = audio.process(responses)
                    stream.stop_buf()

                    print("process voice response time {}".format(time.time() - start))
                    speaker_queue.put(gap_filling_sentences[gap_filling_sentences_id])
                    gap_filling_sentences_id += 1
                    
                    # Send request to VF service and get response
                    response = vf.interact(utterance)
                    print("get gpt response time {}".format(time.time() - start))
                    speaker_queue.put(response)
                    continue

                    for item in response["trace"]:
                        if item["type"] == "speak":
                            payload = item["payload"]
                            message = payload["message"]
                            print("Response: " + message)
                            audio.play(payload["src"])
                        elif item["type"] == "end":
                            print("-----END-----")
                            vf.clear_state()
                            end = True
                            audio.beep()
    speaker_worker.join()

if __name__ == "__main__":
    main()
