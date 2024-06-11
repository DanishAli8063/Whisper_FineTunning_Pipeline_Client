import os
import requests
import json
from similarity import SimilarityFinder
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
from pydub import AudioSegment

temp_folder = "./Input"


class Handler():
    def __init__(self):
        self.similarity = SimilarityFinder()
        self.bot_sentences = self.similarity.bot_sentences
        self.output_path = "./Output"

    def process_all_wav_files(self, directory):
        """
        Processes all .wav files in a given directory.

        Args:
            directory (str): Path to the directory containing .wav files.
        """
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if file_path.endswith('.wav'):
                print(f"Processing file: {file_path}")
                self.process_new_file(file_path)

    def process_new_file(self, file_path):
        """
        Processes a new audio file by uploading it to a server, receiving a response,
        and saving the JSON data if the response is successful.

        Args:
            file_path (str): Path to the audio file.

        """
        if file_path.endswith('.wav'):
            try:
                with open(file_path, 'rb') as file:
                    response = requests.post('http://113.203.209.145:9001/uploadfile', files={'file': file})
            except requests.RequestException as e:
                print(f"Error uploading file: {e}")
                return

            if response.status_code == 200:
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    print("Failed to decode JSON from the response")
                    return

                if data.get("status"):
                    print("Returned from server side successfully!")
                    self.save_json(data.get("msg"), file_path)
                else:
                    print("Server returned an error status.")
            else:
                print(f"Server returned status code {response.status_code}")
        else:
            print("File is not a .wav file.")

    def save_json(self, data, original_file_path):
        """
        Processes the input JSON data, extracts necessary information,
        and performs audio chunking and transcript processing.

        Args:
            data (dict): Dictionary containing transcript and timestamp information.
            original_file_path (str): Path to the original audio file.

        """
        if not isinstance(data, dict):
            print("Data is not a dictionary.")
            return

        # Extract main text from each speaker
        processed_data = {
            key: value["trascript"]
            for key, value in data.items()
            if isinstance(value, dict) and "trascript" in value
        }

        # Extract timestamps
        time_stamps = {
            key: {
                "time_stamp_start": value["time_stamp_start"],
                "time_stamp_end": value["time_stamp_end"]
            }
            for key, value in data.items()
            if isinstance(value, dict) and "time_stamp_start" in value and "time_stamp_end" in value
        }

        print("time_stamps:", time_stamps)

        # Chunk the audio based on timestamps
        chunks = self.chunking(time_stamps)
        self.split_audio(original_file_path, chunks, processed_data)

        # Process transcripts if data is not empty
        if processed_data:
            self.process_transcripts(processed_data, original_file_path)
            print("Converted successfully.")
        else:
            print("No valid transcript data found.")

    def chunking(self, time_stamps):
        """
        Converts timestamps into chunks of (start, end) tuples in milliseconds.

        Args:
            time_stamps (dict): Dictionary with keys representing chunks and values 
                                containing 'time_stamp_start' and 'time_stamp_end'.

        Returns:
            list: List of tuples where each tuple contains start and end times in milliseconds.
        """
        # Initialize chunks list
        chunks = []

        # Iterate over the timestamps dictionary to create chunks
        for key, value in time_stamps.items():
            start = int(value['time_stamp_start'] * 1000)  # Convert to milliseconds
            end = int(value['time_stamp_end'] * 1000)  # Convert to milliseconds
            chunks.append((start, end))

        return chunks

    def split_audio(self, file_path, chunks, processed_data):
        """
        Splits an audio file into smaller chunks and saves them along with their transcripts.

        Args:
            file_path (str): Path to the input audio file.
            chunks (list of tuples): List of (start, end) timestamps in milliseconds for each chunk.
            processed_data (dict): Dictionary containing the transcripts for each chunk.

        """
        # Load the audio file
        try:
            audio = AudioSegment.from_wav(file_path)
        except Exception as e:
            raise ValueError(f"Error loading audio file: {e}")

        filename_with_extension = os.path.basename(file_path)
        filename, _ = os.path.splitext(filename_with_extension)

        # Create directory structure
        audio_folder_path = os.path.join(self.output_path, filename)
        chunks_folder_path = os.path.join(audio_folder_path, "Chunks")
        json_folder_path = os.path.join(audio_folder_path, "JSON")

        os.makedirs(chunks_folder_path, exist_ok=True)
        os.makedirs(json_folder_path, exist_ok=True)

        # Create a dictionary to hold chunk filenames and their transcripts
        chunk_transcripts = {}

        # Create chunks based on start and end timestamps
        for i, (start, end) in enumerate(chunks):
            chunk = audio[start:end]
            chunk_filename = f"chunk_{i}.wav"
            chunk_path = os.path.join(chunks_folder_path, chunk_filename)
            try:
                chunk.export(chunk_path, format="wav")
                print(f"Chunk {i} from {start}ms to {end}ms saved as {chunk_filename}")
            except Exception as e:
                raise ValueError(f"Error exporting chunk {i}: {e}")

            # Add the chunk filename and its corresponding transcript to the dictionary
            chunk_transcripts[chunk_filename] = list(processed_data.values())[i]

        # Save the chunk transcripts dictionary as a JSON file
        json_transcripts_path = os.path.join(json_folder_path, "Chunk_Transcripts.json")
        try:
            with open(json_transcripts_path, 'w') as outfile:
                json.dump(chunk_transcripts, outfile, indent=4)
            print(f"Chunk transcripts saved to: {json_transcripts_path}")
        except Exception as e:
            raise ValueError(f"Error saving chunk transcripts: {e}")
        
    def process_transcripts(self, processed_data, original_file_path):
        """
        Processes transcripts by tagging them as 'Bot Transcript' or 'Customer Transcript'
        based on similarity and saves the tagged transcripts as a JSON file.

        Args:
            processed_data (dict): Dictionary containing transcripts keyed by speaker.
            original_file_path (str): Path to the original audio file.

        """
        # Extract speaker list and their transcripts
        speaker_list = list(processed_data.keys())
        splitted_transcript = [processed_data[speaker] for speaker in speaker_list]

        # Find indexes of bot sentences
        bot_indexes = self.similarity.similarityFinder(self.bot_sentences, splitted_transcript)
        bot_indexes = list(set(bot_indexes))

        # Tag transcripts based on whether they are bot or customer transcripts
        tagged_transcript = []
        for index, transcript in enumerate(splitted_transcript):
            speaker_tag = "Bot Transcript" if index in bot_indexes else "Customer Transcript"
            tagged_transcript.append({speaker_tag: transcript})

        # Prepare output directory
        filename_with_extension = os.path.basename(original_file_path)
        filename, _ = os.path.splitext(filename_with_extension)
        audio_folder_path = os.path.join(self.output_path, filename)
        json_folder_path = os.path.join(audio_folder_path, "JSON")
        os.makedirs(json_folder_path, exist_ok=True)

        # Save the tagged transcripts to a JSON file
        output_file_path = os.path.join(json_folder_path, "Segregated_Transcript.json")
        try:
            with open(output_file_path, 'w') as outfile:
                json.dump(tagged_transcript, outfile, indent=4)
            print(f"Transcripts saved to: {output_file_path}")
        except Exception as e:
            raise ValueError(f"Error saving transcripts: {e}")


app = FastAPI()

# Assuming the Handler class and its methods are correctly defined as you provided

handler = Handler()  # Create an instance of your Handler        

handler.process_all_wav_files(temp_folder)
