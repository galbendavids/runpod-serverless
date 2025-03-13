import base64
import tempfile
import torch
import requests
import os
import runpod
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if CUDA is available and set device accordingly
if torch.cuda.is_available():
    logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
    device = 'cuda'
    compute_type = 'float16'
else:
    logger.info("CUDA not available. Using CPU.")
    device = 'cpu'
    compute_type = 'int8'

# Limit file size to 200MB
MAX_PAYLOAD_SIZE = 200 * 1024 * 1024

def download_file(url, max_size_bytes, output_filename, api_key=None):
    """Download a file from a URL and save it to the specified output filename."""
    headers = {}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    
    try:
        with requests.get(url, headers=headers, stream=True) as response:
            if response.status_code != 200:
                logger.error(f"Failed to download file. Status code: {response.status_code}")
                return False
            
            # Get file size from headers or set to None if not available
            file_size = int(response.headers.get('content-length', 0))
            if file_size > max_size_bytes:
                logger.error(f"File too large: {file_size} bytes. Maximum allowed: {max_size_bytes} bytes")
                return False
            
            with open(output_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"File downloaded successfully: {output_filename}")
            return True
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return False

def transcribe_core(model, audio_file, language=None):
    """Core transcription function that yields segments"""
    logger.info("Transcribing...")
    
    # Set transcription parameters
    beam_size = 5
    
    # Handle language parameter
    transcribe_options = {}
    if language and language.lower() != "auto":
        transcribe_options["language"] = language
    
    # Run the transcription
    segments, info = model.transcribe(
        audio_file,
        beam_size=beam_size,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        **transcribe_options
    )
    
    logger.info(f"Detected language: {info.language} with probability {info.language_probability:.2f}")
    
    # Process each segment
    for segment in segments:
        yield {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "words": [{"word": word.word, "start": word.start, "end": word.end, "probability": word.probability} 
                      for word in segment.words] if segment.words else []
        }

def transcribe(job):
    """Handler function for RunPod serverless"""
    try:
        # Get input parameters
        job_input = job["input"]
        datatype = job_input.get("type")
        model_name = job_input.get("model", "whisper-large-v3")
        is_streaming = job_input.get("streaming", False)
        language = job_input.get("language", "en")
        api_key = job_input.get("api_key")
        
        # Validate input
        if not datatype:
            yield {"error": "datatype field not provided. Should be 'blob' or 'url'."}
            return
            
        if datatype not in ['blob', 'url']:
            yield {"error": f"datatype should be 'blob' or 'url', but is {datatype} instead."}
            return
        
        # Create temp directory for audio file
        temp_dir = tempfile.mkdtemp()
        audio_file = os.path.join(temp_dir, "audio.mp3")
        
        # Handle different input types
        if datatype == 'blob':
            if 'data' not in job_input:
                yield {"error": "No 'data' field provided for blob type."}
                return
                
            try:
                mp3_bytes = base64.b64decode(job_input['data'])
                with open(audio_file, 'wb') as f:
                    f.write(mp3_bytes)
            except Exception as e:
                yield {"error": f"Error decoding base64 data: {str(e)}"}
                return
                
        elif datatype == 'url':
            if 'url' not in job_input:
                yield {"error": "No 'url' field provided for url type."}
                return
                
            success = download_file(job_input['url'], MAX_PAYLOAD_SIZE, audio_file, api_key)
            if not success:
                yield {"error": f"Error downloading data from {job_input['url']}"}
                return
        
        # Load the model - we do this after file download to avoid timeout issues
        try:
            # Import here to avoid initialization issues
            import faster_whisper
            logger.info(f"Loading model: {model_name} on {device} with compute type {compute_type}")
            model = faster_whisper.WhisperModel(model_name, device=device, compute_type=compute_type)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            yield {"error": f"Error loading model: {str(e)}"}
            return
        
        # Run transcription
        try:
            stream_gen = transcribe_core(model, audio_file, language)
            
            if is_streaming:
                for entry in stream_gen:
                    yield entry
            else:
                result = [entry for entry in stream_gen]
                yield {"result": result}
                
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            yield {"error": f"Transcription error: {str(e)}"}
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        yield {"error": f"Unexpected error: {str(e)}"}

# Start the serverless handler
if __name__ == "__main__":
    logger.info("Starting RunPod Whisper Transcription Service")
    runpod.serverless.start({"handler": transcribe, "return_aggregate_stream": True})