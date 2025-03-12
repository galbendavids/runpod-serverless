import base64
import faster_whisper
import tempfile
import torch
import requests

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_PAYLOAD_SIZE = 200 * 1024 * 1024

def download_file(url, max_size_bytes, output_filename, api_key=None):
    # ... (download_file function remains the same)

def transcribe(job):
    datatype = job['input'].get('type', None)
    model_name = job['input'].get('model', 'whisper-large-v3')
    is_streaming = job['input'].get('streaming', False)
    language_ = job['input'].get('language', 'en')

    if not datatype:
        yield { "error" : "datatype field not provided. Should be 'blob' or 'url'." }

    if not datatype in ['blob', 'url']:
        yield { "error" : f"datatype should be 'blob' or 'url', but is {datatype} instead." }

    api_key = job['input'].get('api_key', None)

    # Load the model directly, as it's pre-downloaded
    model = faster_whisper.WhisperModel(model_name, device=device, compute_type='float16')

    d = tempfile.mkdtemp()
    audio_file = f'{d}/audio.mp3'

    if datatype == 'blob':
        mp3_bytes = base64.b64decode(job['input']['data'])
        open(audio_file, 'wb').write(mp3_bytes)
    elif datatype == 'url':
        success = download_file(job['input']['url'], MAX_PAYLOAD_SIZE, audio_file, api_key)
        if not success:
            yield { "error" : f"Error downloading data from {job['input']['url']}" }
            return

    stream_gen = transcribe_core(model, audio_file,language_)

    if is_streaming:
        for entry in stream_gen:
            yield entry
    else:
        result = [entry for entry in stream_gen]
        yield { 'result' : result }

def transcribe_core(model, audio_file,language_):
    # ... (transcribe_core function remains the same)

runpod.serverless.start({"handler": transcribe, "return_aggregate_stream": True})