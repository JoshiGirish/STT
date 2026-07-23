"""
Transcribes an audio file using the faster-whisper model.

This script initializes a large Whisper model, transcribes a specified audio file,
and prints the duration and text content of each segment.
"""
from faster_whisper import WhisperModel
import sys
import os
import requests
import json

# --- Configuration ---
MODEL_NAME = "large-v3"
AUDIO_FILE_PATH = "audio.wav"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

# LLM Configuration
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://localhost:8080/v1/chat/completions")


# 1. Initialize the Whisper Model
print(f"Loading Whisper model '{MODEL_NAME}' on {DEVICE} device...")
try:
    model = WhisperModel(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)


# 2. Check for audio file existence
if not os.path.exists(AUDIO_FILE_PATH):
    print(f"Error: Audio file '{AUDIO_FILE_PATH}' not found. Please ensure the audio file is in the current directory.")
    sys.exit(1)


# 3. Transcribe the audio
print(f"Starting transcription of {AUDIO_FILE_PATH}...")
segments, info = model.transcribe(
    AUDIO_FILE_PATH,
    language="hi",
    beam_size=15,
    patience=2,
    temperature=0,
    condition_on_previous_text=True
)

# 4. Refine and Output Results
def refine_transcription_with_llm(segments, llm_endpoint=None, max_retries=3):
    """
    Sends all transcribed segments to a text refinement LLM.
    
    Args:
        segments: List of Whisper segment objects with 'text' attribute
        llm_endpoint: URL of the locally hosted LLM (default: http://localhost:8080/v1/chat/completions)
        max_retries: Maximum number of retry attempts for the LLM request
    
    Returns:
        Refined/corrected transcription text
    """
    
    if llm_endpoint is None:
        llm_endpoint = LLM_ENDPOINT
    
    # Collect all text from segments
    full_text = ""
    print("\n" + "="*40)
    print(f"Transcription Duration: {info.duration:.2f} seconds")
    print("="*40 + "\n")
    print("--- Transcribed Segments ---")
    for i, segment in enumerate(segments):
        print(f"Segment {i+1}:")
        print(f"  Start: {segment.start:.2f}s | End: {segment.end:.2f}s")
        segTxt = segment.text
        print(f"  Text: {segTxt}")
        full_text = "\n".join([full_text, segTxt])
    
    # Make the LLM request using OpenAI-compatible API format
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",  # Default model for Ollama
        "messages": [
            {
                "role": "user",
                "content": f"""
                आप हिंदी भाषा के विशेषज्ञ संपादक हैं। आपको दिया गया पाठ Speech-to-Text (STT) प्रणाली द्वारा तैयार किया गया है, इसलिए इसमें ध्वन्यात्मक (phonetic) त्रुटियाँ, वर्तनी की गलतियाँ, गलत शब्द, विराम चिह्नों की त्रुटियाँ या व्याकरण संबंधी छोटी-मोटी गलतियाँ हो सकती हैं।
                आपका एकमात्र कार्य मूल अर्थ, संदर्भ और बोलने के उद्देश्य को पूरी तरह बनाए रखते हुए पाठ को शुद्ध करना है।

                नियम:
                केवल वर्तनी, व्याकरण, विराम चिह्न और ध्वन्यात्मक (phonetic) त्रुटियों को सुधारें।
                AI द्वारा गलत पहचाने गए शब्दों को संदर्भ के अनुसार सही हिंदी शब्दों से बदलें।
                मूल अर्थ, जानकारी, व्यक्ति, स्थान, संख्या, तिथि, समय या तथ्य में कोई परिवर्तन न करें।
                कोई नई जानकारी न जोड़ें।
                कोई स्पष्टीकरण, टिप्पणी, कारण या नोट न लिखें।
                सारांश, अनुवाद, पुनर्लेखन या शैली में परिवर्तन न करें।
                बोलचाल की प्राकृतिक शैली को यथासंभव बनाए रखें।
                आवश्यकता होने पर उचित विराम चिह्नों का प्रयोग करें।
                अंग्रेज़ी या तकनीकी शब्दों को उनके मूल रूप में ही रखें, जब तक कि वे स्पष्ट रूप से गलत पहचाने गए शब्द न हों।
                यदि संदर्भ स्पष्ट न हो, तो अनुमान न लगाएँ। यथासंभव मूल पाठ को सुरक्षित रखें।
                उत्तर में केवल अंतिम संशोधित हिंदी पाठ दें। कोई टिप्पणी न लिखें।

                विशेष ध्यान दें:
                समान उच्चारण वाले शब्दों के बीच होने वाले भ्रम (जैसे: है/हैं, करता/करते, नहीं/नहीं है)।
                गलत कारक, लिंग, वचन और क्रिया-रूप।
                लगातार बोलने के कारण गलती से जुड़े हुए या टूटे हुए शब्द।
                गलत नामों को यदि संदर्भ के आधार पर स्पष्ट रूप से सुधारा जा सकता हो तो सुधारें, अन्यथा उन्हें यथावत रखें।
                अनावश्यक पुनरावृत्ति या भराव वाले शब्द (जैसे: मतलब मतलब, तो तो) केवल तभी हटाएँ जब वे स्पष्ट रूप से STT की त्रुटि हों।
                विराम चिह्न और वाक्य संरचना को स्वाभाविक बनाएँ, लेकिन पाठ का पुनर्लेखन न करें।

                उद्देश्य:
                मूल पाठ के अर्थ में किसी भी प्रकार का परिवर्तन किए बिना उसे शुद्ध, सटीक, स्वाभाविक और पढ़ने योग्य हिंदी में प्रस्तुत करना।
                
                मूल पाठ:
                {full_text}
                """   
            }
        ],
        "temperature": 0.7,
        "max_tokens": 4096
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(llm_endpoint, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            result = json.loads(response.text)
            # Handle different possible response structures
            refined_text = result.get("choices", [{}])[0].get("message", {}).get("content", full_text)
            
            print(f"\nLLM refinement complete. Attempt {attempt + 1}/{max_retries}")
            print("\n" + "="*40)   
            print(refined_text)
            print("\n" + "="*40)
            return refined_text
            
        except requests.exceptions.RequestException as e:
            print(f"LLM request failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print("LLM refinement failed after all retries. Returning original text.")
                return full_text
        except (KeyError, IndexError, TypeError):
            print(f"LLM returned unexpected response format. Returning original text.")
            return full_text


refine_transcription_with_llm(segments)

print("\nDone")

# Flush stdout and exit successfully
sys.stdout.flush()
sys.exit(0)