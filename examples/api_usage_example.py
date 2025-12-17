"""
Example usage of VieNeu-TTS API
Demonstrates streaming, batch processing, and custom voice registration
"""

import requests
import json
from pathlib import Path

# API base URL
API_BASE_URL = "http://localhost:8000"

def example_1_basic_synthesis():
    """Example 1: Basic text-to-speech with preset voice"""
    print("\n=== Example 1: Basic Synthesis ===")
    
    url = f"{API_BASE_URL}/synthesize"
    
    data = {
        "text": "Xin chào, đây là ví dụ về tổng hợp giọng nói tiếng Việt.",
        "voice_name": "Vĩnh (nam miền Nam)",
        "speed": 1.0,
        "streaming": False
    }
    
    response = requests.post(url, data=data)
    
    if response.status_code == 200:
        with open("output_basic.wav", "wb") as f:
            f.write(response.content)
        print("✅ Audio saved to output_basic.wav")
    else:
        print(f"❌ Error: {response.status_code}")


def example_2_streaming():
    """Example 2: Streaming synthesis"""
    print("\n=== Example 2: Streaming Synthesis ===")
    
    url = f"{API_BASE_URL}/synthesize"
    
    data = {
        "text": "Đây là ví dụ về tổng hợp giọng nói với chế độ streaming. "
                "Âm thanh sẽ được trả về theo từng phần nhỏ.",
        "voice_name": "Hương (nữ miền Bắc)",
        "speed": 1.0,
        "streaming": True
    }
    
    response = requests.post(url, data=data, stream=True)
    
    if response.status_code == 200:
        with open("output_streaming.wav", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    print(".", end="", flush=True)
        print("\n✅ Audio saved to output_streaming.wav")
    else:
        print(f"❌ Error: {response.status_code}")


def example_3_register_custom_voice():
    """Example 3: Register custom voice for a user"""
    print("\n=== Example 3: Register Custom Voice ===")
    
    url = f"{API_BASE_URL}/register-voice"
    
    # Prepare audio file
    audio_path = "../sample/Bình (nam miền Bắc).wav"
    
    if not Path(audio_path).exists():
        print(f"❌ Audio file not found: {audio_path}")
        return
    
    with open(audio_path, "rb") as audio_file:
        files = {"audio_file": audio_file}
        data = {
            "user_id": "user123",
            "voice_name": "my_custom_voice",
            "ref_text": "Hà Nội mùa thu đẹp lắm."
        }
        
        response = requests.post(url, data=data, files=files)
    
    if response.status_code == 200:
        print(f"✅ {response.json()['message']}")
    else:
        print(f"❌ Error: {response.status_code}")


def example_4_use_registered_voice():
    """Example 4: Use registered custom voice"""
    print("\n=== Example 4: Use Registered Voice ===")
    
    url = f"{API_BASE_URL}/synthesize"
    
    data = {
        "text": "Đây là giọng nói tùy chỉnh của tôi.",
        "user_id": "user123",
        "voice_name": "my_custom_voice",
        "speed": 1.0,
        "streaming": False
    }
    
    response = requests.post(url, data=data)
    
    if response.status_code == 200:
        with open("output_custom_voice.wav", "wb") as f:
            f.write(response.content)
        print("✅ Audio saved to output_custom_voice.wav")
    else:
        print(f"❌ Error: {response.status_code}")


def example_5_batch_synthesis():
    """Example 5: Batch synthesis"""
    print("\n=== Example 5: Batch Synthesis ===")
    
    url = f"{API_BASE_URL}/batch-synthesize"
    
    texts = [
        "Câu thứ nhất.",
        "Câu thứ hai.",
        "Câu thứ ba."
    ]
    
    data = {
        "texts": texts,
        "voice_name": "Ngọc (nữ miền Bắc)",
        "speed": 1.0
    }
    
    response = requests.post(url, data=data)
    
    if response.status_code == 200:
        with open("output_batch.wav", "wb") as f:
            f.write(response.content)
        print("✅ Batch audio saved to output_batch.wav")
    else:
        print(f"❌ Error: {response.status_code}")


def example_6_speed_control():
    """Example 6: Speed control"""
    print("\n=== Example 6: Speed Control ===")
    
    url = f"{API_BASE_URL}/synthesize"
    
    text = "Đây là ví dụ về điều chỉnh tốc độ giọng nói."
    
    for speed in [0.8, 1.0, 1.2]:
        data = {
            "text": text,
            "voice_name": "Sơn (nam miền Nam)",
            "speed": speed,
            "streaming": False
        }
        
        response = requests.post(url, data=data)
        
        if response.status_code == 200:
            filename = f"output_speed_{speed}.wav"
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"✅ Audio saved to {filename} (speed: {speed}x)")
        else:
            print(f"❌ Error: {response.status_code}")


def example_7_list_voices():
    """Example 7: List available voices"""
    print("\n=== Example 7: List Available Voices ===")
    
    url = f"{API_BASE_URL}/voices"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        voices = response.json()["voices"]
        print(f"Available voices ({len(voices)}):")
        for voice in voices:
            print(f"  - {voice}")
    else:
        print(f"❌ Error: {response.status_code}")


def example_8_health_check():
    """Example 8: Health check"""
    print("\n=== Example 8: Health Check ===")
    
    url = f"{API_BASE_URL}/"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        health = response.json()
        print(f"Status: {health['status']}")
        print(f"Model loaded: {health['model_loaded']}")
        print(f"Device: {health['device']}")
        print(f"Active sessions: {health['active_sessions']}")
    else:
        print(f"❌ Error: {response.status_code}")


if __name__ == "__main__":
    print("VieNeu-TTS API Usage Examples")
    print("=" * 50)
    
    # Run examples
    try:
        example_8_health_check()
        example_7_list_voices()
        example_1_basic_synthesis()
        example_2_streaming()
        example_6_speed_control()
        
        # Uncomment to test custom voice registration
        # example_3_register_custom_voice()
        # example_4_use_registered_voice()
        
        # Uncomment to test batch synthesis (requires GPU with LMDeploy)
        # example_5_batch_synthesis()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed!")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to API server.")
        print("Please make sure the API server is running at http://localhost:8000")
        print("Start the server with: python start_services.py")
