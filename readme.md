# YouTube Downloader with AI Question Generation

A FastAPI-based web application that downloads YouTube videos, extracts frames, and generates educational comprehension questions for children ages 4-7 using OpenAI's GPT-4 Vision API.

## Features

- **YouTube Video Download**: Download videos in best quality with English subtitles
- **Frame Extraction**: Extract frames at 1-second intervals from downloaded videos
- **AI Question Generation**: Generate age-appropriate comprehension questions using video frames and transcripts
- **Real-time Processing**: WebSocket-based streaming for live progress updates
- **Web Interface**: User-friendly HTML interface for all operations

## Prerequisites

- Python 3.8 or higher
- OpenAI API key (for question generation features)
- FFmpeg (automatically handled by yt-dlp)

## Installation

### 1. Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd youtube-downloader

# Or download and extract the project files
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Required Packages

```bash
pip install fastapi uvicorn yt-dlp opencv-python pandas pillow openai jinja2 python-multipart aiofiles
```

Or create a `requirements.txt` file with the following content:

```txt
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
yt-dlp>=2023.10.13
opencv-python>=4.8.0
pandas>=2.0.0
pillow>=10.0.0
openai>=1.30.0
jinja2>=3.1.2
python-multipart>=0.0.6
aiofiles>=23.0.0
```

Then install with:
```bash
pip install -r requirements.txt
```

### 4. Set Up OpenAI API Key

You need an OpenAI API key to use the question generation features. Set it as an environment variable:

```bash
# Windows (Command Prompt)
set OPENAI_API_KEY=your_api_key_here

# Windows (PowerShell)
$env:OPENAI_API_KEY="your_api_key_here"

# macOS/Linux
export OPENAI_API_KEY="your_api_key_here"
```

Alternatively, you can enter the API key directly in the web interface when generating questions.

### 5. Create Required Directories

The application will create necessary directories automatically, but you can create the template directory structure:

```
your-project/
├── main.py
├── templates/
│   ├── download.html
│   ├── preview.html
│   ├── frames.html
│   └── questions.html
├── downloads/
└── venv/
```

## HTML Templates

You'll need to create the following HTML template files in the `templates/` directory. Here are basic examples:

### templates/download.html
```html
<!DOCTYPE html>
<html>
<head>
    <title>YouTube Downloader</title>
</head>
<body>
    <h1>YouTube Video Downloader</h1>
    <form action="/download" method="post">
        <label for="url">YouTube URL:</label>
        <input type="text" id="url" name="url" required style="width: 400px;">
        <button type="submit">Download</button>
    </form>
</body>
</html>
```

### templates/preview.html
```html
<!DOCTYPE html>
<html>
<head>
    <title>Download Result</title>
</head>
<body>
    <h1>Download Result</h1>
    {% if success %}
        <p style="color: green;">{{ message }}</p>
        <p>Video ID: {{ video_id }}</p>
        
        {% if current_video_url %}
        <video width="640" height="360" controls>
            <source src="{{ current_video_url }}" type="video/mp4">
            {% if current_sub_url %}
            <track src="{{ current_sub_url }}" kind="subtitles" srclang="en" label="English">
            {% endif %}
        </video>
        {% endif %}
        
        <p><a href="/frames/{{ video_id }}">Extract Frames</a></p>
    {% else %}
        <p style="color: red;">{{ message }}</p>
    {% endif %}
    
    <p><a href="/">Download Another Video</a></p>
</body>
</html>
```

### templates/frames.html
```html
<!DOCTYPE html>
<html>
<head>
    <title>Frame Extraction</title>
</head>
<body>
    <h1>Frame Extraction - {{ video_id }}</h1>
    
    {% if not ran %}
    <form method="post">
        <button type="submit">Extract Frames (1 per second)</button>
    </form>
    {% else %}
        {% if success %}
            <p style="color: green;">{{ message }}</p>
            <p>Extracted {{ count }} frames</p>
            <p><a href="/questions/{{ video_id }}">Generate Questions</a></p>
        {% else %}
            <p style="color: red;">{{ message }}</p>
        {% endif %}
    {% endif %}
    
    <p><a href="/">Back to Home</a></p>
</body>
</html>
```

### templates/questions.html
```html
<!DOCTYPE html>
<html>
<head>
    <title>Question Generation</title>
</head>
<body>
    <h1>Generate Questions - {{ video_id }}</h1>
    
    {% if duration_seconds %}
    <p>Video Duration: {{ duration_seconds }} seconds</p>
    {% endif %}
    
    <form method="post" id="questionForm">
        <div>
            <label for="start_seconds">Start Time (seconds):</label>
            <input type="number" id="start_seconds" name="start_seconds" value="{{ start_seconds or 0 }}" min="0">
        </div>
        
        <div>
            <label for="interval_seconds">Interval Length (seconds):</label>
            <input type="number" id="interval_seconds" name="interval_seconds" value="{{ interval_seconds or 60 }}" min="1" required>
        </div>
        
        <div>
            <label for="full_duration">
                <input type="checkbox" id="full_duration" name="full_duration" {% if full_duration %}checked{% endif %}>
                Generate for entire video duration
            </label>
        </div>
        
        <div>
            <label for="api_key">OpenAI API Key (optional if set as environment variable):</label>
            <input type="password" id="api_key" name="api_key" style="width: 400px;">
        </div>
        
        <button type="submit">Generate Questions</button>
        <button type="button" onclick="startWebSocket()">Stream Results</button>
    </form>
    
    <div id="progress" style="margin-top: 20px;"></div>
    
    {% if error %}
    <div style="color: red; margin-top: 20px;">
        <h3>Error:</h3>
        <p>{{ error }}</p>
    </div>
    {% endif %}
    
    {% if result %}
    <div style="margin-top: 20px;">
        <h3>Generated Questions:</h3>
        <pre>{{ result }}</pre>
    </div>
    {% endif %}
    
    <p><a href="/">Back to Home</a></p>
    
    <script>
        function startWebSocket() {
            const form = document.getElementById('questionForm');
            const formData = new FormData(form);
            const progressDiv = document.getElementById('progress');
            
            const ws = new WebSocket(`ws://localhost:8000/ws/questions/{{ video_id }}`);
            
            ws.onopen = function() {
                progressDiv.innerHTML = '<p>Connected to server...</p>';
                ws.send(JSON.stringify({
                    start_seconds: parseInt(formData.get('start_seconds') || '0'),
                    interval_seconds: parseInt(formData.get('interval_seconds')),
                    full_duration: formData.get('full_duration') === 'on',
                    api_key: formData.get('api_key') || null
                }));
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'status') {
                    progressDiv.innerHTML += `<p>${data.message}</p>`;
                } else if (data.type === 'segment_result') {
                    progressDiv.innerHTML += `<div><strong>Segment ${data.start}-${data.end}s:</strong><pre>${JSON.stringify(data.result, null, 2)}</pre></div>`;
                } else if (data.type === 'done') {
                    progressDiv.innerHTML += '<p><strong>Generation complete!</strong></p>';
                } else if (data.type === 'error') {
                    progressDiv.innerHTML += `<p style="color: red;">Error: ${data.message}</p>`;
                }
            };
            
            ws.onclose = function() {
                progressDiv.innerHTML += '<p>Connection closed.</p>';
            };
        }
    </script>
</body>
</html>
```

## Usage

### 1. Start the Application

```bash
# Make sure your virtual environment is activated
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Access the Web Interface

Open your browser and go to: `http://localhost:8000`

### 3. Download a Video

1. Enter a YouTube URL in the form
2. Click "Download" to download the video and subtitles
3. The video will be saved in the `downloads/` directory

### 4. Extract Frames

1. After downloading, click "Extract Frames"
2. This will extract one frame per second from the video
3. Frames are saved as JPEG files with metadata in CSV/JSON format

### 5. Generate Questions

1. After extracting frames, click "Generate Questions"
2. Configure the time range and interval
3. Optionally enter your OpenAI API key if not set as environment variable
4. Choose between single interval or full duration processing
5. Use "Stream Results" for real-time progress updates

## Project Structure

```
your-project/
├── main.py                 # Main FastAPI application
├── templates/              # HTML templates
│   ├── download.html
│   ├── preview.html
│   ├── frames.html
│   └── questions.html
├── downloads/              # Downloaded content (auto-created)
│   └── {video_id}/
│       ├── video.mp4
│       ├── subtitles.vtt
│       ├── extracted_frames/
│       │   ├── frame_0000s.jpg
│       │   ├── frame_data.csv
│       │   └── frame_data.json
│       └── questions/
│           └── questions_interval_60s.json
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## API Endpoints

- `GET /` - Main download page
- `POST /download` - Download YouTube video
- `GET /frames/{video_id}` - Frame extraction page
- `POST /frames/{video_id}` - Extract frames
- `GET /questions/{video_id}` - Question generation page
- `POST /questions/{video_id}` - Generate questions (HTTP)
- `WebSocket /ws/questions/{video_id}` - Generate questions (streaming)

## Troubleshooting

### Common Issues

1. **"No module named 'cv2'"**
   ```bash
   pip install opencv-python
   ```

2. **"FFmpeg not found"**
   - yt-dlp usually handles this automatically
   - On Windows: Download FFmpeg and add to PATH
   - On macOS: `brew install ffmpeg`
   - On Ubuntu: `sudo apt install ffmpeg`

3. **OpenAI API errors**
   - Verify your API key is correct
   - Check your OpenAI account has sufficient credits
   - Ensure you have access to GPT-4 Vision API

4. **WebSocket connection issues**
   - Check firewall settings
   - Ensure the server is running on the correct port
   - Try using HTTP endpoints instead

### Performance Tips

- For long videos, use smaller intervals (30-60 seconds) to avoid API timeouts
- The application resizes images to 512x512 for efficiency
- Frame extraction can take several minutes for long videos

## License

This project is for educational purposes. Please respect YouTube's Terms of Service and copyright laws when downloading content.

## Support

For issues related to:
- **yt-dlp**: Check [yt-dlp documentation](https://github.com/yt-dlp/yt-dlp)
- **OpenAI API**: Check [OpenAI documentation](https://platform.openai.com/docs)
- **FastAPI**: Check [FastAPI documentation](https://fastapi.tiangolo.com)