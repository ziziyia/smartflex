# ComfyUI Card Generator - Streamlit Version

A Streamlit web application for generating custom credit card designs using ComfyUI workflows.

## Features

- **Step 1**: Generate custom card chips with different colors and shapes
- **Step 2**: Create card background designs using ControlNet
- **Step 3**: Combine elements and export final card design

## Prerequisites

1. **ComfyUI Server**: You need a running ComfyUI instance with the following models:
   - `realvisxlV50_v50LightningBakedvae.safetensors`
   - `realvis_GPTsettings.safetensors`
   - `SDXL/t2i-adapter-canny-sdxl-1.0.fp16.safetensors`

2. **ComfyUI Extensions**: Install these custom nodes in ComfyUI:
   - AIO Preprocessor
   - InspyrenetRembg (for Step 3)
   - Image Transpose (for Step 3)
   - CR Color Bars (for Step 3)

## Local Development Setup

1. **Clone/Download the files**:
   ```bash
   # Create project directory
   mkdir comfyui-card-generator
   cd comfyui-card-generator
   
   # Save the app.py and requirements.txt files here
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure ComfyUI connection**:
   - Edit `app.py` and update `COMFYUI_SERVER_URL` if your ComfyUI runs on a different address
   - Default is `http://localhost:8188`

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Access the application**:
   - Open your browser and go to `http://localhost:8501`
   - The app will automatically open if running locally

## Production Deployment Options

### Option 1: Streamlit Cloud (Free)

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Set main file as `app.py`
   - Add secrets for ComfyUI server URL if needed

**Note**: Streamlit Cloud has limited resources and may not work well with ComfyUI's computational requirements.

### Option 2: Docker Deployment

1. **Create Dockerfile**:
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY app.py .

   EXPOSE 8501

   HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

   ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and run**:
   ```bash
   docker build -t comfyui-card-generator .
   docker run -p 8501:8501 comfyui-card-generator
   ```

### Option 3: Cloud VPS (Recommended)

1. **Set up Ubuntu server** (DigitalOcean, AWS EC2, etc.)

2. **Install dependencies**:
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip nginx
   pip3 install -r requirements.txt
   ```

3. **Configure systemd service** (`/etc/systemd/system/cardgen.service`):
   ```ini
   [Unit]
   Description=ComfyUI Card Generator
   After=network.target

   [Service]
   Type=simple
   User=ubuntu
   WorkingDirectory=/home/ubuntu/comfyui-card-generator
   ExecStart=/usr/local/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

4. **Configure Nginx reverse proxy** (`/etc/nginx/sites-available/cardgen`):
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://127.0.0.1:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

5. **Enable and start services**:
   ```bash
   sudo systemctl enable cardgen
   sudo systemctl start cardgen
   sudo systemctl enable nginx
   sudo systemctl start nginx
   ```

## Environment Configuration

### Environment Variables

Create a `.streamlit/secrets.toml` file for sensitive configuration:

```toml
[general]
COMFYUI_SERVER_URL = "http://your-comfyui-server:8188"

[comfyui]
api_key = "your-api-key-if-needed"
timeout = 120
```

### Production Configuration

For production, update these settings in `app.py`:

```python
# Production settings
COMFYUI_SERVER_URL = os.getenv("COMFYUI_SERVER_URL", "http://localhost:8188")
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
TIMEOUT_SECONDS = 180
```

## Scaling Considerations

### Multiple ComfyUI Instances

For high traffic, set up load balancing:

```python
COMFYUI_SERVERS = [
    "http://comfyui-1:8188",
    "http://comfyui-2:8188",
    "http://comfyui-3:8188"
]

# Add round-robin server selection in ComfyUIClient
```

### Redis for Session Management

For multi-instance deployments:

```bash
pip install streamlit-redis
```

```python
import streamlit_redis as redis_client

# Store generated images in Redis instead of session state
redis_client.set(f"chip_{user_id}", image_url)
```

## Monitoring and Logging

### Add logging to app.py:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### Health checks:

```python
def health_check():
    try:
        response = requests.get(f"{COMFYUI_SERVER_URL}/system_stats", timeout=5)
        return response.status_code == 200
    except:
        return False
```

## Security Considerations

1. **API Authentication**: Add authentication if deploying publicly
2. **Rate Limiting**: Implement request rate limiting
3. **Input Validation**: Sanitize all user inputs
4. **HTTPS**: Use SSL certificates in production
5. **Firewall**: Restrict access to ComfyUI server

## Troubleshooting

### Common Issues:

1. **ComfyUI Connection Failed**:
   - Check if ComfyUI server is running
   - Verify the server URL and port
   - Check firewall settings

2. **Generation Timeout**:
   - Increase timeout values
   - Check ComfyUI server resources
   - Verify models are loaded

3. **Memory Issues**:
   - Monitor server memory usage
   - Implement image cleanup
   - Use image compression

4. **WebSocket Errors**:
   - Check proxy configuration
   - Verify WebSocket support
   - Update websocket-client version

### Logs Location:

- Streamlit logs: Check terminal output
- ComfyUI logs: Check ComfyUI console
- System logs: `/var/log/syslog`

## Performance Optimization

1. **Image Caching**: Cache generated images
2. **CDN**: Use CDN for static assets
3. **Compression**: Implement image compression
4. **Async Processing**: Use background tasks for long operations

## Support

For issues and questions:
- Check ComfyUI documentation
- Review Streamlit documentation
- Monitor server logs for errors

## License

[Add your license information here]
