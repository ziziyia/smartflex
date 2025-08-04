# ComfyUI Card Generator - Streamlit Version

A modern Streamlit web application for generating custom credit card designs using ComfyUI workflows with AI-powered chip and card creation.

## 🌟 Features

- **Step 1**: Generate custom card chips with different colors, shapes, and ControlNet support
- **Step 2**: Create card background designs using ControlNet with Gemini AI prompt enhancement
- **Step 3**: Combine elements and export final card design
- **AI Enhancement**: Gemini API integration for professional prompt optimization
- **Advanced Options**: ControlNet support for custom shapes and designs
- **Progress Tracking**: Visual step indicator and progress monitoring
- **Error Handling**: Comprehensive error handling and recovery

## 🛠️ Prerequisites

### ComfyUI Server Setup

1. **ComfyUI Server**: You need a running ComfyUI instance with the following models:
   - `realvisxlV50_v50LightningBakedvae.safetensors`
   - `realvis_GPTsettings.safetensors` (LoRA)
   - `SDXL/t2i-adapter-canny-sdxl-1.0.fp16.safetensors` (ControlNet)

2. **ComfyUI Extensions**: Install these custom nodes in ComfyUI:
   - **comfyui_manager** (general) https://github.com/Comfy-Org/ComfyUI-Manager
   - **comfyui_controlnet_aux** (for Controlnet) https://github.com/Fannovel16/comfyui_controlnet_aux
   - **comfyui_inspyrenet_rembg** (for background removal in Step 3) https://github.com/john-mnz/ComfyUI-Inspyrenet-Rembg
   - **was_node_suite_comfyui** (for image positioning in Step 3) https://github.com/ltdrdata/was-node-suite-comfyui
   - **comfyui_comfyroll_customnodes** (for base canvas creation in Step 3) https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes
   - **comfyui-hiforce-plugin** (for color conversion) https://github.com/hiforce/comfyui-hiforce-plugin

### API Keys

1. **Gemini API Key** (optional but recommended):
   - Get your key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Used for intelligent prompt enhancement

## 🚀 Local Development Setup

### Configuration Options

#### Environment Variables

Set these in your environment or modify directly in `app.py`:

```python
COMFYUI_SERVER_URL = "http://your-comfyui-server:8188"
GEMINI_API_KEY = "your-gemini-api-key"
```

#### Streamlit Secrets (Recommended for production)

Create `.streamlit/secrets.toml`:

```toml
[general]
COMFYUI_SERVER_URL = "http://your-comfyui-server:8188"
GEMINI_API_KEY = "your-gemini-api-key"

[timeouts]
request_timeout = 30
generation_timeout = 180
```

## Deployment Options

### Option 1: Docker Deployment (Recommended)

1. **Create Dockerfile**:
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       curl \
       && rm -rf /var/lib/apt/lists/*

   # Copy requirements and install Python dependencies
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   # Copy application files
   COPY app.py .
   COPY default_card_outline.png .

   # Create directories for logs and temp files
   RUN mkdir -p /app/logs /app/temp

   EXPOSE 8501

   # Health check
   HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
       CMD curl -f http://localhost:8501/_stcore/health || exit 1

   # Run the application
   ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and run**:
   ```bash
   docker build -t comfyui-card-generator .
   docker run -p 8501:8501 \
     -e COMFYUI_SERVER_URL=http://your-comfyui:8188 \
     -e GEMINI_API_KEY=your-key \
     comfyui-card-generator
   ```

### Option 2: Docker Compose (Full Stack)

Use the provided `docker-compose.yml` for a complete setup including ComfyUI.

## 🔧 Configuration Guide

### Required Files

1. **default_card_outline.png**: Default Controlnet File
2. **app.py**: Main application file
3. **requirements.txt**: Python dependencies

## 🔍 Troubleshooting

### Common Issues & Solutions

#### 1. ComfyUI Connection Failed
**Symptoms**: "ComfyUI Server Offline" or connection refused errors

**Solutions**:
- Verify ComfyUI is running: `curl http://localhost:8188/system_stats`
- Check firewall settings
- Ensure correct server URL in configuration
- Check ComfyUI logs for errors

#### 2. Generation Timeout
**Symptoms**: "Generation timed out after 3 minutes"

**Solutions**:
- Check ComfyUI server resources (GPU memory, CPU)
- Verify all required models are loaded
- Increase timeout in code if needed
- Check ComfyUI queue status

#### 3. Missing Models/Extensions
**Symptoms**: Workflow errors or "model not found"

**Required Models**:
```bash
# Download to ComfyUI/models/checkpoints/
realvisxlV50_v50LightningBakedvae.safetensors

# Download to ComfyUI/models/loras/
realvis_GPTsettings.safetensors

# Download to ComfyUI/models/controlnet/
SDXL/t2i-adapter-canny-sdxl-1.0.fp16.safetensors
```

**Required Extensions**:
- Install via ComfyUI Manager or manually clone to `ComfyUI/custom_nodes/`

#### 4. Gemini API Issues
**Symptoms**: "Gemini API not available" or authentication errors

**Solutions**:
- Verify API key is correct
- Check API quota and billing
- Ensure internet connectivity
- Try different Gemini models

#### 5. Upload/Download Issues
**Symptoms**: Failed to upload images or download results

**Solutions**:
- Check file permissions
- Verify network connectivity
- Ensure sufficient disk space
- Check file size limits

### Debug Mode

Enable debug logging by adding to `app.py`:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Server Requirements

**Minimum Requirements**:
- 4GB RAM
- 2 CPU cores
- 10GB storage
- GPU with 8GB VRAM (for ComfyUI)

**Recommended Requirements**:
- 8GB+ RAM
- 4+ CPU cores  
- 50GB+ storage
- GPU with 12GB+ VRAM
