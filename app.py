import streamlit as st
import requests
import json
import time
import io
import base64
import os
from PIL import Image
import websocket
import threading
import uuid
from typing import Optional, Dict, Any
import google.generativeai as genai

# Page configuration
st.set_page_config(
    page_title="ComfyUI Card Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .step-indicator {
        display: flex;
        justify-content: center;
        margin: 2rem 0;
    }
    
    .step {
        display: flex;
        align-items: center;
        margin: 0 1rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    
    .step.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .step.completed {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    .step.inactive {
        background: #f0f0f0;
        color: #666;
    }
    
    .preview-container {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration - Use environment variables with fallbacks
COMFYUI_SERVER_URL = os.getenv("COMFYUI_SERVER_URL", "http://10.3.250.181:8891")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBgTf8G5pJDocColgdFZjNA9XKXLH9EodY")

# Initialize Gemini
gemini_model = None
try:
    if GEMINI_API_KEY and GEMINI_API_KEY != "your-gemini-api-key-here":
        genai.configure(api_key=GEMINI_API_KEY)
        # Try different model names in order of preference
        model_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-1.0-pro']
        
        for model_name in model_names:
            try:
                gemini_model = genai.GenerativeModel(model_name)
                st.success(f"✅ Gemini initialized with model: {model_name}")
                break
            except Exception as model_error:
                st.warning(f"Model {model_name} not available: {str(model_error)}")
                continue
        
        if gemini_model is None:
            st.error("❌ No available Gemini models found")
    else:
        st.warning("⚠️ Gemini API key not configured. Prompt enhancement will be disabled.")
        
except Exception as e:
    st.error(f"Failed to initialize Gemini API: {str(e)}")
    gemini_model = None

class ComfyUIClient:
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')  # Remove trailing slash
        self.client_id = str(uuid.uuid4())
    
    def generate_client_id(self) -> str:
        return str(uuid.uuid4())
    
    def queue_prompt(self, workflow: Dict[str, Any]) -> str:
        """Queue a prompt and return the prompt_id"""
        try:
            response = requests.post(
                f"{self.server_url}/prompt",
                json={
                    "prompt": workflow,
                    "client_id": self.client_id
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result["prompt_id"]
        except requests.exceptions.Timeout:
            st.error("Request timed out. Please check your ComfyUI server.")
            raise
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to ComfyUI server. Please check if it's running.")
            raise
        except Exception as e:
            st.error(f"Failed to queue prompt: {str(e)}")
            raise
    
    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Get the history for a specific prompt"""
        try:
            response = requests.get(f"{self.server_url}/history/{prompt_id}", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to get history: {str(e)}")
            raise
    
    def get_image_url(self, filename: str, subfolder: str = "", type: str = "output") -> str:
        """Generate image URL for viewing"""
        return f"{self.server_url}/view?filename={filename}&subfolder={subfolder}&type={type}"
    
    def upload_image(self, image_data: bytes, filename: str) -> str:
        """Upload image to ComfyUI server"""
        try:
            files = {"image": (filename, image_data, "image/png")}
            data = {"type": "input", "subfolder": ""}
            
            response = requests.post(f"{self.server_url}/upload/image", files=files, data=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result.get("name", filename)
        except Exception as e:
            st.error(f"Failed to upload image: {str(e)}")
            raise

def enhance_prompt_with_gemini(user_prompt: str) -> Optional[str]:
    """Use Gemini API to enhance user's prompt for better card design generation"""
    if not gemini_model:
        st.error("Gemini API not available")
        return None
    
    try:
        enhancement_instruction = f"""
        You are an expert AI prompt engineer specializing in credit card design generation for RealVisXL v5.0 Lightning model.
        The user wants to create a credit card design and has provided this basic prompt: "{user_prompt}"

        Please enhance this prompt specifically optimized for RealVisXL v5.0 Lightning model to create a premium, professional credit card design.

        CRITICAL REQUIREMENTS for RealVisXL v5.0 Lightning:
        - Start with clear subject identification: "premium credit card design"
        - Use specific lighting terms: "studio lighting", "professional photography", "soft shadows"
        - Include material specifications: "matte finish", "metallic accents", "glossy surface", "brushed metal"
        - Add quality boosters: "8k resolution", "ultra detailed", "sharp focus", "photorealistic"
        - Specify camera/lens terms: "macro lens", "product photography", "clean background"

        Enhancement focus areas:
        - Visual aesthetics: specific colors (hex codes if relevant), gradient directions, texture details
        - Professional design elements: embossed text, holographic strips, magnetic stripe placement
        - Modern credit card features: EMV chip positioning, contactless payment symbol, bank logos
        - Luxury materials: carbon fiber, titanium, gold plating, diamond texture
        - Technical photography details: depth of field, reflection control, edge lighting

        Format the enhanced prompt as:
        "premium credit card design, [USER_CONCEPT], [MATERIAL_DETAILS], [LIGHTING_SETUP], [QUALITY_TERMS], [PHOTOGRAPHY_STYLE]"

        Avoid these terms that work poorly with RealVisXL: "cartoon", "anime", "sketch", "painting", "artistic", "abstract"

        Keep the core concept from "{user_prompt}" but make it much more detailed and technically specific for optimal RealVisXL v5.0 Lightning generation.

        Return ONLY the enhanced prompt, no explanations or additional text.
        """

        response = gemini_model.generate_content(enhancement_instruction)
        enhanced_prompt = response.text.strip()
        
        return enhanced_prompt
        
    except Exception as e:
        st.error(f"Failed to enhance prompt with Gemini: {str(e)}")
        return None

def initialize_session_state():
    """Initialize session state variables"""
    if "current_step" not in st.session_state:
        st.session_state.current_step = 1
    if "generated_chip" not in st.session_state:
        st.session_state.generated_chip = None
    if "generated_card" not in st.session_state:
        st.session_state.generated_card = None
    if "final_result" not in st.session_state:
        st.session_state.final_result = None
    if "comfyui_client" not in st.session_state:
        st.session_state.comfyui_client = ComfyUIClient(COMFYUI_SERVER_URL)
    if "enhanced_prompt" not in st.session_state:
        st.session_state.enhanced_prompt = None
    if "original_prompt" not in st.session_state:
        st.session_state.original_prompt = None

def render_header():
    """Render the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🎨 ComfyUI Card Generator</h1>
        <p>Generate custom credit card designs with AI-powered chip and card creation</p>
    </div>
    """, unsafe_allow_html=True)

def render_step_indicator():
    """Render step progress indicator"""
    steps = [
        {"num": 1, "name": "🔧 Chip Generation", "status": "active" if st.session_state.current_step == 1 else ("completed" if st.session_state.generated_chip else "inactive")},
        {"num": 2, "name": "🎨 Card Design", "status": "active" if st.session_state.current_step == 2 else ("completed" if st.session_state.generated_card else "inactive")},
        {"num": 3, "name": "🔄 Combine & Export", "status": "active" if st.session_state.current_step == 3 else "inactive"}
    ]
    
    step_html = '<div class="step-indicator">'
    for step in steps:
        step_html += f'<div class="step {step["status"]}">{step["name"]}</div>'
    step_html += '</div>'
    
    st.markdown(step_html, unsafe_allow_html=True)

def wait_for_completion(prompt_id: str, progress_bar, status_text, use_controlnet: bool = False) -> Optional[str]:
    """Wait for ComfyUI to complete generation and return image URL"""
    max_wait_time = 180  # 3 minutes timeout (increased for stability)
    check_interval = 3   # Check every 3 seconds (reduced frequency)
    elapsed_time = 0
    
    while elapsed_time < max_wait_time:
        try:
            history = st.session_state.comfyui_client.get_history(prompt_id)
            
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                
                # For ControlNet chip generation (node 32)
                if use_controlnet and "32" in outputs and outputs["32"].get("images"):
                    image_info = outputs["32"]["images"][0]
                    return st.session_state.comfyui_client.get_image_url(
                        image_info["filename"], 
                        image_info.get("subfolder", ""), 
                        image_info.get("type", "output")
                    )
                
                # For regular chip generation (node 31)
                if "31" in outputs and outputs["31"].get("images"):
                    image_info = outputs["31"]["images"][0]
                    return st.session_state.comfyui_client.get_image_url(
                        image_info["filename"], 
                        image_info.get("subfolder", ""), 
                        image_info.get("type", "output")
                    )
                
                # For card design (node 13)
                if "13" in outputs and outputs["13"].get("images"):
                    image_info = outputs["13"]["images"][0]
                    return st.session_state.comfyui_client.get_image_url(
                        image_info["filename"], 
                        image_info.get("subfolder", ""), 
                        image_info.get("type", "output")
                    )
                
                # For final combination (check multiple possible output nodes)
                for node_id in ["12", "13", "9"]:  # Check multiple nodes for combination workflow
                    if node_id in outputs and outputs[node_id].get("images"):
                        image_info = outputs[node_id]["images"][0]
                        return st.session_state.comfyui_client.get_image_url(
                            image_info["filename"], 
                            image_info.get("subfolder", ""), 
                            image_info.get("type", "output")
                        )
            
            # Update progress
            progress_percentage = min(95, (elapsed_time / max_wait_time) * 100)
            progress_bar.progress(progress_percentage / 100)
            status_text.text(f"Processing... ({elapsed_time}s)")
            
            time.sleep(check_interval)
            elapsed_time += check_interval
            
        except Exception as e:
            st.error(f"Error checking completion: {str(e)}")
            return None
    
    st.error("Generation timed out after 3 minutes. Please try again.")
    return None

def generate_chip(chip_color: str, chip_shape: str, custom_details: str, use_controlnet: bool = False, control_image_name: str = None) -> Optional[str]:
    """Generate chip using ComfyUI with optional ControlNet"""
    # Construct prompt
    base_prompt = f"card chip, {chip_color}, {chip_shape}, white background"
    final_prompt = f"{base_prompt}, {custom_details}" if custom_details else base_prompt
    
    st.info(f"Generated prompt: {final_prompt}")
    st.info(f"Using ControlNet: {use_controlnet}")
    if use_controlnet and control_image_name:
        st.info(f"Control image: {control_image_name}")
    
    # Generate a random seed for each generation
    seed = int(time.time() * 1000000) % 1000000000000000
    
    if use_controlnet and control_image_name:
        # ControlNet-based chip generation workflow
        workflow = {
            "32": {
                "inputs": {"images": ["39", 0]},
                "class_type": "PreviewImage",
                "_meta": {"title": "Preview Image"}
            },
            "33": {
                "inputs": {"ckpt_name": "realvisxlV50_v50LightningBakedvae.safetensors"},
                "class_type": "CheckpointLoaderSimple",
                "_meta": {"title": "Load Checkpoint"}
            },
            "34": {
                "inputs": {
                    "lora_name": "realvis_GPTsettings.safetensors",
                    "strength_model": 1,
                    "strength_clip": 1,
                    "model": ["33", 0],
                    "clip": ["33", 1]
                },
                "class_type": "LoraLoader",
                "_meta": {"title": "Load LoRA"}
            },
            "35": {
                "inputs": {"text": final_prompt, "clip": ["34", 1]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive Prompt"}
            },
            "36": {
                "inputs": {"text": "blurry, lowres, deformed, watermark", "clip": ["34", 1]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Negative Prompt"}
            },
            "37": {
                "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
                "class_type": "EmptyLatentImage",
                "_meta": {"title": "Empty Latent Image"}
            },
            "38": {
                "inputs": {
                    "seed": seed,
                    "steps": 50,
                    "cfg": 5,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1,
                    "model": ["34", 0],
                    "positive": ["42", 0],
                    "negative": ["42", 1],
                    "latent_image": ["37", 0]
                },
                "class_type": "KSampler",
                "_meta": {"title": "KSampler"}
            },
            "39": {
                "inputs": {"samples": ["38", 0], "vae": ["33", 2]},
                "class_type": "VAEDecode",
                "_meta": {"title": "VAE Decode"}
            },
            "40": {
                "inputs": {"control_net_name": "SDXL/t2i-adapter-canny-sdxl-1.0.fp16.safetensors"},
                "class_type": "ControlNetLoader",
                "_meta": {"title": "Load ControlNet Model"}
            },
            "41": {
                "inputs": {
                    "preprocessor": "PyraCannyPreprocessor",
                    "resolution": 512,
                    "image": ["43", 0]
                },
                "class_type": "AIO_Preprocessor",
                "_meta": {"title": "AIO Aux Preprocessor"}
            },
            "42": {
                "inputs": {
                    "strength": 1,
                    "start_percent": 0,
                    "end_percent": 1,
                    "positive": ["35", 0],
                    "negative": ["36", 0],
                    "control_net": ["40", 0],
                    "image": ["41", 0],
                    "vae": ["33", 2]
                },
                "class_type": "ControlNetApplyAdvanced",
                "_meta": {"title": "Apply ControlNet"}
            },
            "43": {
                "inputs": {"image": control_image_name},
                "class_type": "LoadImage",
                "_meta": {"title": "Load Image"}
            },
            "44": {
                "inputs": {"images": ["41", 0]},
                "class_type": "PreviewImage",
                "_meta": {"title": "Preview Image"}
            }
        }
    else:
        # Original chip generation workflow (without ControlNet)
        workflow = {
            "24": {
                "inputs": {"ckpt_name": "realvisxlV50_v50LightningBakedvae.safetensors"},
                "class_type": "CheckpointLoaderSimple",
                "_meta": {"title": "Load Checkpoint"}
            },
            "25": {
                "inputs": {
                    "lora_name": "realvis_GPTsettings.safetensors",
                    "strength_model": 1,
                    "strength_clip": 1,
                    "model": ["24", 0],
                    "clip": ["24", 1]
                },
                "class_type": "LoraLoader",
                "_meta": {"title": "Load LoRA"}
            },
            "26": {
                "inputs": {"text": final_prompt, "clip": ["25", 1]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive Prompt"}
            },
            "27": {
                "inputs": {"text": "blurry, lowres, deformed, watermark", "clip": ["25", 1]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Negative Prompt"}
            },
            "28": {
                "inputs": {
                    "seed": seed,
                    "steps": 50,
                    "cfg": 5,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1,
                    "model": ["25", 0],
                    "positive": ["26", 0],
                    "negative": ["27", 0],
                    "latent_image": ["29", 0]
                },
                "class_type": "KSampler",
                "_meta": {"title": "KSampler"}
            },
            "29": {
                "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
                "class_type": "EmptyLatentImage",
                "_meta": {"title": "Empty Latent Image"}
            },
            "30": {
                "inputs": {"samples": ["28", 0], "vae": ["24", 2]},
                "class_type": "VAEDecode",
                "_meta": {"title": "VAE Decode"}
            },
            "31": {
                "inputs": {"images": ["30", 0]},
                "class_type": "PreviewImage",
                "_meta": {"title": "Preview Image"}
            }
        }
    
    try:
        # Queue the prompt
        st.info("Sending request to ComfyUI...")
        prompt_id = st.session_state.comfyui_client.queue_prompt(workflow)
        st.success(f"Prompt queued successfully! ID: {prompt_id}")
        return prompt_id
    except Exception as e:
        st.error(f"Failed to generate chip: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        return None

def load_default_control_image() -> str:
    """Load and upload the default control image"""
    try:
        # First check if the file exists locally
        default_image_path = "default_card_outline.png"
        
        if os.path.exists(default_image_path):
            # Read and upload the local default image
            with open(default_image_path, 'rb') as f:
                image_bytes = f.read()
            
            # Upload to ComfyUI
            uploaded_name = st.session_state.comfyui_client.upload_image(image_bytes, "default_card_outline.png")
            st.success("✅ Uploaded default control image to ComfyUI")
            
            return uploaded_name
        else:
            st.warning("Default card outline image not found. Creating fallback...")
            raise FileNotFoundError("Default image not found")
        
    except Exception as e:
        st.error(f"Failed to load default control image: {str(e)}")
        # Create a simple card outline as fallback
        try:
            from PIL import Image, ImageDraw
            import io
            
            st.info("Creating simple card outline as fallback...")
            
            # Create a simple card outline
            img = Image.new('RGB', (400, 250), color='white')
            draw = ImageDraw.Draw(img)
            
            # Draw card outline
            draw.rectangle([10, 10, 390, 240], outline='black', width=3)
            draw.rectangle([20, 20, 100, 60], outline='gray', width=2)  # Chip area
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Upload the default image
            control_image_name = st.session_state.comfyui_client.upload_image(img_byte_arr, "fallback_card_outline.png")
            st.success("✅ Created and uploaded fallback card outline")
            
            return control_image_name
            
        except Exception as fallback_error:
            st.error(f"Could not create fallback image: {str(fallback_error)}")
            return "default_card_outline.png"  # Return default name as last resort

def generate_card_design(prompt: str, control_image_name: str = "default_card_outline.png") -> Optional[str]:
    """Generate card design using ComfyUI"""
    # Optimize prompt
    base_enhancements = "high quality, detailed, professional design, clean background, studio lighting, sharp focus, premium card design"
    card_enhancements = "credit card design, elegant, modern style, glossy finish"
    optimized_prompt = f"{prompt}, {base_enhancements}, {card_enhancements}"
    
    st.info(f"Optimized prompt: {optimized_prompt}")
    st.info(f"Using control image: {control_image_name}")
    
    # If using default image, load and upload it
    if control_image_name == "default_card_outline.png":
        control_image_name = load_default_control_image()
    
    # Generate a random seed
    seed = int(time.time() * 1000000) % 1000000000000000
    
    # Card design workflow
    workflow = {
        "1": {
            "inputs": {"control_net_name": "SDXL/t2i-adapter-canny-sdxl-1.0.fp16.safetensors"},
            "class_type": "ControlNetLoader",
            "_meta": {"title": "Load ControlNet Model"}
        },
        "3": {
            "inputs": {"preprocessor": "CannyEdgePreprocessor", "resolution": 512, "image": ["9", 0]},
            "class_type": "AIO_Preprocessor",
            "_meta": {"title": "AIO Aux Preprocessor"}
        },
        "4": {
            "inputs": {
                "strength": 1, "start_percent": 0, "end_percent": 1,
                "positive": ["6", 0], "negative": ["7", 0],
                "control_net": ["1", 0], "image": ["3", 0], "vae": ["5", 2]
            },
            "class_type": "ControlNetApplyAdvanced",
            "_meta": {"title": "Apply ControlNet"}
        },
        "5": {
            "inputs": {"ckpt_name": "realvisxlV50_v50LightningBakedvae.safetensors"},
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Checkpoint"}
        },
        "6": {
            "inputs": {"text": optimized_prompt, "clip": ["5", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Positive Prompt"}
        },
        "7": {
            "inputs": {"text": "autumn, winter, blurry, lowres, deformed, watermark", "clip": ["5", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Negative Prompt"}
        },
        "8": {
            "inputs": {
                "seed": seed,
                "steps": 15, "cfg": 7, "sampler_name": "dpm_adaptive", "scheduler": "normal", "denoise": 1,
                "model": ["5", 0], "positive": ["4", 0], "negative": ["4", 1], "latent_image": ["15", 0]
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"}
        },
        "9": {
            "inputs": {"image": control_image_name},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Image"}
        },
        "12": {
            "inputs": {"samples": ["8", 0], "vae": ["5", 2]},
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"}
        },
        "13": {
            "inputs": {"images": ["12", 0]},
            "class_type": "PreviewImage",
            "_meta": {"title": "Preview Image"}
        },
        "15": {
            "inputs": {"width": 1600, "height": 1024, "batch_size": 1},
            "class_type": "EmptyLatentImage",
            "_meta": {"title": "Empty Latent Image"}
        },
        "17": {
            "inputs": {"images": ["3", 0]},
            "class_type": "PreviewImage",
            "_meta": {"title": "Preview Image"}
        }
    }
    
    try:
        st.info("Sending card design request to ComfyUI...")
        prompt_id = st.session_state.comfyui_client.queue_prompt(workflow)
        st.success(f"Card design prompt queued! ID: {prompt_id}")
        return prompt_id
    except Exception as e:
        st.error(f"Failed to generate card design: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        return None

def combine_elements_workflow() -> Optional[str]:
    """Execute the final combination workflow using ComfyUI"""
    try:
        # First, upload both images to ComfyUI
        st.info("Uploading chip and card images to ComfyUI...")
        
        # Download and upload chip image
        chip_response = requests.get(st.session_state.generated_chip, timeout=30)
        chip_bytes = chip_response.content
        chip_filename = st.session_state.comfyui_client.upload_image(chip_bytes, "generated_chip.png")
        
        # Download and upload card image  
        card_response = requests.get(st.session_state.generated_card, timeout=30)
        card_bytes = card_response.content
        card_filename = st.session_state.comfyui_client.upload_image(card_bytes, "generated_card.png")
        
        st.success(f"Uploaded chip: {chip_filename}, card: {card_filename}")
        
        # Final combination workflow
        workflow = {
            "1": {
                "inputs": {"image": card_filename},
                "class_type": "LoadImage",
                "_meta": {"title": "Load Card Image"}
            },
            "2": {
                "inputs": {"image": chip_filename},
                "class_type": "LoadImage",
                "_meta": {"title": "Load Chip Image"}
            },
            "3": {
                "inputs": {
                    "width": 200,
                    "height": 200,
                    "X": 175,
                    "Y": 345,
                    "rotation": 0,
                    "feathering": 0,
                    "image": ["6", 0],
                    "image_overlay": ["2", 0]
                },
                "class_type": "Image Transpose",
                "_meta": {"title": "Position Chip on Card"}
            },
            "4": {
                "inputs": {"images": ["3", 0]},
                "class_type": "HfImageToRGB",
                "_meta": {"title": "Convert Image to RGB"}
            },
            "6": {
                "inputs": {
                    "mode": "2-color",
                    "width": 1600,
                    "height": 1024,
                    "color_1": "white",
                    "color_2": "white",
                    "orientation": "vertical",
                    "bar_frequency": 9,
                    "offset": 0,
                    "color1_hex": "#000000",
                    "color2_hex": "#000000"
                },
                "class_type": "CR Color Bars",
                "_meta": {"title": "Create Base Canvas"}
            },
            "9": {
                "inputs": {"images": ["4", 0]},
                "class_type": "PreviewImage",
                "_meta": {"title": "Preview Combined Result"}
            },
            "12": {
                "inputs": {
                    "width": 1600,
                    "height": 1024,
                    "X": 0,
                    "Y": 0,
                    "rotation": 0,
                    "feathering": 0,
                    "image": ["1", 0],
                    "image_overlay": ["17", 0]
                },
                "class_type": "Image Transpose",
                "_meta": {"title": "Overlay Processed Chip"}
            },
            "13": {
                "inputs": {"images": ["12", 0]},
                "class_type": "PreviewImage",
                "_meta": {"title": "Final Preview"}
            },
            "17": {
                "inputs": {
                    "torchscript_jit": "default",
                    "image": ["4", 0]
                },
                "class_type": "InspyrenetRembg",
                "_meta": {"title": "Remove Chip Background"}
            },
            "19": {
                "inputs": {"images": ["17", 0]},
                "class_type": "PreviewImage",
                "_meta": {"title": "Chip with Background Removed"}
            }
        }
        
        st.info("Executing combination workflow...")
        prompt_id = st.session_state.comfyui_client.queue_prompt(workflow)
        st.success(f"Combination workflow queued! ID: {prompt_id}")
        return prompt_id
        
    except Exception as e:
        st.error(f"Failed to execute combination workflow: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        return None

def step_1_chip_generation():
    """Step 1: Chip Generation Interface"""
    st.header("🔧 Step 1: Generate Card Chip")
    st.write("Create a realistic credit card chip using AI generation. Choose from different chip styles and customize the appearance.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Configuration")
        
        chip_color = st.selectbox("Chip Color", ["gold", "silver"], index=0)
        chip_shape = st.text_input("Chip Shape", value="classic", help="e.g., heart, star, classic, whale")
        custom_details = st.text_area("Custom Details (Optional)", help="Add custom details for chip generation...")
        
        # Advanced ControlNet Option
        st.markdown("---")
        st.subheader("🔧 Advanced Options")
        
        use_controlnet = st.checkbox(
            "Use Custom Shape (ControlNet)", 
            help="Upload a shape outline to control the chip's form"
        )
        
        control_image_name = None
        if use_controlnet:
            st.info("📝 Upload a black outline image on white background (like the cat example)")
            control_image = st.file_uploader(
                "Upload Shape Outline", 
                type=["png", "jpg", "jpeg"],
                help="Upload a black outline image that will define the chip's shape"
            )
            
            if control_image:
                try:
                    # Show preview of uploaded image
                    st.image(control_image, caption="Shape Control Image", use_container_width=True, width=200)
                    
                    # Upload to ComfyUI
                    image_bytes = control_image.read()
                    control_image_name = st.session_state.comfyui_client.upload_image(image_bytes, control_image.name)
                    st.success(f"✅ Uploaded: {control_image.name}")
                except Exception as e:
                    st.error(f"Failed to upload control image: {str(e)}")
                    use_controlnet = False
            else:
                st.warning("⚠️ Please upload a shape outline image to use ControlNet")
                use_controlnet = False
        
        # Generate button - disabled if ControlNet is selected but no image uploaded
        can_generate = not use_controlnet or (use_controlnet and control_image_name is not None)
        generate_btn = st.button(
            "🎨 Generate Chip", 
            type="primary", 
            use_container_width=True,
            disabled=not can_generate
        )
        
        if use_controlnet and not control_image_name:
            st.caption("⚠️ Upload a control image to enable generation")
    
    with col2:
        st.subheader("Preview")
        
        if st.session_state.generated_chip:
            st.image(st.session_state.generated_chip, caption="Generated Chip", use_container_width=True)
        else:
            st.markdown("""
            <div class="preview-container">
                <h3>🔧</h3>
                <p>Click "Generate Chip" to create</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show example of what ControlNet can do
        if use_controlnet:
            st.markdown("---")
            st.subheader("ControlNet Example")
            st.caption("With ControlNet, you can create chips in custom shapes like animals, symbols, or any outline you provide.")
    
    if generate_btn and can_generate:
        with st.spinner("Generating chip..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            prompt_id = generate_chip(chip_color, chip_shape, custom_details, use_controlnet, control_image_name)
            
            if prompt_id:
                status_text.text("Waiting for generation to complete...")
                progress_bar.progress(0.1)
                
                image_url = wait_for_completion(prompt_id, progress_bar, status_text, use_controlnet)
                
                if image_url:
                    st.session_state.generated_chip = image_url
                    progress_bar.progress(1.0)
                    status_text.text("Chip generated successfully!")
                    st.rerun()
                else:
                    st.error("Failed to generate chip. Please try again.")
            else:
                st.error("Failed to start chip generation.")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        if st.session_state.generated_chip:
            if st.button("Next: Card Design →", type="primary"):
                st.session_state.current_step = 2
                st.rerun()

def step_2_card_design():
    """Step 2: Card Design Interface"""
    st.header("🎨 Step 2: Design Card Background")
    st.write("Create the main card design using AI-powered generation. Describe your vision and customize the ControlNet reference image.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Configuration")
        
        design_prompt = st.text_area(
            "Design Prompt", 
            help="Describe your ideal card design...",
            placeholder="e.g., 'A luxury credit card with golden accents and geometric patterns'",
            key="user_design_prompt"
        )
        
        # Advanced Gemini Enhancement Option
        st.markdown("---")
        st.subheader("🤖 AI Prompt Enhancement")
        
        use_gemini = st.checkbox(
            "Enhance prompt with Gemini AI", 
            help="Let Gemini AI improve your prompt for better results"
        )
        
        enhanced_prompt_text = ""
        final_prompt_to_use = design_prompt.strip()
        
        if use_gemini and design_prompt.strip():
            if st.button("✨ Enhance Prompt", use_container_width=True):
                if gemini_model:
                    with st.spinner("Enhancing prompt with Gemini AI..."):
                        enhanced = enhance_prompt_with_gemini(design_prompt.strip())
                        if enhanced:
                            st.session_state.enhanced_prompt = enhanced
                            st.session_state.original_prompt = design_prompt.strip()
                            st.success("✅ Prompt enhanced successfully!")
                        else:
                            st.error("Failed to enhance prompt")
                else:
                    st.error("Gemini API not configured")
            
            # Show enhanced prompt if available
            if hasattr(st.session_state, 'enhanced_prompt') and st.session_state.enhanced_prompt and hasattr(st.session_state, 'original_prompt') and st.session_state.original_prompt == design_prompt.strip():
                st.subheader("Enhanced Prompt")
                enhanced_prompt_text = st.text_area(
                    "Gemini Enhanced Version",
                    value=st.session_state.enhanced_prompt,
                    height=150,
                    help="You can edit this enhanced version if needed"
                )
                
                # Let user choose which prompt to use
                prompt_choice = st.radio(
                    "Which prompt would you like to use?",
                    ["Original Prompt", "Enhanced Prompt"],
                    index=1,  # Default to enhanced
                    horizontal=True
                )
                
                if prompt_choice == "Enhanced Prompt":
                    final_prompt_to_use = enhanced_prompt_text
                else:
                    final_prompt_to_use = design_prompt.strip()
                
                # Show comparison
                with st.expander("📊 Prompt Comparison", expanded=False):
                    col_orig, col_enh = st.columns(2)
                    with col_orig:
                        st.markdown("**Original:**")
                        st.text(design_prompt.strip())
                    with col_enh:
                        st.markdown("**Enhanced:**")
                        st.text(enhanced_prompt_text)
        
        elif use_gemini and not design_prompt.strip():
            st.info("💡 Enter a prompt above to use Gemini enhancement")
        
        st.markdown("---")
        st.subheader("ControlNet Reference Image")
        
        # Show current control image info
        st.info("Using default card outline (you can upload a custom image below)")
        
        control_image = st.file_uploader(
            "Upload Custom Control Image (Optional)", 
            type=["png", "jpg", "jpeg"],
            help="Upload a custom control image or leave empty to use default card outline"
        )
        
        control_image_name = "default_card_outline.png"  # Default filename
        
        if control_image:
            # Upload custom control image
            try:
                image_bytes = control_image.read()
                control_image_name = st.session_state.comfyui_client.upload_image(image_bytes, control_image.name)
                st.image(control_image, caption="Custom Control Image", use_container_width=True)
                st.success(f"Uploaded: {control_image.name}")
            except Exception as e:
                st.error(f"Failed to upload control image: {str(e)}")
                control_image_name = "default_card_outline.png"  # Fallback to default
        else:
            # Display info about default image
            st.caption("Default card outline will be used for ControlNet guidance")
        
        # Show final prompt that will be used
        if final_prompt_to_use:
            st.markdown("---")
            st.subheader("🎯 Final Prompt")
            st.info(f"**Will use:** {final_prompt_to_use[:100]}{'...' if len(final_prompt_to_use) > 100 else ''}")
        
        generate_btn = st.button("🎨 Generate Design", type="primary", use_container_width=True, disabled=not final_prompt_to_use)
    
    with col2:
        st.subheader("Preview")
        
        if st.session_state.generated_card:
            st.image(st.session_state.generated_card, caption="Generated Card Design", use_container_width=True)
        else:
            st.markdown("""
            <div class="preview-container">
                <h3>🎨</h3>
                <p>Enter prompt and generate design</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show Gemini enhancement info
        if use_gemini:
            st.markdown("---")
            st.subheader("🤖 AI Enhancement")
            if gemini_model:
                st.success("✅ Gemini AI Ready")
                st.caption("Gemini will analyze your prompt and add professional design details, color schemes, and technical specifications for better AI generation results.")
            else:
                st.error("❌ Gemini API Not Configured")
                st.caption("Please configure your Gemini API key in the code.")
    
    if generate_btn and final_prompt_to_use:
        with st.spinner("Generating card design..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            prompt_id = generate_card_design(final_prompt_to_use, control_image_name)
            
            if prompt_id:
                status_text.text("Processing design generation...")
                progress_bar.progress(0.1)
                
                image_url = wait_for_completion(prompt_id, progress_bar, status_text)
                
                if image_url:
                    st.session_state.generated_card = image_url
                    progress_bar.progress(1.0)
                    status_text.text("Card design generated successfully!")
                    st.rerun()
                else:
                    st.error("Failed to generate card design. Please try again.")
            else:
                st.error("Failed to start card design generation.")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back to Chip"):
            st.session_state.current_step = 1
            st.rerun()
    with col3:
        if st.session_state.generated_card:
            if st.button("Next: Combine →", type="primary"):
                st.session_state.current_step = 3
                st.rerun()

def step_3_combine_export():
    """Step 3: Combine & Export Interface"""
    st.header("🔄 Step 3: Combine & Export")
    st.write("Combine the generated chip with the card design to create your final credit card. Download the result when ready.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Generated Chip")
        if st.session_state.generated_chip:
            st.image(st.session_state.generated_chip, use_container_width=True)
        else:
            st.markdown("""
            <div class="preview-container">
                <h3>🔧</h3>
                <p>No chip generated</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Card Design")
        if st.session_state.generated_card:
            st.image(st.session_state.generated_card, use_container_width=True)
        else:
            st.markdown("""
            <div class="preview-container">
                <h3>🎨</h3>
                <p>No design generated</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.subheader("Final Result")
        if st.session_state.final_result:
            st.image(st.session_state.final_result, use_container_width=True)
        else:
            st.markdown("""
            <div class="preview-container">
                <h3>✨</h3>
                <p>Click "Combine" to create</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        can_combine = st.session_state.generated_chip and st.session_state.generated_card
        
        if st.button("🔄 Combine Elements", type="primary", use_container_width=True, disabled=not can_combine):
            if can_combine:
                with st.spinner("Combining elements..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Execute the actual combination workflow
                    prompt_id = combine_elements_workflow()
                    
                    if prompt_id:
                        status_text.text("Processing combination...")
                        progress_bar.progress(0.2)
                        
                        final_image_url = wait_for_completion(prompt_id, progress_bar, status_text)
                        
                        if final_image_url:
                            st.session_state.final_result = final_image_url
                            progress_bar.progress(1.0)
                            status_text.text("Elements combined successfully!")
                            st.success("✅ Final card created successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to combine elements. Please try again.")
                    else:
                        st.error("Failed to start combination process.")
            else:
                st.error("Please generate both chip and card design first.")
        
        if st.session_state.final_result:
            if st.button("📥 Download Card", use_container_width=True):
                # Convert image URL to downloadable format
                try:
                    response = requests.get(st.session_state.final_result, timeout=30)
                    img_data = response.content
                    
                    st.download_button(
                        label="Download Final Card",
                        data=img_data,
                        file_name=f"credit_card_design_{int(time.time())}.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Download failed: {str(e)}")
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back to Design"):
            st.session_state.current_step = 2
            st.rerun()
    with col3:
        if st.button("🔄 Start Over", type="secondary"):
            # Add confirmation in session state
            st.session_state.show_reset_confirmation = True
            st.rerun()
        
        # Show confirmation dialog if needed
        if hasattr(st.session_state, 'show_reset_confirmation') and st.session_state.show_reset_confirmation:
            st.warning("⚠️ This will delete all generated content!")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("✅ Yes, Reset All", type="primary", use_container_width=True):
                    st.session_state.generated_chip = None
                    st.session_state.generated_card = None
                    st.session_state.final_result = None
                    st.session_state.enhanced_prompt = None
                    st.session_state.original_prompt = None
                    st.session_state.current_step = 1
                    st.session_state.show_reset_confirmation = False
                    st.success("🔄 Application reset successfully!")
                    st.rerun()
            with col_no:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_reset_confirmation = False
                    st.rerun()

def main():
    """Main application function"""
    initialize_session_state()
    
    render_header()
    render_step_indicator()
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        st.write(f"Current Step: {st.session_state.current_step}")
        
        # Server status check
        try:
            response = requests.get(f"{COMFYUI_SERVER_URL}/system_stats", timeout=5)
            if response.status_code == 200:
                st.success("✅ ComfyUI Server Connected")
                # Show server info
                try:
                    stats = response.json()
                    st.caption(f"Queue: {stats.get('exec_info', {}).get('queue_remaining', 'Unknown')}")
                except:
                    pass
            else:
                st.error(f"❌ ComfyUI Server Error (Status: {response.status_code})")
        except requests.exceptions.ConnectionError:
            st.error("❌ ComfyUI Server Offline - Connection refused")
            st.caption("Make sure ComfyUI is running on the configured URL")
        except requests.exceptions.Timeout:
            st.error("❌ ComfyUI Server Timeout")
        except Exception as e:
            st.error(f"❌ ComfyUI Server Error: {str(e)}")
        
        st.markdown("---")
        
        # Progress summary
        st.subheader("Progress")
        st.write(f"✅ Chip Generated: {'Yes' if st.session_state.generated_chip else 'No'}")
        st.write(f"✅ Card Designed: {'Yes' if st.session_state.generated_card else 'No'}")
        st.write(f"✅ Final Result: {'Yes' if st.session_state.final_result else 'No'}")
        
        # Reset button with confirmation
        if st.button("🔄 Reset All", type="secondary"):
            st.session_state.show_sidebar_reset_confirmation = True
            st.rerun()
        
        # Show sidebar confirmation if needed
        if hasattr(st.session_state, 'show_sidebar_reset_confirmation') and st.session_state.show_sidebar_reset_confirmation:
            st.warning("⚠️ Reset all progress?")
            if st.button("✅ Confirm Reset", type="primary", use_container_width=True):
                st.session_state.generated_chip = None
                st.session_state.generated_card = None
                st.session_state.final_result = None
                st.session_state.enhanced_prompt = None
                st.session_state.original_prompt = None
                st.session_state.current_step = 1
                st.session_state.show_sidebar_reset_confirmation = False
                st.success("🔄 Reset complete!")
                st.rerun()
            if st.button("❌ Cancel Reset", use_container_width=True):
                st.session_state.show_sidebar_reset_confirmation = False
                st.rerun()
    
    # Main content based on current step
    if st.session_state.current_step == 1:
        step_1_chip_generation()
    elif st.session_state.current_step == 2:
        step_2_card_design()
    elif st.session_state.current_step == 3:
        step_3_combine_export()

if __name__ == "__main__":
    main()
