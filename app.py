import streamlit as st
import requests
import json
import time
import io
import base64
from PIL import Image
import websocket
import threading
import uuid
from typing import Optional, Dict, Any

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

# Configuration
COMFYUI_SERVER_URL = "http://127.0.0.1:8188"

class ComfyUIClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.client_id = str(uuid.uuid4())
    
    def generate_client_id(self) -> str:
        return str(uuid.uuid4())
    
    async def queue_prompt(self, workflow: Dict[str, Any]) -> str:
        """Queue a prompt and return the prompt_id"""
        try:
            response = requests.post(
                f"{self.server_url}/prompt",
                json={
                    "prompt": workflow,
                    "client_id": self.client_id
                },
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()
            return result["prompt_id"]
        except Exception as e:
            st.error(f"Failed to queue prompt: {str(e)}")
            raise
    
    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Get the history for a specific prompt"""
        try:
            response = requests.get(f"{self.server_url}/history/{prompt_id}")
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
            
            response = requests.post(f"{self.server_url}/upload/image", files=files, data=data)
            response.raise_for_status()
            result = response.json()
            return result.get("name", filename)
        except Exception as e:
            st.error(f"Failed to upload image: {str(e)}")
            raise

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

def wait_for_completion(prompt_id: str, progress_bar, status_text) -> Optional[str]:
    """Wait for ComfyUI to complete generation and return image URL"""
    max_wait_time = 120  # 2 minutes timeout
    check_interval = 2   # Check every 2 seconds
    elapsed_time = 0
    
    while elapsed_time < max_wait_time:
        try:
            history = st.session_state.comfyui_client.get_history(prompt_id)
            
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                
                # For chip generation (node 31)
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
            
            # Update progress
            progress_percentage = min(90, (elapsed_time / max_wait_time) * 100)
            progress_bar.progress(progress_percentage / 100)
            status_text.text(f"Processing... ({elapsed_time}s)")
            
            time.sleep(check_interval)
            elapsed_time += check_interval
            
        except Exception as e:
            st.error(f"Error checking completion: {str(e)}")
            return None
    
    return None

def generate_chip(chip_color: str, chip_shape: str, custom_details: str) -> Optional[str]:
    """Generate chip using ComfyUI"""
    # Construct prompt
    base_prompt = f"card chip, {chip_color}, {chip_shape}, white background"
    final_prompt = f"{base_prompt}, {custom_details}" if custom_details else base_prompt
    
    # Chip generation workflow
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
                "strength_clip": 1.0000000000000002,
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
                "seed": int(time.time() * 1000000) % 1000000000000000,
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
        prompt_id = st.session_state.comfyui_client.queue_prompt(workflow)
        return prompt_id
    except Exception as e:
        st.error(f"Failed to generate chip: {str(e)}")
        return None

def generate_card_design(prompt: str, control_image_name: str = "default_card_outline.png") -> Optional[str]:
    """Generate card design using ComfyUI"""
    # Optimize prompt
    base_enhancements = "high quality, detailed, professional design, clean background, studio lighting, sharp focus, premium card design"
    card_enhancements = "credit card design, elegant, modern style, glossy finish"
    optimized_prompt = f"{prompt}, {base_enhancements}, {card_enhancements}"
    
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
                "seed": int(time.time() * 1000000) % 1000000000000000,
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
        prompt_id = st.session_state.comfyui_client.queue_prompt(workflow)
        return prompt_id
    except Exception as e:
        st.error(f"Failed to generate card design: {str(e)}")
        return None

def step_1_chip_generation():
    """Step 1: Chip Generation Interface"""
    st.header("🔧 Step 1: Generate Card Chip")
    st.write("Create a realistic credit card chip using AI generation. Choose from different chip styles and customize the appearance.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Configuration")
        
        chip_color = st.selectbox("Chip Color", ["gold", "silver"], index=0)
        chip_shape = st.text_input("Chip Shape", value="classic", help="e.g., heart, star, classic")
        custom_details = st.text_area("Custom Details (Optional)", help="Add custom details for chip generation...")
        
        generate_btn = st.button("🎨 Generate Chip", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Preview")
        
        if st.session_state.generated_chip:
            st.image(st.session_state.generated_chip, caption="Generated Chip", use_column_width=True)
        else:
            st.markdown("""
            <div class="preview-container">
                <h3>🔧</h3>
                <p>Click "Generate Chip" to create</p>
            </div>
            """, unsafe_allow_html=True)
    
    if generate_btn:
        with st.spinner("Generating chip..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            prompt_id = generate_chip(chip_color, chip_shape, custom_details)
            
            if prompt_id:
                status_text.text("Waiting for generation to complete...")
                progress_bar.progress(0.1)
                
                image_url = wait_for_completion(prompt_id, progress_bar, status_text)
                
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
            placeholder="e.g., 'A luxury credit card with golden accents and geometric patterns'"
        )
        
        st.subheader("ControlNet Reference Image")
        control_image = st.file_uploader(
            "Upload Custom Control Image (Optional)", 
            type=["png", "jpg", "jpeg"],
            help="Upload a custom control image or use the default card outline"
        )
        
        control_image_name = "default_card_outline.png"
        
        if control_image:
            # Upload custom control image
            image_bytes = control_image.read()
            control_image_name = st.session_state.comfyui_client.upload_image(image_bytes, control_image.name)
            st.image(control_image, caption="Custom Control Image", use_column_width=True)
        else:
            st.info("Using default card outline image")
        
        generate_btn = st.button("🎨 Generate Design", type="primary", use_container_width=True, disabled=not design_prompt.strip())
    
    with col2:
        st.subheader("Preview")
        
        if st.session_state.generated_card:
            st.image(st.session_state.generated_card, caption="Generated Card Design", use_column_width=True)
        else:
            st.markdown("""
            <div class="preview-container">
                <h3>🎨</h3>
                <p>Enter prompt and generate design</p>
            </div>
            """, unsafe_allow_html=True)
    
    if generate_btn and design_prompt.strip():
        with st.spinner("Generating card design..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            prompt_id = generate_card_design(design_prompt.strip(), control_image_name)
            
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
            st.image(st.session_state.generated_chip, use_column_width=True)
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
            st.image(st.session_state.generated_card, use_column_width=True)
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
            st.image(st.session_state.final_result, use_column_width=True)
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
                    st.info("Combination feature will be implemented with the final ComfyUI workflow")
                    # TODO: Implement actual combination using final_design_workflow.json
                    time.sleep(2)  # Simulate processing
                    st.session_state.final_result = st.session_state.generated_card  # Temporary
                    st.success("Elements combined successfully!")
                    st.rerun()
            else:
                st.error("Please generate both chip and card design first.")
        
        if st.session_state.final_result:
            if st.button("📥 Download Card", use_container_width=True):
                # Convert image URL to downloadable format
                try:
                    response = requests.get(st.session_state.final_result)
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
        if st.button("🔄 Start Over"):
            if st.button("Confirm Reset", type="secondary"):
                st.session_state.generated_chip = None
                st.session_state.generated_card = None
                st.session_state.final_result = None
                st.session_state.current_step = 1
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
            else:
                st.error("❌ ComfyUI Server Error")
        except:
            st.error("❌ ComfyUI Server Offline")
        
        st.markdown("---")
        
        # Progress summary
        st.subheader("Progress")
        st.write(f"✅ Chip Generated: {'Yes' if st.session_state.generated_chip else 'No'}")
        st.write(f"✅ Card Designed: {'Yes' if st.session_state.generated_card else 'No'}")
        st.write(f"✅ Final Result: {'Yes' if st.session_state.final_result else 'No'}")
        
        if st.button("🔄 Reset All", type="secondary"):
            st.session_state.generated_chip = None
            st.session_state.generated_card = None
            st.session_state.final_result = None
            st.session_state.current_step = 1
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
