import modal
import os

# Define the container image and dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "libgl1-mesa-glx", "libglib2.0-0") # Required for OpenCV and compiling C extensions
    .pip_install(
        "fastapi==0.111.0",
        "uvicorn==0.30.1",
        "python-multipart==0.0.9",
        "httpx==0.27.0",
        "sqlalchemy==2.0.30",
        "asyncpg==0.29.0",
        "pydantic==2.7.2",
        "numpy==1.26.4",
        "shapely==2.0.4",
        "torch==2.3.0",
        "torchvision==0.18.0",
        "opencv-python-headless==4.9.0.80"
    )
    .run_commands(
        "pip install git+https://github.com/IDEA-Research/GroundingDINO.git",
        "pip install git+https://github.com/facebookresearch/segment-anything-2.git"
    )
    .run_commands(
        "mkdir -p /weights",
        "wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P /weights/",
        "wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -P /weights/"
    )
    # Modern Modal syntax: Copy the local codebase into the image directly
    .add_local_dir(
        local_path=os.path.dirname(__file__), 
        remote_path="/root/services/pre_inference"
    )
    .add_local_dir(
        local_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../common")),
        remote_path="/root/common"
    )
)

app = modal.App("pre-inference-service")

@app.function(
    image=image, 
    gpu="T4", 
    secrets=[modal.Secret.from_name("modal-db-secrets")],
    timeout=600 # 10 mins timeout
)
@modal.asgi_app()
def fastapi_app():
    import sys
    # Ensure our mounted paths are in the Python search path
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/services/pre_inference")
    
    # Import the FastAPI app inside the function to ensure it loads within the container context
    from main import app as web_app
    return web_app
