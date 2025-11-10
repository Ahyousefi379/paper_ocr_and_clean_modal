"""
OlmOCR API Server for Modal.com with A100 GPU
Fully integrated with Modal's serverless infrastructure
"""

import asyncio
import base64
import io
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

import modal

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Modal Configuration ---
app = modal.App("olmocr-api")

# Create a Modal image with all dependencies including system packages
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "poppler-utils",  # Required for pdfinfo, pdftoppm, pdftotext
        "libpoppler-cpp-dev",
        "pkg-config",
        "python3-dev",
    )
    .pip_install(
        "fastapi[standard]",
        "httpx",
        "huggingface-hub",
        "pypdf",
        "pydantic",
        "python-multipart",
        "pillow",
        "vllm>=0.6.0",
        "pdf2image",  # For PDF rendering
    )
    .pip_install(
        "olmocr",  # Add your custom package if it's on PyPI
        # Or use .run_commands() to install from git if needed:
        # Example: "git+https://github.com/allenai/olmocr.git"
    ).pip_install_from_requirements("requirements.txt")
)

# Shared volume for model caching
volume = modal.Volume.from_name("olmocr-models", create_if_missing=True)

# Model path constant
MODEL_NAME = "allenai/olmOCR-7B-0825-FP8"
MODEL_CACHE_PATH = "/cache/models"

# --- Configuration ---
VLLM_PORT = 8000
MAX_PAGE_RETRIES = 8
TARGET_LONGEST_IMAGE_DIM = 1288
MAX_PAGE_ERROR_RATE = 0.004
GPU_MEMORY_UTILIZATION = 0.95
MAX_MODEL_LEN = 16384
TENSOR_PARALLEL_SIZE = 1
TASK_TIMEOUT = 240

# Global task storage (in production, use a database or Redis)
tasks: Dict[str, Dict[str, Any]] = {}


# --- OCR Processing Functions ---
async def process_page(vllm_url: str, pdf_path: str, page_num: int) -> Dict[str, Any]:
    """Process a single page with retry logic"""
    import httpx
    from io import BytesIO
    from PIL import Image
    from olmocr.data.renderpdf import render_pdf_to_base64png
    from olmocr.prompts.prompts import PageResponse, build_no_anchoring_yaml_prompt
    from olmocr.prompts.anchor import get_anchor_text
    from olmocr.train.dataloader import FrontMatterParser

    MAX_TOKENS = 4500
    TEMPERATURES = [0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0]
    attempt, cumulative_rotation = 0, 0

    while attempt < MAX_PAGE_RETRIES:
        try:
            image_base64 = await asyncio.to_thread(
                render_pdf_to_base64png, pdf_path, page_num,
                target_longest_image_dim=TARGET_LONGEST_IMAGE_DIM
            )

            if cumulative_rotation:
                image_bytes = base64.b64decode(image_base64)
                with Image.open(BytesIO(image_bytes)) as img:
                    transpose = {
                        90: Image.Transpose.ROTATE_90,
                        180: Image.Transpose.ROTATE_180,
                        270: Image.Transpose.ROTATE_270
                    }[cumulative_rotation]
                    img = img.transpose(transpose)
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    image_base64 = base64.b64encode(buf.getvalue()).decode()

            query = {
                "model": "olmocr",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_no_anchoring_yaml_prompt()},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                }],
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURES[min(attempt, len(TEMPERATURES) - 1)]
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(f"{vllm_url}/v1/chat/completions", json=query)
            
            if r.status_code != 200:
                raise ValueError(f"vLLM status {r.status_code}: {r.text}")
            
            data = r.json()
            model_response = data["choices"][0]["message"]["content"]
            parser = FrontMatterParser(front_matter_class=PageResponse)
            fm, text = parser._extract_front_matter_and_text(model_response)
            pr = parser._parse_front_matter(fm, text)

            if not pr.is_rotation_valid and attempt < MAX_PAGE_RETRIES - 1:
                cumulative_rotation = (cumulative_rotation + pr.rotation_correction) % 360
                attempt += 1
                continue

            return {
                "text": pr.natural_text,
                "tokens": {
                    "input": data["usage"].get("prompt_tokens", 0),
                    "output": data["usage"].get("completion_tokens", 0)
                },
                "success": True
            }
        except Exception as e:
            logger.warning(f"Page {page_num} attempt {attempt + 1} failed: {e}")
            attempt += 1
            await asyncio.sleep(min(2 ** attempt, 10))

    # Fallback to anchor text
    try:
        from olmocr.prompts.anchor import get_anchor_text
        fb_text = get_anchor_text(pdf_path, page_num, pdf_engine="pdftotext")
    except Exception as ex:
        logger.warning(f"Fallback anchor text extraction failed for page {page_num}: {ex}")
        fb_text = ""
    
    return {"text": fb_text, "tokens": {"input": 0, "output": 0}, "success": False}


async def process_pdf(vllm_url: str, pdf_path: str, filename: str) -> Dict[str, Any]:
    """Process entire PDF document"""
    from pypdf import PdfReader
    
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)

    sem = asyncio.Semaphore(4)
    
    async def worker(i):
        async with sem:
            return await process_page(vllm_url, pdf_path, i)

    logger.info(f"Processing {num_pages} pages from {filename}")
    results = await asyncio.gather(*[worker(i) for i in range(1, num_pages + 1)])

    text = "\n".join(r["text"] for r in results if r.get("text"))
    total_in = sum(r["tokens"]["input"] for r in results)
    total_out = sum(r["tokens"]["output"] for r in results)
    failed = sum(1 for r in results if not r["success"])

    error_rate = failed / num_pages if num_pages else 0
    if error_rate > MAX_PAGE_ERROR_RATE:
        raise ValueError(f"Too many failed pages: {failed}/{num_pages}")

    return {
        "text": text,
        "metadata": {"filename": filename, "model_used": MODEL_NAME},
        "completed_pages": num_pages - failed,
        "failed_pages": failed,
        "page_failure_rate": f"{(failed / num_pages * 100):.2f}%" if num_pages else "0.00%",
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
    }


# --- Modal Class for vLLM Server ---
@app.cls(
    image=image,
    gpu=modal.gpu.H100(),  # A100 40GB GPU
    volumes={MODEL_CACHE_PATH: volume},
    timeout=3600,  # 1 hour timeout
    container_idle_timeout=600,  # Keep warm for 5 minutes
    allow_concurrent_inputs=10,
)
class VLLMServer:
    @modal.enter()
    def start_vllm(self):
        """Start vLLM server when container starts"""
        import subprocess
        import sys
        import time
        import httpx
        
        logger.info("Starting vLLM server...")
        
        # Verify pdfinfo is available
        try:
            result = subprocess.run(["which", "pdfinfo"], capture_output=True, text=True)
            logger.info(f"pdfinfo location: {result.stdout.strip()}")
        except Exception as e:
            logger.warning(f"Could not verify pdfinfo: {e}")
        
        # Start vLLM in background
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL_NAME,
            "--port", str(VLLM_PORT),
            "--disable-log-requests",
            "--served-model-name", "olmocr",
            "--tensor-parallel-size", str(TENSOR_PARALLEL_SIZE),
            "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
            "--max-model-len", str(MAX_MODEL_LEN),
            "--download-dir", MODEL_CACHE_PATH,
        ]
        
        self.vllm_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "OMP_NUM_THREADS": "1"}
        )
        
        # Wait for server to be ready
        url = f"http://localhost:{VLLM_PORT}/v1/models"
        max_retries = 120  # 10 minutes
        
        for i in range(max_retries):
            try:
                response = httpx.get(url, timeout=5.0)
                if response.status_code == 200:
                    logger.info(f"vLLM server ready after {i * 5}s")
                    self.vllm_url = f"http://localhost:{VLLM_PORT}"
                    return
            except Exception as e:
                if i % 12 == 0:  # Log every minute
                    logger.info(f"Waiting for vLLM... ({i * 5}s)")
            time.sleep(5)
        
        raise RuntimeError("vLLM server failed to start")
    
    @modal.exit()
    def stop_vllm(self):
        """Stop vLLM server when container shuts down"""
        if hasattr(self, 'vllm_process'):
            logger.info("Stopping vLLM server...")
            self.vllm_process.terminate()
            try:
                self.vllm_process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.vllm_process.kill()
    
    @modal.method()
    async def process_document(self, file_content: bytes, filename: str, task_id: str) -> Dict[str, Any]:
        """Process a document (PDF or image)"""
        from olmocr.image_utils import convert_image_to_pdf_bytes, is_jpeg, is_png
        
        start = time.time()
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            try:
                tmp.write(file_content)
                tmp.flush()

                # Handle image files by converting to PDF
                if is_png(tmp.name) or is_jpeg(tmp.name):
                    logger.info(f"Converting image {filename} to PDF")
                    pdf_bytes = convert_image_to_pdf_bytes(tmp.name)
                    tmp.seek(0)
                    tmp.truncate()
                    tmp.write(pdf_bytes)
                    tmp.flush()

                # Process PDF with timeout
                try:
                    result = await asyncio.wait_for(
                        process_pdf(self.vllm_url, tmp.name, filename),
                        timeout=TASK_TIMEOUT
                    )
                    result["processing_time"] = time.time() - start
                    result["status"] = "complete"
                    logger.info(f"Task {task_id} completed in {result['processing_time']:.1f}s")
                    return result
                except asyncio.TimeoutError:
                    elapsed = time.time() - start
                    error_msg = f"Task timed out after {elapsed:.1f}s"
                    logger.error(f"Task {task_id} timed out: {error_msg}")
                    return {
                        "status": "failed",
                        "error": error_msg,
                        "timeout": True,
                        "processing_time": elapsed
                    }
            except Exception as e:
                elapsed = time.time() - start
                logger.exception(f"Task {task_id} failed: {str(e)}")
                return {
                    "status": "failed",
                    "error": str(e),
                    "timeout": False,
                    "processing_time": elapsed
                }
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)


# --- FastAPI Web Endpoints ---
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

web_app = FastAPI(
    title="OlmOCR API Server",
    description="API for OCR processing using OlmOCR model on Modal",
    version="1.0.0"
)


class SubmitResponse(BaseModel):
    task_id: str = Field(..., description="The unique ID for the processing task.")


class StatusResponse(BaseModel):
    status: str = Field(..., description="The current status of the task.")
    result: Dict[str, Any] = Field(default_factory=dict, description="The OCR result if complete.")


@web_app.post("/submit", response_model=SubmitResponse)
async def submit_ocr_task(file: UploadFile = File(...)):
    """Submit a PDF or image file for OCR processing"""
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing", "result": {}}
    
    content = await file.read()
    logger.info(f"Submitted task {task_id} for file: {file.filename}")
    
    # Process asynchronously
    server = VLLMServer()
    
    async def process_async():
        try:
            result = await server.process_document.remote.aio(content, file.filename, task_id)
            tasks[task_id]["status"] = result.get("status", "complete")
            tasks[task_id]["result"] = result
        except Exception as e:
            logger.exception(f"Error processing task {task_id}")
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["result"] = {"error": str(e)}
    
    # Start processing in background
    asyncio.create_task(process_async())
    
    return {"task_id": task_id}


@web_app.get("/status/{task_id}", response_model=StatusResponse)
async def get_task_status(task_id: str):
    """Get the status of a submitted OCR task"""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@web_app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_tasks": len([t for t in tasks.values() if t["status"] == "processing"])
    }


@web_app.get("/tasks")
async def list_tasks():
    """List all tasks and their statuses"""
    return {
        "total_tasks": len(tasks),
        "processing": len([t for t in tasks.values() if t["status"] == "processing"]),
        "completed": len([t for t in tasks.values() if t["status"] == "complete"]),
        "failed": len([t for t in tasks.values() if t["status"] == "failed"]),
        "tasks": {k: {"status": v["status"]} for k, v in tasks.items()}
    }


# --- Mount FastAPI app to Modal ---
@app.function(
    image=image,
    keep_warm=1,  # Keep one instance warm
    timeout=600,
)
@modal.asgi_app()
def fastapi_app():
    return web_app


# --- CLI for local testing ---
@app.local_entrypoint()
def main():
    """Local CLI for testing"""
    import sys
    if len(sys.argv) < 2:
        print("Usage: modal run modal_olmocr_server.py <path_to_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    with open(pdf_path, "rb") as f:
        content = f.read()
    
    server = VLLMServer()
    result = server.process_document.remote(content, pdf_path, "test-task")
    print(result)