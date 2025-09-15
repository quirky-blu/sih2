from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio
import json
import os
from typing import Optional, Dict, Any
from pydantic import BaseModel
import logging
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Tripo3D API Backend",
    description="FastAPI backend for Tripo3D Text-to-3D API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration - Corrected API URL
TRIPO3D_BASE_URL = "https://api.tripo3d.ai/v2/openapi"
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    logger.error("API_KEY environment variable not set. Please add it to your .env file.")
else:
    logger.info(f"Tripo3D API key loaded successfully (key starts with: {API_KEY[:10]}...)")

class Text3DRequest(BaseModel):
    """Request model for text to 3D generation"""
    prompt: str
    model_version: Optional[str] = "v2.5-20250123"  # Updated default version

# Store for tracking active tasks
active_tasks = {}

async def get_tripo3d_headers() -> Dict[str, str]:
    """Get headers for Tripo3D API requests"""
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Tripo3D API key not configured")
    
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Tripo3D API Backend is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "tripo3d-backend"}

@app.post("/generate/text-to-3d")
async def create_text_to_3d_task(request: Text3DRequest):
    """Create a text-to-3D task"""
    try:
        headers = await get_tripo3d_headers()
        url = f"{TRIPO3D_BASE_URL}/task"
        
        # Prepare the payload according to Tripo3D API spec
        payload = {
            "type": "text_to_model",
            "prompt": request.prompt
        }
        
        # Add model version if specified
        if request.model_version:
            payload["model_version"] = request.model_version
        
        logger.info(f"Creating 3D task with payload: {payload}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"Tripo3D API error: {response.status_code} - {error_text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Tripo3D API error: {error_text}"
                )
            
            result = response.json()
            logger.info(f"API Response: {result}")
            
            # According to Tripo3D docs, the response structure is different
            # The task_id is in the data field, not directly in the response
            task_id = None
            
            if "data" in result and isinstance(result["data"], dict):
                task_id = result["data"].get("task_id")
            elif "task_id" in result:
                task_id = result.get("task_id")
            
            if not task_id:
                logger.error(f"No task ID found in response: {result}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"No task ID returned from Tripo3D API. Response: {result}"
                )
            
            # Store task info
            active_tasks[task_id] = {
                "created_at": time.time(),
                "prompt": request.prompt,
                "status": result.get("data", {}).get("status", "pending") if "data" in result else result.get("status", "pending")
            }
            
            logger.info(f"Task created successfully with ID: {task_id}")
            
            return JSONResponse({
                "task_id": task_id,
                "status": result.get("data", {}).get("status", "created") if "data" in result else result.get("status", "created"),
                "message": "3D generation task created successfully",
                "data": result
            })
            
    except httpx.TimeoutException:
        logger.error("Request to Tripo3D API timed out")
        raise HTTPException(status_code=408, detail="Request timeout")
    except httpx.RequestError as e:
        logger.error(f"Request error: {e}")
        raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get task status and results"""
    try:
        headers = await get_tripo3d_headers()
        url = f"{TRIPO3D_BASE_URL}/task/{task_id}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            
            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Task not found")
            elif response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to get task status: {response.text}"
                )
            
            task_data = response.json()
            
            # Update local task storage
            if task_id in active_tasks:
                status = task_data.get("data", {}).get("status", "unknown") if "data" in task_data else task_data.get("status", "unknown")
                active_tasks[task_id]["status"] = status
            
            return task_data
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks")
async def list_tasks(limit: int = 10):
    """List recent tasks from local storage"""
    try:
        # Return tasks from local storage
        tasks_list = []
        for task_id, task_info in list(active_tasks.items())[-limit:]:
            tasks_list.append({
                "task_id": task_id,
                "prompt": task_info.get("prompt", ""),
                "status": task_info.get("status", "unknown"),
                "created_at": task_info.get("created_at")
            })
        
        return {"tasks": tasks_list}
        
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task from local storage"""
    try:
        if task_id in active_tasks:
            del active_tasks[task_id]
            return {"message": f"Task {task_id} deleted from local storage"}
        else:
            raise HTTPException(status_code=404, detail="Task not found in local storage")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks/{task_id}/stream")
async def stream_task_updates(task_id: str):
    """Stream task status updates via Server-Sent Events"""
    async def generate_task_updates():
        headers = await get_tripo3d_headers()
        url = f"{TRIPO3D_BASE_URL}/task/{task_id}"

        # Send initial connection message
        yield f"data: {json.dumps({'status': 'connected', 'task_id': task_id})}\n\n"

        max_attempts = 180  # 6 minutes maximum (2 second intervals)
        attempts = 0

        while attempts < max_attempts:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(url, headers=headers)

                    if response.status_code == 200:
                        task_data = response.json()
                        status = task_data.get("data", {}).get("status", "unknown") if "data" in task_data else task_data.get("status", "unknown")

                        # Send update
                        yield f"data: {json.dumps(task_data)}\n\n"

                        # Check for completion status and send final result
                        if status == "success":
                            logger.info(f"Task {task_id} finished successfully. Extracting model URL.")
                            # Extract the URL from the response
                            model_url = task_data.get("data", {}).get("result", {}).get("model")
                            
                            if model_url:
                                final_payload = {
                                    "status": "completed",
                                    "task_id": task_id,
                                    "model_url": model_url,
                                    "message": "3D model is ready."
                                }
                                yield f"data: {json.dumps(final_payload)}\n\n"
                                logger.info(f"Sent final model URL for task {task_id}")
                            break
                        elif status in ["failed", "cancelled"]:
                            logger.info(f"Task {task_id} finished with status: {status}")
                            break

                await asyncio.sleep(2)  # Poll every 2 seconds
                attempts += 1
            
            except Exception as e:
                logger.error(f"Error in streaming: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                await asyncio.sleep(5) # Wait longer on error
        
        # Final message if we exit the loop
        if attempts >= max_attempts:
            yield f"data: {json.dumps({'status': 'timeout', 'message': 'Streaming timeout reached'})}\n\n"
        
        yield f"data: {json.dumps({'status': 'stream_ended'})}\n\n"

    return StreamingResponse(
        generate_task_updates(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )


@app.get("/models")
async def get_available_models():
    """Get available model versions"""
    return {
        "models": [
            {
                "id": "v2.5-20250123", 
                "name": "v2.5 (Latest Stable)", 
                "description": "Latest stable model with unprecedented detail and fidelity",
                "type": "stable"
            },
            {
                "id": "v2.0-20240919", 
                "name": "v2.0", 
                "description": "Industry-leading geometry with PBR materials",
                "type": "stable"
            },
            {
                "id": "Turbo-v1.0-20250506", 
                "name": "Turbo v1.0", 
                "description": "Fastest generation model",
                "type": "turbo"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
                )
