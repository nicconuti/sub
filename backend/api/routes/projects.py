"""
Project Management API Routes
REST endpoints for project CRUD operations
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, List, Optional, Any
import logging
import json
import io
import pandas as pd
from datetime import datetime
import uuid

from ..models.events import SimulationParams, SourceData

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for demo (replace with database in production)
_projects_storage: Dict[str, Dict[str, Any]] = {}

@router.post("/", response_model=Dict[str, Any])
async def create_project(project_data: Dict[str, Any]):
    """Create new project"""
    try:
        # Validate required fields
        if 'name' not in project_data:
            raise HTTPException(status_code=400, detail="Project name is required")
        
        # Generate project ID
        project_id = str(uuid.uuid4())
        
        # Create project structure
        project = {
            "id": project_id,
            "name": project_data["name"],
            "description": project_data.get("description", ""),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "sources": project_data.get("sources", []),
            "simulation_params": project_data.get("simulation_params", {
                "frequency": 80.0,
                "speed_of_sound": 343.0,
                "grid_resolution": 0.1,
                "room_vertices": [[0, 0], [10, 0], [10, 8], [0, 8]],
                "target_areas": [],
                "avoidance_areas": []
            }),
            "target_areas": project_data.get("target_areas", []),
            "avoidance_areas": project_data.get("avoidance_areas", []),
            "results": project_data.get("results", {}),
            "metadata": project_data.get("metadata", {})
        }
        
        # Store project
        _projects_storage[project_id] = project
        
        logger.info(f"📁 Created project {project_id}: {project['name']}")
        
        return {
            "message": "Project created successfully",
            "project_id": project_id,
            "project": project
        }
        
    except Exception as e:
        logger.error(f"❌ Error creating project: {e}")
        raise HTTPException(status_code=500, detail=f"Project creation failed: {str(e)}")

@router.get("/", response_model=List[Dict[str, Any]])
async def list_projects():
    """List all projects"""
    try:
        projects = list(_projects_storage.values())
        
        # Sort by updated_at (most recent first)
        projects.sort(key=lambda p: p.get("updated_at", ""), reverse=True)
        
        # Return summary info only
        project_summaries = []
        for project in projects:
            summary = {
                "id": project["id"],
                "name": project["name"],
                "description": project["description"],
                "created_at": project["created_at"],
                "updated_at": project["updated_at"],
                "sources_count": len(project.get("sources", [])),
                "has_results": bool(project.get("results"))
            }
            project_summaries.append(summary)
        
        return project_summaries
        
    except Exception as e:
        logger.error(f"❌ Error listing projects: {e}")
        raise HTTPException(status_code=500, detail=f"Project listing failed: {str(e)}")

@router.get("/{project_id}", response_model=Dict[str, Any])
async def get_project(project_id: str):
    """Get specific project"""
    try:
        if project_id not in _projects_storage:
            raise HTTPException(status_code=404, detail="Project not found")
        
        project = _projects_storage[project_id]
        
        return {
            "message": "Project retrieved successfully",
            "project": project
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Project retrieval failed: {str(e)}")

@router.put("/{project_id}", response_model=Dict[str, Any])
async def update_project(project_id: str, project_data: Dict[str, Any]):
    """Update existing project"""
    try:
        if project_id not in _projects_storage:
            raise HTTPException(status_code=404, detail="Project not found")
        
        project = _projects_storage[project_id]
        
        # Update fields
        if 'name' in project_data:
            project['name'] = project_data['name']
        if 'description' in project_data:
            project['description'] = project_data['description']
        if 'sources' in project_data:
            project['sources'] = project_data['sources']
        if 'simulation_params' in project_data:
            project['simulation_params'] = project_data['simulation_params']
        if 'target_areas' in project_data:
            project['target_areas'] = project_data['target_areas']
        if 'avoidance_areas' in project_data:
            project['avoidance_areas'] = project_data['avoidance_areas']
        if 'results' in project_data:
            project['results'] = project_data['results']
        if 'metadata' in project_data:
            project['metadata'] = project_data['metadata']
        
        project['updated_at'] = datetime.now().isoformat()
        
        logger.info(f"📁 Updated project {project_id}: {project['name']}")
        
        return {
            "message": "Project updated successfully",
            "project": project
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error updating project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Project update failed: {str(e)}")

@router.delete("/{project_id}")
async def delete_project(project_id: str):
    """Delete project"""
    try:
        if project_id not in _projects_storage:
            raise HTTPException(status_code=404, detail="Project not found")
        
        project_name = _projects_storage[project_id]["name"]
        del _projects_storage[project_id]
        
        logger.info(f"🗑️ Deleted project {project_id}: {project_name}")
        
        return {
            "message": f"Project '{project_name}' deleted successfully",
            "project_id": project_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Project deletion failed: {str(e)}")

@router.post("/import")
async def import_project(file: UploadFile = File(...), project_name: str = Form(...)):
    """Import project from Excel or JSON file"""
    try:
        # Check file type
        if file.content_type not in ["application/json", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            raise HTTPException(status_code=400, detail="Only JSON and Excel files are supported")
        
        # Read file content
        content = await file.read()
        
        if file.content_type == "application/json":
            # Parse JSON
            try:
                data = json.loads(content.decode('utf-8'))
                project_data = data
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
                
        else:
            # Parse Excel
            try:
                df = pd.read_excel(io.BytesIO(content))
                
                # Convert Excel to project format
                project_data = {
                    "sources": [],
                    "simulation_params": {
                        "frequency": 80.0,
                        "speed_of_sound": 343.0,
                        "grid_resolution": 0.1,
                        "room_vertices": [[0, 0], [10, 0], [10, 8], [0, 8]]
                    }
                }
                
                # Parse sources from Excel
                for _, row in df.iterrows():
                    source = {
                        "id": f"source_{len(project_data['sources']) + 1}",
                        "x": float(row.get('x', 0)),
                        "y": float(row.get('y', 0)),
                        "spl_rms": float(row.get('spl_rms', 105)),
                        "gain_db": float(row.get('gain_db', 0)),
                        "delay_ms": float(row.get('delay_ms', 0)),
                        "angle": float(row.get('angle', 0)),
                        "polarity": int(row.get('polarity', 1))
                    }
                    project_data["sources"].append(source)
                    
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid Excel format: {str(e)}")
        
        # Add project name
        project_data["name"] = project_name
        project_data["description"] = f"Imported from {file.filename}"
        
        # Create project
        result = await create_project(project_data)
        
        return {
            "message": "Project imported successfully",
            "project_id": result["project_id"],
            "sources_imported": len(project_data.get("sources", [])),
            "file_type": file.content_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error importing project: {e}")
        raise HTTPException(status_code=500, detail=f"Project import failed: {str(e)}")

@router.get("/{project_id}/export")
async def export_project(project_id: str, format: str = "json"):
    """Export project to JSON or Excel"""
    try:
        if project_id not in _projects_storage:
            raise HTTPException(status_code=404, detail="Project not found")
        
        project = _projects_storage[project_id]
        
        if format.lower() == "json":
            # Export as JSON
            json_data = json.dumps(project, indent=2)
            
            return StreamingResponse(
                io.StringIO(json_data),
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename={project['name']}.json"
                }
            )
            
        elif format.lower() == "xlsx":
            # Export as Excel
            sources_data = []
            for source in project.get("sources", []):
                sources_data.append({
                    "id": source.get("id", ""),
                    "x": source.get("x", 0),
                    "y": source.get("y", 0),
                    "spl_rms": source.get("spl_rms", 105),
                    "gain_db": source.get("gain_db", 0),
                    "delay_ms": source.get("delay_ms", 0),
                    "angle": source.get("angle", 0),
                    "polarity": source.get("polarity", 1)
                })
            
            df = pd.DataFrame(sources_data)
            
            # Create Excel file in memory
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Sources', index=False)
                
                # Add project info sheet
                info_df = pd.DataFrame([
                    ["Project Name", project["name"]],
                    ["Description", project["description"]],
                    ["Created", project["created_at"]],
                    ["Updated", project["updated_at"]],
                    ["Frequency", project["simulation_params"].get("frequency", 80)],
                    ["Speed of Sound", project["simulation_params"].get("speed_of_sound", 343)],
                    ["Grid Resolution", project["simulation_params"].get("grid_resolution", 0.1)]
                ], columns=["Property", "Value"])
                
                info_df.to_excel(writer, sheet_name='Project Info', index=False)
            
            excel_buffer.seek(0)
            
            return StreamingResponse(
                io.BytesIO(excel_buffer.read()),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={
                    "Content-Disposition": f"attachment; filename={project['name']}.xlsx"
                }
            )
            
        else:
            raise HTTPException(status_code=400, detail="Supported formats: json, xlsx")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error exporting project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Project export failed: {str(e)}")

@router.post("/{project_id}/duplicate")
async def duplicate_project(project_id: str, new_name: Optional[str] = None):
    """Duplicate existing project"""
    try:
        if project_id not in _projects_storage:
            raise HTTPException(status_code=404, detail="Project not found")
        
        original_project = _projects_storage[project_id]
        
        # Create duplicate
        duplicate_data = original_project.copy()
        duplicate_data["name"] = new_name or f"{original_project['name']} (Copy)"
        duplicate_data["description"] = f"Copy of {original_project['name']}"
        
        # Remove results (they're specific to the original)
        duplicate_data["results"] = {}
        
        # Create new project
        result = await create_project(duplicate_data)
        
        return {
            "message": "Project duplicated successfully",
            "original_id": project_id,
            "duplicate_id": result["project_id"],
            "duplicate_name": duplicate_data["name"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error duplicating project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Project duplication failed: {str(e)}")

@router.get("/{project_id}/sources")
async def get_project_sources(project_id: str):
    """Get sources for specific project"""
    try:
        if project_id not in _projects_storage:
            raise HTTPException(status_code=404, detail="Project not found")
        
        project = _projects_storage[project_id]
        sources = project.get("sources", [])
        
        return {
            "project_id": project_id,
            "sources": sources,
            "sources_count": len(sources)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting sources for project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Sources retrieval failed: {str(e)}")

@router.put("/{project_id}/sources")
async def update_project_sources(project_id: str, sources: List[Dict[str, Any]]):
    """Update sources for specific project"""
    try:
        if project_id not in _projects_storage:
            raise HTTPException(status_code=404, detail="Project not found")
        
        project = _projects_storage[project_id]
        project["sources"] = sources
        project["updated_at"] = datetime.now().isoformat()
        
        logger.info(f"📁 Updated sources for project {project_id}: {len(sources)} sources")
        
        return {
            "message": "Sources updated successfully",
            "project_id": project_id,
            "sources_count": len(sources)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error updating sources for project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Sources update failed: {str(e)}")

@router.get("/stats")
async def get_projects_stats():
    """Get projects statistics"""
    try:
        total_projects = len(_projects_storage)
        total_sources = sum(len(p.get("sources", [])) for p in _projects_storage.values())
        
        # Find most recent project
        recent_project = None
        if _projects_storage:
            recent_project = max(
                _projects_storage.values(),
                key=lambda p: p.get("updated_at", "")
            )
        
        return {
            "total_projects": total_projects,
            "total_sources": total_sources,
            "average_sources_per_project": total_sources / max(1, total_projects),
            "most_recent_project": {
                "id": recent_project["id"],
                "name": recent_project["name"],
                "updated_at": recent_project["updated_at"]
            } if recent_project else None
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting project stats: {e}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")