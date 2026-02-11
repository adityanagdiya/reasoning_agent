from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field

# --- Final Output Models (for Validation) ---

class FinalPosition(BaseModel):
    x: float
    y: float

class FinalNodeConfigData(BaseModel):
    label: Optional[str] = None
    notes: Optional[str] = None
    # Dynamic fields for different node types
    incomingSockets: Optional[str] = None
    outgoingSockets: Optional[str] = None
    logic: Optional[str] = None
    message: Optional[str] = None # For Core:Log
    expression: Optional[str] = None # For Core:If
    # HTTP Request fields
    mode: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    headers: Optional[str] = None
    queryStrings: Optional[str] = None
    body: Optional[str] = None
    typeOfResponse: Optional[str] = None
    proxy: Optional[str] = None
    proxyPort: Optional[int] = None
    proxyUser: Optional[str] = None
    proxyPassword: Optional[str] = None
    basicAuthUser: Optional[str] = None
    basicAuthPassword: Optional[str] = None
    keyStore: Optional[str] = None
    keyStorePassword: Optional[str] = None
    trustStore: Optional[str] = None
    trustStorePassword: Optional[str] = None

    model_config = {"extra": "allow"}

class FinalConfigSchemaItem(BaseModel):
    label: str
    name: str
    type: str
    rows: Optional[int] = None
    inputType: Optional[str] = None
    defaultValue: Optional[Any] = None
    options: Optional[List[str]] = None

class FinalNodeData(BaseModel):
    inputs: List[str] #= Field(default_factory=list)
    outputs: List[str]#= Field(default_factory=list)
    nodeName: str
    title: str
    configData: FinalNodeConfigData
    legacyType: str
    legacyName: str
    unavailable: bool = False
    twoColumn: bool = True
    nodeType: str
    configSchema: Optional[List[FinalConfigSchemaItem]] = None
    action: str

class FinalNodeStyle(BaseModel):
    width: int = 250

class FinalNode(BaseModel):
    id: str
    type: Literal["custom"] = "custom"
    initialized: bool = False
    position: FinalPosition
    data: FinalNodeData
    style: FinalNodeStyle
    group: str

class FinalEdge(BaseModel):
    id: str
    type: Literal["custom"] = "custom"
    source: str
    target: str
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    label: str = ""
    # Coordinates are required for linear layout strategy
    sourceX: float
    sourceY: float
    targetX: float
    targetY: float

class FinalViewport(BaseModel):
    x: float
    y: float
    zoom: float

class FinalFlow(BaseModel):
    task: Optional[str] = None
    nodes: List[FinalNode]
    edges: List[FinalEdge]
    position: List[float] = Field(default_factory=lambda: [0.0, 0.0])
    zoom: float = 1.0
    viewport: FinalViewport = Field(default_factory=lambda: FinalViewport(x=0.0, y=0.0, zoom=1.0))
    


# --- Simple Input Models (for LLM Generation) ---

class NodeData(BaseModel):
    inputs: List[str] = Field(..., description="List of incoming socket names")
    outputs: List[str] = Field(..., description="List of outgoing socket names")
    nodeName: str = Field(..., description="For logic: 'logic#<id>'. For others: the YAML key (e.g., 'Core:Log')")
    action: str = Field(..., description="Action to be performed by the node")
    configData: Dict[str, Any] = Field(default_factory=dict, description="Configuration values from YAML")

class FlowNode(BaseModel):
    id: str = Field(..., description="Unique string ID (e.g., 'n1')")
    data: NodeData
    group: str = Field(..., description="Node group (e.g., 'Basic', 'System')")
    

class FlowEdge(BaseModel):
    source: str = Field(..., description="Source node id")
    target: str = Field(..., description="Target node id")
    sourceHandle: str = Field(..., description="e.g., 'out#undefinedsource'")
    targetHandle: str = Field(..., description="e.g., 'inp#undefinedtarget'")
    # execution order...

class Flow(BaseModel):
    nodes: List[FlowNode]
    edges: List[FlowEdge]