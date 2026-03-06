from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field
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
    message: Optional[str] = None
    expression: Optional[str] = None
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
    inputs: List[str]
    outputs: List[str]
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
    type: str = "custom"
    initialized: bool = False
    position: FinalPosition
    data: FinalNodeData
    style: FinalNodeStyle
    group: str

class FinalEdge(BaseModel):
    id: str
    type: str = "custom"
    source: str
    target: str
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    label: str = ""
    sourceX: float = 0.0
    sourceY: float = 0.0
    targetX: float = 0.0
    targetY: float = 0.0

class FinalViewport(BaseModel):
    x: float
    y: float
    zoom: float


class InputParameterDef(BaseModel):
    """Definition of a flow-level input parameter (id added in post_process with UUID). Refer in groovy: flowStore['inputParameters']['<name>'].value; in python: env['flowStore']['inputParameters']['<name>']['value']."""
    id: str = Field(..., description="UUID assigned in post_process_flow")
    name: str = Field(..., description="Parameter key/name (e.g. 'input_string_1')")
    inputType: str = Field(..., description="Type: 'string', 'int', 'float', 'boolean'")
    inputRequired: bool = Field(default=False, description="Whether the parameter is required")
    value: Any = Field(..., description="Default/current value")


class FinalFlow(BaseModel):
    task: Optional[str] = None
    nodes: List[FinalNode]
    edges: List[FinalEdge]
    inputParameters: Optional[Dict[str, InputParameterDef]] = Field(default_factory=dict)
    position: List[float] = Field(default_factory=lambda: [0.0, 0.0])
    zoom: float = 1.0
    viewport: FinalViewport = Field(default_factory=lambda: FinalViewport(x=0.0, y=0.0, zoom=1.0))
    


# --- Simple Input Models (for LLM Generation) ---

class NodeData(BaseModel):
    inputs: List[str] = Field(..., description="List of incoming socket names")
    outputs: List[str] = Field(..., description="List of outgoing socket names")
    nodeName: str = Field(..., description="For logic: 'logic#<id>'. For others: the YAML key (e.g., 'Core:Log')")
    action: str = Field(..., description="Action to be performed by the node, use the 'action' value from the node's definition in Available Node Definitions YAML (e.g. log, httpRequest, executeIf, etc.).")
    configData: Dict[str, Any] = Field(default_factory=dict, description="Configuration values from YAML")

class FlowNode(BaseModel):
    id: str = Field(..., description="Unique string ID (e.g., 'n1')")
    data: NodeData
    group: str = Field(..., description="Group for this node; use the 'group' value from the node's definition in Available Node Definitions YAML (e.g. Basic, System, Outbound, Logic).")

class FlowEdge(BaseModel):
    source: str = Field(..., description="Source node id")
    target: str = Field(..., description="Target node id")
    sourceHandle: str = Field(..., description="e.g., 'out#undefinedsource'")
    targetHandle: str = Field(..., description="e.g., 'inp#undefinedtarget'")
    data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Edge metadata, e.g. {'order': 1} for execution order")


class FlowInputParameterDef(BaseModel):
    """LLM output: flow-level input parameter. Keys: name, inputType, inputRequired, value. id is added in post_process with UUID."""
    name: str = Field(..., description="Parameter key/name (e.g. 'input_string_1')")
    inputType: str = Field(..., description="Type: 'string', 'int', 'float', 'boolean'")
    inputRequired: bool = Field(default=False, description="Whether the parameter is required")
    value: Optional[Union[str, int, float, bool]] = Field(
        default=None,
        description="Default value (e.g. 'hello from user')"
    )


class Flow(BaseModel):
    nodes: List[FlowNode]
    edges: List[FlowEdge]
    inputParameters: Optional[Dict[str, FlowInputParameterDef]] = Field(
        default_factory=dict,
        description="Flow-level input parameters. Key = parameter name; value = {name, inputType, inputRequired, value}. Refer in groovy: flowStore['inputParameters']['<name>'].value; in python: env['flowStore']['inputParameters']['<name>']['value']."
    )

