import yaml
import json
import uuid
from agno.agent import Agent #, RunResponse
from schemas import Flow, FinalFlow

# Load Node Definitions
def load_node_definitions():
    with open("all_six_nodes_definition.yaml", "r") as f:
        return f.read()

node_definitions = load_node_definitions()


system_prompt = f"""
You are an expert Flow Generator for a low-code platform. Your task is to create a valid JSON structure for a flow based on the user's request.
You must return the data strictly complying with the provided Pydantic schema for `Flow`.


### Available Node Definitions YAML
Use these definitions to construct the `data` and `configData` for each node correctly.
{node_definitions}


### Rules
1. **Node Selection**: Use ONLY the provided nodes.
    *   'Start' & 'End' nodes are MANDATORY for every flow.
    *   Use 'Core:HTTPRequest' for API calls.
    *   Use 'Core:Log' for logging.
    *   Use 'Logic' node only if complex transformation is strictly needed.
    *   Use 'Core:If' for branching.

2.  **Strict Schema Compliance**: The output must validly parse into the `Flow` Pydantic model.
3.  **Node IDs**: Generate 4 digit unique IDs for nodes and then prefixed with 'dndnode_'(e.g.: dndnode_<1111>).
5.  **Logical Flow**: Ensure the nodes are connected in a logical order described by the user (or implied).
6.  **Start & End**: Most flows should have a Start and End node unless specified otherwise.
7.  **Inputs & Outputs under each Node's data key**:
    *   You MUST populate `node.data.inputs` and `node.data.outputs` for EVERY node. These are the incoming and outgoing SocketNames of that node.
    *   You can find these from the `inputs` and `outputs` sections of the "Available Node Definitions" YAML and put them into lists.    
    *   This field is REQUIRED. You MUST NOT omit it.
    *   If a node truly has no inputs or outputs in its definition, ONLY THEN use `[]`.
    *   If the node is "Logic" node, then pay a close attention on the configurable additional sockets. and update them accordingly here. 

8.  **Edge Handles**: For EVERY edge, you MUST populate `sourceHandle` and `targetHandle` using the format `{{SocketName}}#undefined{{source|target}}`.
    *   Identify the socket names you are connecting.
    *   Example: Start Node (`_Data` output) -> Log Node (`_In` input).
        - `sourceHandle` = `_Data#undefinedsource`
        - `targetHandle` = `_In#undefinedtarget`
    *   Ensure the SocketName used exists in the corresponding node's definitions.

9.  **Group**: Each node should have a `group` field. Use the following categories:
    *   "Basic" for Start, End, and Logic nodes.
    *   "System" for Core:Log.
    *   "Outbound" for Core:HTTPRequest.
    *   "Logic" for Core:If.
    *   Use "Basic" as a fallback if unsure.
    
10. action field: 
    *   For Start node: "" (empty string)
    *   For End node: "" (empty string)
    *   For Logic node: "" (empty string)
    *   For Core:Log node: "log"
    *   For Core:HTTPRequest node: "httpRequest"
    *   Use "" as a fallback if unsure.

11. **Parallel Execution Order**:
    *   If a single node has MULTIPLE outgoing edges (parallel execution), and <<<"if there is a need to define an execution order for parallel execution">>> then you MUST define an execution order.
    *   Add a `data` field to the edge: `"data": {{"order": 1}}`, `"data": {{"order": 2}}`, etc.
    *   Example: Node A connects to Node B and Node C.
        - Edge A->B: `"data": {{"order": 1}}`
        - Edge A->C: `"data": {{"order": 2}}`
    *   This is CRITICAL for parallel flows.

"""


from agno.models.ollama import Ollama

# Initialize Agent
# Using the specific Ollama model requested by the user

which_model = "gpt-oss-120b-long-context:latest"
# which_model = "gpt-oss-20b-long-context"

agent = Agent(
    model=Ollama(id=which_model, host="http://100.113.113.188:2802",  options = {"temperature":0.0}),
    description="Agent for generating flow JSON",
    instructions=system_prompt,
    output_schema=Flow,
    structured_outputs=True,
    debug_mode=True
)

def post_process_flow(flow_dict):
    """
    Enriches the simple Flow JSON from the agent with UI fields,
    correct UUIDs, and resets positions.
    """
    new_nodes = []
    new_edges = []
    
    # Map old temporary IDs to new UUIDs
    # e.g. "dndnode_1001" -> "dndnode_<uuid>"
    id_map = {}
    
    # 1. Process Nodes
    for node in flow_dict.get("nodes", []):
        old_id = node.get("id")
        
        # specific requirement: "replace each node id by an actual random uuid"
        # Prefix with 'node_' as seen in sample? Sample: "node_5e9bd51d..."
        new_uuid = str(uuid.uuid4())
        new_id = f"dndnode_{new_uuid}"
        id_map[old_id] = new_id
        
        node_name_raw = node.get("data", {}).get("nodeName", "")
        # Normalize types based on name
        # Sample logic inferred:
        # Start -> nodeName="start", nodeType="start", legacyType="start", title="Start"
        # End -> nodeName="end", nodeType="end", legacyType="end", title="End"
        # Logic -> nodeName="logic#<id>", nodeType="logic", legacyType="logic", title="Logic"
        # Core:HTTPRequest -> nodeType="workflow", legacyType="workflow", title="Core:HTTPRequest"
        
        node_type = "custom" # Outer type is always custom
        
        inner_node_type = "workflow" # default
        legacy_type = "workflow"
        legacy_name = node_name_raw
        title = node_name_raw
        
        lower_name = node_name_raw.lower()
        
        if "start" in lower_name:
            node.get("data", {})["nodeName"] = "start" # normalize to lowercase
            inner_node_type = "start"
            legacy_type = "start"
            legacy_name = "start"
            title = "Start"
        elif "end" in lower_name:
            node.get("data", {})["nodeName"] = "end"
            inner_node_type = "end"
            legacy_type = "end"
            legacy_name = "end"
            title = "End"
        elif "logic" in lower_name:
            # logic node name includes id in sample: logic#node_...
            node.get("data", {})["nodeName"] = f"logic#{new_id}"
            inner_node_type = "logic"
            legacy_type = "logic"
            legacy_name = "logic"
            title = "Logic"
        else:
            # Keep original name for Core nodes
            pass
        
        # Construct the enriched node
        new_node = {
            "id": new_id,
            "type": node_type,
            "initialized": False,
            "position": {"x": 0, "y": 0},
            "data": {
                **node.get("data", {}),
                "title": title,
                "legacyType": legacy_type,
                "legacyName": legacy_name,
                "unavailable": False,
                "twoColumn": True,
                "nodeType": inner_node_type
            },
            "style": {"width": 250},
            "group": node.get("group", "Basic")
        }
        
        new_nodes.append(new_node)
        
    # 2. Process Edges
    for edge in flow_dict.get("edges", []):
        old_source = edge.get("source")
        old_target = edge.get("target")
        
        new_source = id_map.get(old_source, old_source)
        new_target = id_map.get(old_target, old_target)
        
        # specific requirement: "for every edge id should be edge_<source_node_id>_<target_node_id>"
        # Note: in sample it is "vueflow__edge-node_...". 
        # But user explicitly asked for "edge_<source_node_id>_<target_node_id>"
        edge_id = f"edge_{new_source}_{new_target}"
        
        new_edge = {
            "id": edge_id,
            "type": "custom",
            "source": new_source,
            "target": new_target,
            "sourceHandle": edge.get("sourceHandle"),
            "targetHandle": edge.get("targetHandle"),
            "data": edge.get("data", {}),
            "label": edge.get("label", ""),
            "sourceX": 0,
            "sourceY": 0,
            "targetX": 0,
            "targetY": 0
        }
        new_edges.append(new_edge)
        
    # 3. Construct Final Dict
    final_flow = {
        "nodes": new_nodes,
        "edges": new_edges,
        "position": [0, 0],
        "zoom": 1,
        "viewport": {
            "x": 0,
            "y": 0,
            "zoom": 1
        }
    }
        
    return final_flow

if __name__ == "__main__":
    import sys
    # task = "Create a simple flow that starts, logs 'Hello World', checks if a variable 'x' is greater than 10, and ends."
   
    # task = "Create a flow that fetches a joke from the Chuck Norris API (REST interface) and then simply write the logic to only get the joke from the response json." 
    # task = """  Create a flow that fetches a joke from the Chuck Norris API (REST interface), 
    #             then log the whole json response. 
    #             then simply write the logic to only get the joke from the previous log output (use .value) 
    #             and log this too again."""
    
    # task = "Create a flow that fetches a joke from the Chuck Norris API (REST interface) and then write the logic to only get the joke from the response json (by .value) . and at the end log that joke."
    # task = """  Create a flow that fetches a 2 jokes from the Chuck Norris API (2 different API calls in parallel), 
    #             and then simply write a single logic to concatinate those two jokes (use .value to get only the joke from the response json) . 
    #             at the end log this final concatinated joke."""

    task = """  Create a flow that fetches a 2 jokes from the Chuck Norris API (2 different API calls in parallel), 
                then separately log the whole json response of both the API calls. 
                and then simply write a single logic to concatinate those two jokes (use .value to get only the joke from the response json) . 
                at the end log this final concatinated joke."""

    if len(sys.argv) > 1:
        task = sys.argv[1]
    
    print(f"Generating flow for task: {task}...")
    
    try:
        response = agent.run(task)
        flow_data = response.content
        
        print(f"Flow Data Type: {type(flow_data)}")
        
        base_dict = None
        
        # Check if it's a dict directly
        if isinstance(flow_data, dict):
            print("Output is a dictionary.")
            base_dict = flow_data
        # Check if it's a Pydantic model (Flow)
        elif isinstance(flow_data, Flow):
            print("Output is a Flow object.")
            if hasattr(flow_data, "model_dump"):
                base_dict = flow_data.model_dump()
            elif hasattr(flow_data, "dict"):
                 base_dict = flow_data.dict()
            else:
                print("Error: Flow object has neither model_dump nor dict method.")
        else:
             print(f"Error: Unknown output type: {type(flow_data)}")

        if base_dict:
            print("Successfully extracted flow dictionary!")
            
            # Apply Post Processing
            processed_flow_dict = post_process_flow(base_dict)
            
            # # Add task metadata
            processed_flow_dict["task"] = task

            # --- Validation Step ---
            print("Validating against FinalFlow schema...")
            try:
                final_model = FinalFlow(**processed_flow_dict)
                print("Validation SUCCESS: Post-processed JSON matches the final Pydantic schema.")
                final_json_dict = final_model.model_dump()
            except Exception as e:
                print(f"Validation FAILED: {e}")
                print("Saving unvalidated dictionary for debugging...")
                final_json_dict = processed_flow_dict

            # Save to file for inspection
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_flow_{which_model.split('-long')[0]}_{timestamp}.json"
            
            with open(filename, "w") as f:
                json.dump(final_json_dict, f, indent=3)
            print(f"Saved to {filename}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        with open("error.log", "w") as f:
            f.write(str(e))
        print(f"An error occurred: {e}")
