# The Agent
import yaml
import json
import uuid
from agno.agent import Agent #, RunResponse
from agno.models.ollama import Ollama
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
1. **Node Selection**: Use ONLY the provided nodes in : Available Node Definitions YAML
    *   'Start' & 'End' nodes are MANDATORY for every flow.
   
2.  **Strict Schema Compliance**: The output must validly parse into the `Flow` Pydantic model.
3.  **Node IDs**: Generate 4 digit unique IDs for nodes and then prefixed with 'dndnode_'(e.g.: dndnode_<1111>).
4.  **Logical Flow**: Ensure the nodes are connected in a logical order described by the user (or implied).
5.  **Start & End**: Most flows should have a Start and End node unless specified otherwise.
6.  **Inputs & Outputs under each Node's data key**:
    *   You MUST populate `node.data.inputs` and `node.data.outputs` for EVERY node. These are the incoming and outgoing SocketNames of that node.
    *   You can find these from the `inputs` and `outputs` sections of the "Available Node Definitions" YAML and put them into lists.    
    *   This field is REQUIRED. You MUST NOT omit it.
    *   If a node truly has no inputs or outputs in its definition, ONLY THEN use `[]`.
    *   If the node is "Logic" node, then pay a close attention on the configurable additional sockets. and update them accordingly here. 

7.  **Edge Handles**: For EVERY edge, you MUST populate `sourceHandle` and `targetHandle` using the format `{{SocketName}}#undefined{{source|target}}`.
    *   Identify the socket names you are connecting.
    *   Example: Start Node (`_Data` output) -> Log Node (`_In` input).
        - `sourceHandle` = `_Data#undefinedsource`
        - `targetHandle` = `_In#undefinedtarget`
    *   Ensure the SocketName used exists in the corresponding node's definitions.

8. **Parallel Execution Order**:
    *   If a single node has MULTIPLE outgoing edges (parallel execution), and <<<"if there is a need to define an execution order for parallel execution">>> then you MUST define an execution order.
    *   Add a `data` field to the edge: `"data": {{"order": 1}}`, `"data": {{"order": 2}}`, etc.
    *   Example: Node A connects to Node B and Node C.
        - Edge A->B: `"data": {{"order": 1}}`
        - Edge A->C: `"data": {{"order": 2}}`
    *   This is CRITICAL for parallel flows.

9. **Code (Python or Groovy) in Logic Node**: 
    *   If logic needs to be writte in python code, then use `in_sockets[<name_of_input_socket>]` to access the input socket and `out_sockets[<name_of_output_socket>]` to access the output socket.
    *   But If logic needs to be writte in groovy code, then directly use name_of_input_sockets and name_of_output_sockets to access the input and output sockets.
    *   Example python code: `#python\nout_sockets["out"]="hi "+in_sockets["inp"]`
    *   Example groovy code: `out = "hi "+inp`
    *   if you find input socket as a json object, its actually a dict object in python. So you can access the fields of the dict object using the dot notation directly.

10. **configuration/creation of Input Parameters**:
    *   When in the task, user requests inputs that are provided at flow run time (e.g. taking input from user, user-provided config, API keys, or values to use in Logic), you MUST add an `inputParameters` object (parallel to `nodes` and `edges`).
    *   Example inputParameters object:
    `"inputParameters": {{"input_string_1": {{"name": "input_string_1", "inputType": "string", "inputRequired": false, "value": "hello from user"}}}}`
    *   The key of the input parameter is the name of the input parameter. (e.g. `input_string_1`)
    *   The value of the input parameter is a dictionary with the following keys: `name`, `inputType`, `inputRequired`, `value`.
    *   The `name` key is the name of the input parameter. (e.g. `input_string_1`)
    *   The `inputType` key is the type of the input parameter. (e.g. `"string"`, `"int"`, `"float"`, `"boolean"`)
    *   The `inputRequired` key is a boolean value indicating if the input parameter is required. (e.g. `false`)
    *   The `value` key is the default value of the input parameter. (e.g. `"hello from user"`)

11. **Referring to the configured Input Parameters in Nodes**:
    *   **Important note**: if you have created an input parameter, then you need to refer it in the nodes using the name of the input parameter. 
    *   This is differnt then the input sockets of the nodes. Input sockets are the sockets that are already defined in the nodes. Input parameters are the parameters that are created by you.
    **Rules to refer the input parameter in the nodes**:
        *   In nodes like Logic node code, these created input parameters can be referred using the key of the input parameter.
        *   to refer a input parameter you created in groovy code via: `flowStore['inputParameters']['<input_parameter_name>'].value` (e.g. `flowStore['inputParameters']['input_string_1'].value`).
        *   to refer a input parameter you created in python code via: `env['flowStore']['inputParameters']['<input_parameter_name>']['value']` (e.g. `env['flowStore']['inputParameters']['input_string_1']['value']`).
        *   Example python code: `#python\nout_sockets["out"]="hi "+env['flowStore']['inputParameters']['input_string_1']['value']`
        *   Example groovy code: `out = "hi "+flowStore['inputParameters']['input_string_1'].value`


"""

# Initialize Agent
# Using the specific Ollama model requested by the user
# which_model = "qwen3-coder-next-128k-custom:latest"
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

def _normalize_config_value(v):
    """Return None for empty or empty-JSON-string values; otherwise return the value."""
    if v == "" or v == "{}":
        return None
    return v


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
        
        # Normalize configData: empty string "" and "{}" -> null
        raw_data = node.get("data", {})
        config_data_raw = raw_data.get("configData")
        if isinstance(config_data_raw, dict):
            config_data_normalized = {k: _normalize_config_value(v) for k, v in config_data_raw.items()}
        else:
            config_data_normalized = {}
        
        # Construct the enriched node
        new_node = {
            "id": new_id,
            "type": node_type,
            "initialized": False,
            "position": {"x": 0, "y": 0},
            "data": {
                **raw_data,
                "configData": config_data_normalized,
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
        
    # 3. Enrich inputParameters: add id (UUID) to each param for FinalFlow schema; keep name, inputType, inputRequired, value
    raw_input_params = flow_dict.get("inputParameters") or {}
    input_params = {}
    for key, param in raw_input_params.items():
        if isinstance(param, dict):
            input_params[key] = {
                "id": str(uuid.uuid4()),
                "name": param.get("name", key),
                "inputType": param.get("inputType", "string"),
                "inputRequired": bool(param.get("inputRequired", False)),
                "value": param.get("value"),
            }
        else:
            input_params[key] = param

    # 4. Construct Final Dict
    final_flow = {
        "nodes": new_nodes,
        "edges": new_edges,
        "inputParameters": input_params,
        "position": [0, 0],
        "zoom": 1.0,
        "viewport": {
            "x": 0.0,
            "y": 0.0,
            "zoom": 1.0
        }
    }
        
    return final_flow

if __name__ == "__main__":
    import sys
    # task = "Create a simple flow that starts, logs 'Hello World', checks if a variable 'x' is greater than 10, and ends."
    
    # task= "create a flow for logging 'hello world' ."

    # task = """Create a flow that fetches a joke from the Chuck Norris API 
    # and then extract only the joke from the response and then log this joke."""

    # task = """  Create a flow that fetches a joke from the Chuck Norris API (REST interface), 
    #             then log the whole json response. 
    #             then simply write the logic to only get the joke from the previous log output (use .value) 
    #             and log this too again."""
    
    # task = "Create a flow that fetches a joke from the Chuck Norris API (REST interface) and then write the logic to only get the joke from the response json (by .value) . and at the end log that joke."
    # task = """  Create a flow that fetches a 2 jokes from the Chuck Norris API (2 different API calls in parallel), 
    #             and then simply write a single logic to concatinate those two jokes (use .value to get only the joke from the response json) . 
    #             at the end log this final concatinated joke."""

    # task = """  Create a flow that fetches a 2 jokes from the Chuck Norris API (2 different API calls in parallel), 
    #             then separately log the whole json response of both the API calls. 
    #             and then simply write a single logic to concatinate those two jokes (use .value to get only the joke from the response json) . 
    #             at the end log this final concatinated joke."""

    # task = """Create a flow that fetches a joke from the Chuck Norris API (REST interface) 
    # and then simply write the logic to get the length of the joke from the response json(use .value to get only the joke from the response json). 
    # and then if the length is greater than 100, then log 'Joke is too long' otherwise log 'Joke is short'.""" 

    # task = """Create a flow that fetches a joke from the Chuck Norris API. 
    # Use a Logic node to get the joke string from the response (use .value). 
    # Then use an If node: if the length of the joke is greater than 80, log 'Long joke' and send to End; 
    # otherwise log 'Short joke' and send to End. Both branches must reach the End node."""

    # task=""" 
    # create a flow that fatches a joke from the chuck norris api 
    # and then consider the joke string as an input for computing md5 hash of that joke 
    # and also log that joke.
    # then compute length of that hashed value , and log this too.
    # then at the end log last 5 characters of this hashed value.
    # """ 

    task="""
    create a flow that fatches a joke from the chuck norris api 
    then use this joke as input for computing md5  
    log that joke.
    compute length of that hashed value , and log this.
    then log last 5 characters of hashed value.
    """
#--------------------------------------------------------------------------------------------------

    task = """Create a flow that GETs https://api.github.com/repos/microsoft/vscode. 
    In a Logic node extract stargazers_count. Log that number. 
    Then an If node: if stargazers_count > 100000 log 'Very popular' and go to End; 
    else another If: if stargazers_count > 50000 log 'Popular' and go to End, 
    else log 'Moderate' and go to End. 
    All three branches must reach the same End. 
    Use _True and _False handles correctly."""

    task = """Create a flow that GETs https://api.github.com/repos/reclosedev/pyautocad. 
    extract stargazers_count. Log that number. 
    if stargazers_count > 100000 log 'Very popular', 
    if stargazers_count > 50000 log 'Popular', 
    else log 'Moderate'. """

    task = """Create a flow that fetches in parallel: https://jsonplaceholder.typicode.com/posts/1 and https://jsonplaceholder.typicode.com/posts/2 (order 1 and 2). 
    One Logic node with two inputs (p1, p2) that extracts title from each and returns the combined string 'Post1: <title1> | Post2: <title2>'. 
    Log that string. 
    Then an If node: if the combined string length > 50 log 'Long titles' else log 'Short titles'. Both to End."""

    task = """Create a flow that GETs https://api.github.com/repos/python/cpython. 
    In Logic extract : full_name, stargazers_count, and default_branch. 
    Format one string 'Repo: <full_name>, Stars: <stargazers_count>, Branch: <default_branch>' and output it. 
    Log that string. 
    Then in a second Logic compute the length of that string; If length > 40 log 'Long summary' else log 'Short summary'. Both paths to End."""

    task = """"Create a flow that takes a user provided input string and then calculate its md5 and log this."""  


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
            filename = f"generated_flow_{which_model.split(':')[0]}_{timestamp}.json"
            
            with open(filename, "w") as f:
                json.dump(final_json_dict, f, indent=3)
            print(f"Saved to {filename}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        with open("error.log", "w") as f:
            f.write(str(e))
        print(f"An error occurred: {e}")
