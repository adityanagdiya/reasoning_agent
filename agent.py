# The Agent
import yaml
import json
import re
import uuid
from agno.agent import Agent #, RunResponse
from agno.models.ollama import Ollama
from schemas import Flow, FinalFlow

# ---------------------------------------------------------------------------
# Dynamic node definitions from JSON (with optional fallback for Start/End/Logic)
# ---------------------------------------------------------------------------

def load_dynamic_node_definitions(json_path: str) -> dict:
    """
    Load node definitions from the dynamic JSON.
    JSON is expected to have key 'Nodes' (array of objects with 'content' containing YAML).
    Each node in the content YAML may include group, action, inputs, outputs, config,
    and description (used for relevance selection and passed through to the flow generator).
    Returns a flat dict: node_name -> node_definition (for building YAML).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    nodes_array = data.get("Nodes", data.get("nodes", []))
    merged = {}
    for item in nodes_array:
        content = item.get("content") or item.get("Content") or ""
        if not content or not content.strip():
            continue
        try:
            parsed = yaml.safe_load(content)
            if not parsed:
                continue
            # Content YAML has top-level key "Nodes" with dict of node_name -> def
            inner = parsed.get("Nodes", parsed.get("nodes", {}))
            if isinstance(inner, dict):
                for name, defn in inner.items():
                    if name and isinstance(defn, dict):
                        merged[str(name)] = defn
        except yaml.YAMLError:
            continue
    return merged


def load_basic_nodes(yaml_path: str) -> dict:
    """Load only the Nodes section from a YAML file (e.g. Start, End, Logic)."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        parsed = yaml.safe_load(f)
    return parsed.get("Nodes", parsed.get("nodes", {})) or {}


def merge_basic_into_dynamic(all_nodes: dict, basic_yaml_path: str) -> dict:
    """
    Ensure Start, End, and Logic exist in the node catalog (from static YAML if missing).
    """
    basic = load_basic_nodes(basic_yaml_path)
    for key in ("Start", "End", "Logic"):
        if key not in all_nodes and key in basic:
            all_nodes[key] = basic[key]
    return all_nodes


def _select_relevant_nodes_heuristic(task: str, all_nodes: dict) -> list[str]:
    """Keyword-based fallback when LLM selection is not used or fails."""
    mandatory = ["Start", "End", "Logic"]
    task_lower = task.lower()
    selected = [n for n in mandatory if n in all_nodes]
    available = [n for n in all_nodes if n not in mandatory]
    # Simple keyword hints for common nodes
    keywords = {
        "Core:Log": ["log", "logging", "print", "output"],
        "Core:HTTPRequest": ["http", "api", "fetch", "get", "post", "request", "rest", "url"],
        "Core:If": ["if", "condition", "branch", "check", "greater", "less", "compare"],
        "Core:Repeat": ["loop", "repeat", "times", "n times", "iterat"],
        "Core:Expression": ["expression", "eval"],
        "Core:ApplyExpression": ["apply", "expression"],
        "Core:Join": ["join", "merge"],
        "Core:MD5": ["md5", "hash"],
        "Core:Template": ["template"],
        "Core:SendEmail": ["email", "mail"],
        "Core:JsonResponse": ["json", "response"],
    }
    for name in available:
        for kw, terms in keywords.items():
            if name == kw and any(t in task_lower for t in terms):
                if name not in selected:
                    selected.append(name)
                break
    # If no specific match, add Logic and a few common ones for generic tasks
    if "Logic" not in selected and "logic" in task_lower:
        selected.append("Logic")
    if "Core:Log" not in selected:
        selected.append("Core:Log")
    if "Core:HTTPRequest" not in selected and any(x in task_lower for x in ["http", "api", "fetch"]):
        selected.append("Core:HTTPRequest")
    if "Core:If" not in selected and any(x in task_lower for x in ["if", "condition", "branch"]):
        selected.append("Core:If")
    if "Core:Repeat" not in selected and any(x in task_lower for x in ["loop", "repeat", " times"]):
        selected.append("Core:Repeat")
    return selected[:30]


def select_relevant_node_names(task: str, all_nodes: dict, model) -> list[str]:
    """
    Select which node names are relevant for the given task.
    Always includes Start, End, and Logic if present in all_nodes.
    Uses the provided model for a short selection call when possible.
    """
    mandatory = ["Start", "End", "Logic"]
    available = [n for n in all_nodes if n not in mandatory]
    if not available:
        return [n for n in mandatory if n in all_nodes]
    # print("*********available*********",available)

    # Build a short hint per node (group, action) for better selection
    hints = []
    for name in available:
        defn = all_nodes.get(name, {})
        group = defn.get("group", "")
        action = defn.get("action", "")
        desc = defn.get("description", "")
        hint = name
        # print("*********name*********",name)
        if group or action:
            hint += f" (group={group}, action={action})"
        if isinstance(desc, str) and desc.strip():
            # Use first line or first 200 chars of description for relevance selection
            first_line = desc.strip().split("\n")[0].strip()
            hint += " " + (first_line[:200] + "..." if len(first_line) > 200 else first_line)
        hints.append(hint)
        # print("*********hint*********",hint)
    

    prompt = f"""You are a flow design assistant. Given the task below and the list of available nodes, output ONLY a comma-separated list of node names that are needed to implement this task. Do not include any explanation.

Rules:
- Always include Start and End.
- Always include Logic if it appears in the catalog: Logic is the node for arbitrary Python/Groovy. Use it whenever the task needs behavior that is not a single built-in node.
- Include Logic especially when the user asks for "logic node", "custom code", "in Python", "transform", "filter", "analyze", or "write the whole code".
- Include Core:Repeat when the task says "loop", "repeat", "N times", "do this 10 times", or "n times (user provided)".
- Include Core:If when the task has conditional branching (if/else, "if X then ... otherwise ...", "greater than", etc.).
- Include only other nodes that are clearly relevant (e.g. HTTP request, Log, If, etc.).
- Do NOT include Core:JSON2MAP when the task involves HTTP/API request and then processing the response in Logic or custom code. In that case the HTTP response is already provided as a dict/map to Logic—parsing JSON is unnecessary.


Task:
{task[:2000]}

Available nodes (name and optional hint):
{chr(10).join(hints)}

Reply with only the comma-separated node names, nothing else."""

    try:
        selector_agent = Agent(model=model, instructions="Reply with only the requested list, no explanation.", debug_mode=True)
        response = selector_agent.run(prompt)
        text = (response.content if hasattr(response, "content") else str(response)).strip()
        # Parse comma-separated names; allow newlines and extra spaces
        names = [n.strip().strip("'\"") for n in re.split(r"[,;\n]+", text) if n.strip()]
        seen = set()
        selected = []
        for n in names:
            if n in all_nodes and n not in seen:
                seen.add(n)
                selected.append(n)
        for m in mandatory:
            if m in all_nodes and m not in seen:
                if m == "Start":
                    selected.insert(0, m)
                else:
                    selected.append(m)
        # When both HTTP and Logic are selected, exclude Core:JSON2MAP—HTTP response is already a dict for Logic
        if "Core:HTTPRequest" in selected and "Logic" in selected and "Core:JSON2MAP" in selected:
            selected = [n for n in selected if n != "Core:JSON2MAP"]
        return selected if selected else _select_relevant_nodes_heuristic(task, all_nodes)
    except Exception:
        return _select_relevant_nodes_heuristic(task, all_nodes)


def build_node_definitions_yaml(all_nodes: dict, selected_names: list[str]) -> str:
    """Build a single YAML string containing only the selected node definitions."""
    subset = {k: all_nodes[k] for k in selected_names if k in all_nodes}
    if not subset:
        return ""
    doc = {"Nodes": subset}
    return yaml.dump(doc, default_flow_style=False, sort_keys=False, allow_unicode=True)


def get_system_prompt(node_definitions_yaml: str) -> str:
    """Build the full system prompt with the given node definitions YAML."""
    return f"""
You are an expert Flow Generator for a low-code platform. Your task is to create a valid JSON structure for a flow based on the user's request.
You must return the data strictly complying with the provided Pydantic schema for `Flow`.


### Available Node Definitions YAML
Use these definitions to construct the `data` and `configData` for each node correctly.
{node_definitions_yaml}


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
    *   If the node is "Logic", set `configData.incomingSockets` and `configData.outgoingSockets` to match the script; mirror those names in `data.inputs` / `data.outputs` and edges (see Rule 10). 
7.  **configData.label for every node**: Every node definition's `config` includes an item with `name: label` (display "Label"). You MUST set `node.data.configData.label` to a short, human-readable label for that node instance (e.g. "Start", "Log Joke", "HTTP Get Repo"). This field is REQUIRED for every node—do not omit it.

8.  **Edge Handles**: For EVERY edge, you MUST populate `sourceHandle` and `targetHandle` using the format `{{SocketName}}#undefined{{source|target}}`.
    *   Identify the socket names you are connecting.
    *   Example: Start Node (`_Data` output) -> Log Node (`_In` input).
        - `sourceHandle` = `_Data#undefinedsource`
        - `targetHandle` = `_In#undefinedtarget`
    *   Ensure the SocketName used exists in the corresponding node's definitions.

8b. **Fan-out from one output (logging vs further operations)**: A single node's output socket MAY have MULTIPLE outgoing edges (two or more branches). Use this when the task requires BOTH "log X" AND "use X for further operations":
    *   **Correct**: From the same output socket (e.g. `joke` of "Extract Joke"), create TWO edges: one to the Log node (for logging only) and one to the next processing node (e.g. "Compute Length"). So: Extract Joke --[joke]--> Log Joke, and Extract Joke --[joke]--> Compute Length. Both edges share the same source and sourceHandle; targets differ.
    *   **Wrong**: Do NOT put the Log node in the middle of the pipeline (e.g. Extract Joke -> Log Joke -> Compute Length). That would make "Compute Length" receive the Log node's output instead of the original data. Log is for side-effect (logging) only; the "further operations" branch must receive data directly from the node that produced it.
    *   Summary: one out_socket can have 2 (or more) branches—one for logging, one (or more) for further operations.

8c. **Conditional branching (Core:If) — mandatory both branches**:
    *   Core:If has exactly ONE input `_Data` (the value to evaluate, e.g. length) and TWO outputs `_True` and `_False`. Use the socket names exactly as in the node definition.
    *   Connect the value to test to If._Data only (e.g. Compute Length.len → If._Data). Do NOT add a second "control flow" or "_In" input to the If node; it only has _Data.
    *   Connect If._True to the "true" branch node (e.g. "Log Too Long") and If._False to the "false" branch node (e.g. "Log Short"). You MUST create edges for BOTH _True and _False; do not skip either branch.
    *   Set `configData.expression` to a Groovy expression that evaluates to boolean (e.g. `_Data > 100` for length > 100).
    *   When the If is inside a loop: BOTH branch nodes (Log Too Long and Log Short) must have their output connected back to the Repeat node's _SubflowResult_ input (see Rule 8d). If only one branch connects back, the loop will run the wrong number of times or hang.

8d. **Loop (Core:Repeat) — "do N times" / "in a loop"**:
    *   When the task says "do this N times", "repeat 10 times", or "in a loop" (with N fixed or from an input parameter), use Core:Repeat. Its inputs are _Data and _SubflowResult_; outputs are _Result and _Subflow.
    *   Wire: Start._Data → Repeat._Data. Set configData.count to the number (e.g. 10) or the name of an input parameter if "n is user provided".
    *   Loop body entry: Connect Repeat._Subflow → the first node of the loop body (e.g. Fetch Joke._Data). Everything that must run each iteration (fetch, extract, log, compute, If, Log Too Long / Log Short) is inside the loop.
    *   **Critical — ONLY the end of each iteration path connects to _SubflowResult_**: Repeat._SubflowResult_ must receive exactly ONE signal per iteration. 
    So ONLY the terminal nodes of the loop body (the very last step on each path) may have an edge to Repeat._SubflowResult_. 
    When the body has an If with two branches, ONLY the two branch endpoints connect back: Log Too Long._Out → Repeat._SubflowResult_ AND Log Short._Out → Repeat._SubflowResult_. 
    No other node in the loop (e.g. Log Joke, Log Length, Compute Length, If/Length Check, Fetch Joke, Extract Joke) must connect to Repeat._SubflowResult_. 
    If Log Joke, Log Length, or any intermediate node also connects to _SubflowResult_, the Repeat node will count each of those as an extra iteration and the loop will run 2n or 3n times instead of n.
    *   When the loop is done: Connect Repeat._Result → End._Data. So the flow is: Start → Repeat → [loop body: ... → If → (Log Too Long | Log Short) → only these two back to Repeat] → after N iterations Repeat._Result → End.

9. **Parallel Execution Order**:
    *   If a single node has MULTIPLE outgoing edges (parallel execution), and <<<"if there is a need to define an execution order for parallel execution">>> then you MUST define an execution order.
    *   Add a `data` field to the edge: `"data": {{"order": 1}}`, `"data": {{"order": 2}}`, etc.
    *   Example: Node A connects to Node B and Node C.
        - Edge A->B: `"data": {{"order": 1}}`
        - Edge A->C: `"data": {{"order": 2}}`
    *   This is CRITICAL for parallel flows.

10. **Logic node — generalized (read carefully; most failures are here)**:
    **Role**: Logic runs arbitrary scripts. Tasks differ widely (API shaping, scoring, multi-step transforms). 
    **Flow shape**:
    *   **Thin flow, fat Logic** is valid: Start → (optional HTTP / inputs) → **one Logic node with complete Python** → Log / End when most of the work is custom computation.
    *   Use **multiple** Logic nodes only when the graph must branch or when distinct stages feed different downstream nodes—not to avoid writing a longer script.
    **Socket contract (mandatory)**:
    *   Set `configData.incomingSockets` to a comma-separated list of every input socket name the script reads (e.g. `response,payload`). Set `configData.outgoingSockets` for every output the script writes (e.g. `result,safe`). Match `node.data.inputs` / `node.data.outputs` lists and all edges to these names.
    *   Python: first line of `configData.logic` must be `#python`. Use only `in_sockets["<name>"]` and `out_sockets["<name>"]` for those sockets.
    *   Groovy: use the socket variable names directly for inputs/outputs (no `in_sockets` dict).
    *   Example python (minimal): `#python\nout_sockets["out"]="hi "+str(in_sockets["inp"])`
    *   Example groovy (minimal): `out = "hi "+inp`
    **Large / end-to-end scripts**:
    *   The `logic` string may be long (helpers, try/except, full pipelines). That is expected. must include **complete, runnable** logic—not comments like "placeholder for NLP".
    **No need to parsejson**:
    *   JSON from HTTP (or similar nodes) is already automaticaly converted to a Python **dict** or map internally. 
    *   So Do NOT use `json.loads()`** on it. Use `obj["key"]` or safe `.get()`. 
    *   and also do not use Core:JSON2MAP node before this logic node in this flow as well.

**Runtime config**: use `inputParameters` and in Python `env['flowStore']['inputParameters']['<name>']['value']` when the user asks for user-provided keys or config (see rules 11–12).

11. **Configuration / Creation of Input Parameters**:
    *   When a task requires values that are provided at flow run time (for example: user input, configuration values, API keys, or values used in logic), you must create an inputParameters object.
    *   The inputParameters object should be defined at the same level as nodes and edges.
    *   The key of each parameter must be the name of the input parameter.
    Example:
    *   "inputParameters": {{
           "input_string_1": {{
            "name": "input_string_1",
            "inputType": "string",
            "inputRequired": false,
            "value": "hello from user"
            }}
        }}
    *   Each parameter value must contain the following fields:
        *   name: Name of the input parameter	example: "input_string_1"
        *   inputType: Data type of the input	example: "string", "int", "float", "boolean", "file"
        *   inputRequired: Indicates whether the parameter is required	example: true / false
        *   value: Default value	example: "hello from user"

12. **Using Created Input Parameters in Nodes**:
    *   Important:If you create an input parameter, it must be used somewhere in the nodes.
    *   Also note:
        *   Input parameters are different from node input sockets.
        *   Input sockets are predefined in nodes.
        *   Input parameters are custom parameters created by you.
    
    **Using Input Parameters in Logic Nodes**:
    *   In Groovy
        *   Access the value using: `flowStore['inputParameters']['<input_parameter_name>'].value`
        *   Example: `out = "hi " + flowStore['inputParameters']['input_string_1'].value`
    *   In Python
        *   Access the value using: `env['flowStore']['inputParameters']['<input_parameter_name>']['value']`
        *   Example: `out = "hi " + env['flowStore']['inputParameters']['input_string_1']['value']`
        
    **Using Input Parameters in Node as an Expression**:
    *   Case 1 — When in the node it's config key label is $Expression
        *   directly use the value of the input parameter: `flowStore['inputParameters']['<input_parameter_name>'].value`
        *   Example:using in node like Core:ApplyExpression for $Expression use : `flowStore['inputParameters']['input_string_1'].value`

    *   Case 2 — In all other nodes, it's config key label is not $Expression
        *   add prefix `$e:` to the value of the input parameter: `$e:flowStore['inputParameters']['<input_parameter_name>'].value`
        *   Example: using in node like Core:Repeat for count, use : `$e:flowStore['inputParameters']['number_of_times'].value`

"""


def create_agent_for_task(task: str, nodes_json_path: str = "all_nodes_dynamically_coming.json", basic_yaml_path: str = "basic_nodes_definition.yaml", which_model: str = "gpt-oss-120b-long-context:latest"):
    """
    Load dynamic nodes, select those relevant to the task, build prompt, and return an Agent.
    """
    all_nodes = load_dynamic_node_definitions(nodes_json_path)
    # print("*********all_nodes*********",all_nodes)
    all_nodes = merge_basic_into_dynamic(all_nodes, basic_yaml_path)
    if not all_nodes:
        raise ValueError("No node definitions found. Check paths for JSON and basic YAML.")

    # Use same Ollama for selection (lighter call)
    ollama = Ollama(id=which_model, host="http://100.113.113.188:2802", options={"temperature": 0.0})
    selected_names = select_relevant_node_names(task, all_nodes, ollama)
    print("*********selected_names*********", selected_names)
    node_yaml = build_node_definitions_yaml(all_nodes, selected_names)
    print("*********node_yaml*********",node_yaml)
    system_prompt = get_system_prompt(node_yaml)

    return Agent(
        model=ollama,
        description="Agent for generating flow JSON",
        instructions=system_prompt,
        output_schema=Flow,
        structured_outputs=True,
        debug_mode=True,
    )

# Model and paths (can be overridden when calling create_agent_for_task)
which_model = "gpt-oss-120b-long-context:latest"#"gpt-oss-120b-long-context:latest" #glm-4.7-flash-long-context:latest
nodes_json_path = "all_nodes_dynamically_coming.json"
basic_yaml_path = "basic_nodes_definition.yaml"

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
        # Ensure label is always present (from node config); use title/nodeName if missing
        if config_data_normalized.get("label") is None or config_data_normalized.get("label") == "":
            config_data_normalized["label"] = title or node_name_raw or "Node"
        
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
    #  
    # task="""
    # create a flow that fetches a joke from the chuck norris api 
    # then log only that joke.
    # """

    # task="""
    # create a flow that fetches a joke from the chuck norris api 
    # then use this joke as input for computing md5  
    # log that joke.
    # compute length of that hashed value , and log this.
    # then log last 5 characters of hashed value.
    # """
#--------------------------------------------------------------------------------------------------

    # task = """Create a flow that GETs https://api.github.com/repos/microsoft/vscode. 
    # In a Logic node extract stargazers_count. Log that number. 
    # Then an If node: if stargazers_count > 100000 log 'Very popular' and go to End; 
    # else another If: if stargazers_count > 50000 log 'Popular' and go to End, 
    # else log 'Moderate' and go to End. 
    # All three branches must reach the same End. 
    # Use _True and _False handles correctly."""

    # task = """Create a flow that GETs https://api.github.com/repos/reclosedev/pyautocad. 
    # extract stargazers_count. Log that number. 
    # if stargazers_count > 100000 log 'Very popular', 
    # if stargazers_count > 50000 log 'Popular', 
    # else log 'Moderate'. """

    # task = """Create a flow that fetches in parallel: https://jsonplaceholder.typicode.com/posts/1 and https://jsonplaceholder.typicode.com/posts/2 (order 1 and 2). 
    # One Logic node with two inputs (p1, p2) that extracts title from each and returns the combined string 'Post1: <title1> | Post2: <title2>'. 
    # Log that string. 
    # Then an If node: if the combined string length > 50 log 'Long titles' else log 'Short titles'. Both to End."""

    # task = """Create a flow that GETs https://api.github.com/repos/python/cpython. 
    # In Logic extract : full_name, stargazers_count, and default_branch. 
    # Format one string 'Repo: <full_name>, Stars: <stargazers_count>, Branch: <default_branch>' and output it. 
    # Log that string. 
    # Then in a second Logic compute the length of that string; If length > 40 log 'Long summary' else log 'Short summary'. Both paths to End."""

    # task="""create a flow that fetches a joke from chuck norris api and then logs only the joke by .value
    # if the joke content is NSFW, then log 'Joke is NSFW: <joke>' otherwise log 'Joke is not NSFW: <joke>'
    # use proper logics and nlp to check if the joke is NSFW on its content
    # """

    # task = "Create a flow that takes a user provided input string and then calculate its md5 and log this."  
    
    task="create a flow that fetches a joke from chuck norris api and then logs only the joke by .value"
    task="""create a flow that fetches a joke from chuck norris api and then logs only the joke by .value
    do this 10 times in a loop"""
    task="""create a flow that fetches a joke from chuck norris api and then logs only the joke by .value
    then compute the length of the joke and log this too. 
    do all this 10 times in a loop"""
    task="""create a flow that fetches a joke from chuck norris api and then logs only the joke by .value
    then compute the length of the joke and log this too.
    then check if the length is greater than 100, then log 'Joke is too long' otherwise log 'Joke is short'
    do all this 10 times in a loop"""
    task="""create a flow that fetches a joke from chuck norris api and then logs only the joke by .value
    then compute the length of the joke and log this too.
    then check if the length is greater than 100, then log 'Joke is too long' otherwise log 'Joke is short'
    do all this n times in a loop and this n should be a user provided input"""


    # task="""create a flow that takes an excel file as input from user,
    # in this excel the second column contains 'description' of the product,
    # you need to Classify the product description into one of these categories:
    # - Electronics & Accessories
    # - Computer & Office Supplies
    # - Home & Kitchen
    # - Sports & Fitness
    # - Clothing & Footwear
    # - Toys & Kids Products
    # - Food & Beverages
    # - Personal Care & Lifestyle
    # then log the category of the each product description in a new column
    # """




    if len(sys.argv) > 1:
        task = sys.argv[1]
    
    print(f"Generating flow for task: {task}...")
    
    try:
        agent = create_agent_for_task(task, nodes_json_path=nodes_json_path, basic_yaml_path=basic_yaml_path, which_model=which_model)
        response = agent.run(task)
        flow_data = response.content
        
        # print(f"Flow Data Type: {type(flow_data)}")
        
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
