# import yaml
import json
from agno.agent import Agent #, RunResponse
from schemas import Flow

# Load Node Definitions
def load_node_definitions():
    with open("all_six_nodes_definition.yaml", "r") as f:
        return f.read()

node_definitions = load_node_definitions()

# Layout Strategy Instructions
layout_strategy = """
You must position the nodes in a 2D space using a Linear Layout Strategy:
- Fix all node widths to 250 (in style.width).
- Keep y = 0 for all nodes.
- Increment x starting at 0 for the first node.
- For subsequent nodes, increment x by 300 (e.g., Node 1: x=0, Node 2: x=300, Node 3: x=600, etc.) to avoid overlapping.
"""

# System Prompt
system_prompt = f"""
You are an expert Flow Generator for a low-code platform. Your task is to create a valid JSON structure for a flow based on the user's request.
You must return the data strictly complying with the provided Pydantic schema for `Flow`.


### Available Node Definitions YAML
Use these definitions to construct the `data` and `configData` for each node correctly.
{node_definitions}

### Positioning & Layout
{layout_strategy}

### Rules
1. **Node Selection**: Use ONLY the provided nodes.
    *   'Start' & 'End' nodes are MANDATORY for every flow.
    *   Use 'Core:HTTPRequest' for API calls.
    *   Use 'Core:Log' for logging.
    *   Use 'Logic' node only if complex transformation is strictly needed.
    *   Use 'Core:If' for branching.

2.  **Strict Schema Compliance**: The output must validly parse into the `Flow` Pydantic model.
3.  **Node IDs**: Generate random unique IDs for nodes, using uuid and then prefixed with 'node_'(e.g.: node_<uuid_generated_id>).
4.  **Edge IDs**: Generate valid edge IDs linking source and target nodes, (e.g., 'edge_<source_node_id>_to_<target_node_id>').
5.  **Logical Flow**: Ensure the nodes are connected in a logical order described by the user (or implied).
6.  **Start & End**: Most flows should have a Start and End node unless specified otherwise.
7.  **Config Schema**: For EVERY node, you MUST populate the `configSchema` field in `data`. You must COPY the `config` list exactly as it appears in the `Available Node Definitions` YAML for that node type. Do not omit this field.
8.  **Inputs & Outputs under each Node's data key**:
    *   You MUST populate `node.data.inputs` and `node.data.outputs` for EVERY node. These are the incoming and outgoing SocketNames of that node.
    *   You can find these from the `inputs` and `outputs` sections of the "Available Node Definitions" YAML and put them into lists.    
    *   This field is REQUIRED. You MUST NOT omit it.
    *   If a node truly has no inputs or outputs in its definition, ONLY THEN use `[]`.
    *   If the node is "Logic" node, then pay a close attention on the configurable additional sockets. and update them accordingly here. 

9.  **Edge Handles**: For EVERY edge, you MUST populate `sourceHandle` and `targetHandle` using the format `{{SocketName}}#undefined{{source|target}}`.
    *   Identify the socket names you are connecting.
    *   Example: Start Node (`_Data` output) -> Log Node (`_In` input).
        - `sourceHandle` = `_Data#undefinedsource`
        - `targetHandle` = `_In#undefinedtarget`
    *   Ensure the SocketName used exists in the corresponding node's definitions.

10. **Edge Coordinates**: For EVERY edge, you MUST calculate and populate `sourceX`, `sourceY`, `targetX`, and `targetY` based on the node positions:
    - `sourceX` = sourceNode.position.x + sourceNode.style.width (approx 250)
    - `sourceY` = sourceNode.position.y + 40 (approx half height)
    - `targetX` = targetNode.position.x
    - `targetY` = targetNode.position.y + 40
"""

# (e.g. 
#     if in YAML:
#     inputs:
#       inp: any
#     outputs:
#       out: any
#    then in JSON:
#    "inputs": ["inp"],
#    "outputs": ["out"]
#    )

from agno.models.ollama import Ollama

# Initialize Agent
# Using the specific Ollama model requested by the user

# which_model = "gpt-oss-120b-long-context"
which_model = "gpt-oss-20b-long-context"

agent = Agent(
    model=Ollama(id=which_model, host="http://100.113.113.188:2802",  options = {"temperature":0.0}),
    description="Agent for generating flow JSON",
    instructions=system_prompt,
    output_schema=Flow,
    structured_outputs=True,
    debug_mode=True
)

if __name__ == "__main__":
    import sys
    # task = "Create a simple flow that starts, logs 'Hello World', checks if a variable 'x' is greater than 10, and ends."
    task = "Create a flow that fetches a joke from the Chuck Norris API (REST interface) and then simply write the logic to only get the joke from the response json."
    # task = "Create a flow that fetches a joke from the Chuck Norris API (REST interface) and then write the logic to only get the joke from the response json. and at the end log that joke."
    # task = "Create a flow that fetches a 2 jokes from the Chuck Norris API (2 different API calls in parallel), and then simply write the logic to concatinate the only jokes from both the response jsons received. "

    if len(sys.argv) > 1:
        task = sys.argv[1]
    
    print(f"Generating flow for task: {task}...")
    
    try:
        response = agent.run(task)
        flow_data = response.content
        # Verify it's a Flow object
        if isinstance(flow_data, Flow):
            print("Successfully generated valid Flow JSON!")
            # print(json.dumps(flow_data.model_dump(), indent=3))

            # update the flow_data with the task at the top
            flow_data_dict = {"task": task, **flow_data.model_dump()}

            # Save to file for inspection
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"generated_flow_{which_model.split('-long')[0]}_{timestamp}.json", "w") as f:
                # json.dump(flow_data.model_dump(), f, indent=3)
                json.dump(flow_data_dict, f, indent=3)
            print(f"Saved to generated_flow_{which_model.split('-long')[0]}_{timestamp}.json")
        else:
            print("Error: Agent did not return a Flow object.")
            print(response.content)

    except Exception as e:
        with open("error.log", "w") as f:
            f.write(str(e))
        print(f"An error occurred: {e}")
