import copy
import json
import os
import re
import uuid
from pathlib import Path

from pyvis.network import Network

from src.connectors.llmconnector import logger as default_logger

# from ldrag.gptconnector import gpt_request_with_history, gpt_request
from src.rag.ontology import Ontology

MAX_FIND_CLASS_ITERATIONS = 5

# ------------------------------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------------------------------

MAX_LOOPS = 5
STOP_TOKEN = "<STOP>"


# ------------------------------------------------------------------------------------------------------
# RAG
# ------------------------------------------------------------------------------------------------------


def prepare_objects(llm_client, ontology: Ontology, user_query: str, previous_conversation=None, sleep_time=0,
                    logger=default_logger
                    ):
    """
    Step 1 + Step 2 of the RAG pipeline.

    - Step 1: Use class-level ontology to ask GPT which classes are relevant for the user query.
    - Step 2: From instances of those classes, ask GPT to pick the most relevant starting node(s).

    Returns a tuple:
      (ontology_structure, starting_nodes, retrieved_node_dict, history, gpt_response)
    where
      - ontology_structure: structure returned by ontology.get_ontology_structure()
      - starting_nodes: list of node-structure dicts for the selected starting instances
      - retrieved_node_dict: dict[node_id -> node] for the selected starting nodes (will be mutated by later steps)
      - history: conversation history returned by GPT for Step 2
      - gpt_response: latest assistant raw text (used to seed the traversal loop)
    """
    logger.info("Starting RAG - prepare_objects (Steps 1 & 2)")

    # -------------------------------
    # Step 1
    # -------------------------------
    ontology_structure = ontology.get_ontology_structure()
    system_message = (
            f"The following structure illustrates the class level of the ontology, which will be used to answer the subsequent "
            f"questions. The node classes have instances that are not listed here. :{json.dumps(ontology_structure)}."
    )
    user_message = (
            f"Only give as an answer a list of classes (following this syntax: [class1, class2, ...]) which are relevant "
            f"for this user query: {user_query} Return only JSON Syntax without prefix."
    )
    logger.debug(f"user_message: {user_message}")
    logger.debug(f"system_message: {system_message}")
    total_input_tokens, total_completion_tokens, total_reasoning_tokens, total_runtime = 0, 0, 0, 0
    no_class_found = True
    find_class_iterations = 0
    while no_class_found and find_class_iterations < MAX_FIND_CLASS_ITERATIONS:
        response_msg, prompt_tokens, completion_tokens, reasoning_tokens, runtime = llm_client.send_request(
                user_message=user_message,
                system_message=system_message,  # change to retrieved_information=system_message,
                previous_conversation=previous_conversation,
                logger=logger,
                sleep_time=sleep_time,
                )
        if response_msg != "[]":
            no_class_found = False
        else:
            find_class_iterations += 1
            
    if no_class_found:
        logger.info(f"No class found for query {user_query}, returning empty list.")
        return {}, None

    # Parse classes from response
    found_node_class_list = re.findall(r'\w+', response_msg)
    logger.info(f"Found node classes: {found_node_class_list}")

    # -------------------------------
    # Step 2: Identify possible starting nodes
    # -------------------------------
    instance_ids = [node.get_node_id() for node in ontology.get_instances_by_class(found_node_class_list)]
    user_message = (
            f"Here is a list of instances: {str(instance_ids)}. To which of them refers this user query: {user_query}? "
            f"Only use the correct one. You can ignore spelling error or cases. Return only JSON Syntax without prefix."
    )
    response_msg, prompt_tokens, completion_tokens, reasoning_tokens, runtime = llm_client.send_request(
            user_message=user_message,
            previous_conversation=previous_conversation,
            sleep_time=sleep_time,
            logger=logger,
            )
    logger.debug(f"response_msg: {response_msg}")
    found_node_instances_list = re.findall(r'\w+', response_msg)
    logger.info(f"Found node instances: {found_node_instances_list}")

    # retrieved_node_dict = ontology.get_nodes(found_node_instances_list)
    # starting_nodes = [ontology.get_node_structure(node) for node in retrieved_node_dict.values()]

    return found_node_class_list, found_node_instances_list


def iterate_ontology(
        llm_client,
        ontology: Ontology,
        user_query: str,
        # ontology_structure,
        relevant_classes,
        starting_nodes_id,
        sleep_time=0,
        logger=default_logger,
        ):
    """
    Step 3 + Step 4 of the RAG pipeline.

    - Step 3: Iterative ontology traversal using GPT requests (bracketed request syntax),
              updating retrieved_node_dict with discovered nodes until STOP_TOKEN or loop limit.
    - Step 4: Package results and generate graph HTML.

    Returns: (retrieved_relevant_information, graph_path)
    """
    # -------------------------------
    # Step 3: Iterative search
    # -------------------------------

    logger.info("Beginning iterative ontology search: Before Loop ")
    logger.info(f"Iteration 0. Starting node: {starting_nodes_id}")
    # system_message = (
    #     f"You are given a starting node, which is part of an ontology. Your job is to traverse the ontology to gather "
    #     f"enough information to answer given questions. Every node is connected to other nodes. You can find the "
    #     f"connections under  \"'Connections':\" in the form of  \"'Connections': <name of the edge> <name of the connected node>. "
    #     f"For example  'Connections': trainedWith data_1. You can request new nodes. To do so write [name of the "
    #     f"requested node], for example [data_1]. You can ask for more than one instance this way. For example  [data_1, data_2]. "
    #     f"As long as you search for new information, only use this syntax, don't explain yourself. Use the exact name of the "
    #     f"instance and don't use the edge. Your job is to gather enough information to answer given questions. To do so, "
    #     f"traverse trough the ontology. If you have enough information, write {STOP_TOKEN} Use this class level ontology to "
    #     f"orientate yourself: {str(ontology_structure)}. Return only JSON Syntax without prefix."
    # )
    # previous_conversation = [
    #     {"role": "assistant", "content": "What is the starting node for the user query?"},
    #     {
    #         "role": "user",
    #         "content": f"This is the starting node: {starting_nodes}. Write {STOP_TOKEN} if you have enough information to answer the query or request new nodes.",
    #     },
    # ]
    retrieved_node_dict = ontology.get_nodes(starting_nodes_id)
    # starting_nodes = [ontology.get_node_structure(node) for node in retrieved_node_dict.values()]
    # user_message = f"Here is a list of instances: {str(starting_nodes)}. To which of them refers this user query: {user_query}? Only use the correct one. You can ignore spelling error or cases. Return only JSON Syntax without prefix."
    found_node_instances = retrieved_node_dict.values()

    if found_node_instances is None:
        logger.error("No response from GPT, exiting before loop.")
        return [], None

    # -- iterate over retrieved nodes  --

    logger.info("--- RAG: Starting iterative ontology search")
    response_msg = ""
    loop_count = 0
    while STOP_TOKEN not in response_msg and loop_count < MAX_LOOPS:
        loop_count += 1
        if found_node_instances is None:
            found_node_instances = execute_query(response_msg, ontology)
        logger.info(f"Iteration {loop_count} of {MAX_LOOPS}: Requested nodes: {len(found_node_instances)}")
        retrieved_information = []
        if found_node_instances:
            logger.debug(
                    f"Nodes found: {[ontology.get_node_structure(node)['Node Instance ID'] for node in found_node_instances]}."
                    )
            for node in found_node_instances:
                retrieved_information.append(ontology.get_node_structure(node))
                retrieved_node_dict.update({f"{node.get_node_id()}": node})
        else:
            retrieved_information = "No instance exists for that ID. You asked for a class or searched for a non existing instance."
            logger.info(f"Iteration {loop_count}. No nodes where found.")
        user_message = f"This is the result to your query: {retrieved_information}. If you need more information, use another query, otherwise write only {STOP_TOKEN}. Return only JSON Syntax without prefix."

        response_msg, prompt_tokens, completion_tokens, reasoning_tokens, runtime = llm_client.send_request(
                user_message=user_message,
                sleep_time=sleep_time,
                retrieved_information=retrieved_information,
                logger=logger,
                )
        found_node_instances = None
    logger.info(f"Iterative search ended after iteration {loop_count}")
    # logger.debug(
    #         f"Total input tokens: {total_input_tokens:,.0f}, Total completion tokens: {total_completion_tokens:,.0f}"
    #         )
    retrieved_relevant_information = []
    for node in retrieved_node_dict.values():
        retrieved_relevant_information.append(ontology.get_node_structure(node))
    # retrieved_relevant_information = [str(obj) for obj in retrieved_node_dict.values()]
    logger.debug(retrieved_relevant_information)
    return retrieved_relevant_information, retrieved_node_dict


def information_retriever_with_graph(ontology: Ontology, user_query: str, previous_conversation=None, sleep_time=0,
                                     logger=default_logger
                                     ):
    logger.info("Starting RAG")
    total_runtime = 0
    # ------------------------------------------------------------------------------------
    # Step 1
    # ------------------------------------------------------------------------------------
    ontology_structure = ontology.get_ontology_structure()
    logger.debug(f"classes: {ontology_structure}")
    # Identify the used classes, so we don't have to give gpt every single instance to pick an anker node
    total_input_tokens, total_completion_tokens = 0, 0
    system_message = f"The following structure illustrates the class level of the ontology, which will be used to answer the subsequent questions. The node classes have instances that are not listed here. :{json.dumps(ontology_structure)}."
    user_message = f"Only give as an answer a list of classes (following this syntax: [class1, class2, ...]) which are relevant for this user query: {user_query} Return only JSON Syntax without prefix."
    no_class_found = True
    find_class_iterations = 0
    while no_class_found or find_class_iterations > 10:
        # logger.debug(f"user_message: {user_message}")
        # logger.debug(f"system_message: {system_message}")
        # logger.debug(f"previous_conversation: {previous_conversation}")
        gpt_response, _, input_tokens, completion_tokens, runtime = gpt_request_with_history(
                user_message=user_message,
                system_message=system_message,
                previous_conversation=previous_conversation,
                logger=logger,
                sleep_time=sleep_time
                )
        total_runtime += runtime
        total_input_tokens += input_tokens
        total_completion_tokens += completion_tokens
        # gpt_response = gpt_request(user_message=user_message,
        #                            system_message=system_message,
        #                            previous_conversation=previous_conversation,
        #                            logger=logger,
        #                            sleep_time=sleep_time
        #                            )
        if gpt_response != "[]":
            no_class_found = False
        else:
            find_class_iterations += 1
        if no_class_found:
            logger.info(f"No class found for query {user_query}, returning empty list.")
            return [], None

    # Todo Besseres Errorhandling implementieren
    found_node_class_list = re.findall(r'\w+', gpt_response)
    logger.info(f"Found node classes: {found_node_class_list}")

    # ------------------------------------------------------------------------------------
    # Step 2: Identify possible starting nodes
    # ------------------------------------------------------------------------------------

    instance_ids = [node.get_node_id() for node in ontology.get_instances_by_class(found_node_class_list)]
    user_message = f"Here is a list of instances: {str(instance_ids)}. To which of them refers this user query: {user_query}? Only use the correct one. You can ignore spelling error or cases. Return only JSON Syntax without prefix."
    gpt_response, history, input_tokens, completion_tokens, runtime = \
        gpt_request_with_history(
                user_message=user_message,
                previous_conversation=previous_conversation,
                sleep_time=sleep_time, logger=logger,
                filename="gpt_request"
                )
    total_runtime += runtime
    total_input_tokens += input_tokens
    total_completion_tokens += completion_tokens

    found_node_instances_list = re.findall(r'\w+', gpt_response)
    retrieved_node_dict = ontology.get_nodes(found_node_instances_list)
    logger.info(f"Found node instances: {found_node_instances_list}")

    starting_nodes = [ontology.get_node_structure(node) for node in retrieved_node_dict.values()]

    # ------------------------------------------------------------------------------------
    # Step 3: Iterative search
    # ------------------------------------------------------------------------------------

    # -- create initial prompt for graph-traversal
    logger.info("Beginning iterative ontology search: Before Loop ")
    logger.info(f"Iteration 0. Starting node: {starting_nodes}")
    system_message = f"You are given a starting node, which is part of an ontology. Your job is to traverse the ontology to gather enough information to answer given questions. Every node is connected to other nodes. You can find the connections under  \"'Connections':\" in the form of  \"'Connections': <name of the edge> <name of the connected node>. For example  'Connections': trainedWith data_1. You can request new nodes. To do so write [name of the requested node], for example [data_1]. You can ask for more than one instance this way. For example  [data_1, data_2]. As long as you search for new information, only use this syntax, don't explain yourself. Use the exact name of the instance and don't use the edge. Your job is to gather enough information to answer given questions. To do so, traverse trough the ontology. If you have enough information, write {STOP_TOKEN} Use this class level ontology to orientate yourself: {str(ontology_structure)}. Return only JSON Syntax without prefix."
    previous_conversation = [
            {"role": "assistant", "content": "What is the starting node for the user query?"},
            {"role": "user",
             "content": f"This is the starting node: {starting_nodes}. Write {STOP_TOKEN} if you have enough information to answer the query or request new nodes."}
            ]
    # gpt_response, history = gpt_request_with_history(user_message=user_query, system_message=system_message,
    #                                                  previous_conversation=previous_conversation,
    #                                                  sleep_time=sleep_time,logger=logger)
    if gpt_response is None:
        logger.error("No response from GPT, exiting before loop.")
        return [], None

    logger.info("--- RAG: Starting iterative ontology search")
    loop_count = 0
    while loop_count < MAX_LOOPS and STOP_TOKEN not in gpt_response:
        loop_count += 1
        logger.info(f"Iteration {loop_count} of {MAX_LOOPS}: Requested nodes: {gpt_response}")
        found_node_instances = execute_query(gpt_response, ontology)
        retrieved_information = []
        if found_node_instances:
            logger.info(
                    f"Iteration {loop_count}. Nodes found: {[ontology.get_node_structure(node) for node in found_node_instances]}."
                    )
            for node in found_node_instances:
                retrieved_information.append(ontology.get_node_structure(node))
                retrieved_node_dict.update({f"{node.get_node_id()}": node})
        else:
            retrieved_information = "No instance exists for that ID. You asked for a class or searched for a non existing instance."
            logger.info(f"Iteration {loop_count}. No nodes where found.")
        user_message = f"This is the result to your query: {retrieved_information}. If you need more information, use another query, otherwise write {STOP_TOKEN}. Return only JSON Syntax without prefix."

        gpt_response, history, input_tokens, completion_tokens, runtime = gpt_request_with_history(
                user_message=user_message,
                previous_conversation=history,
                sleep_time=sleep_time,
                logger=logger
                )
        total_runtime += runtime
        total_input_tokens += input_tokens
        total_completion_tokens += completion_tokens
    logger.info(f"Iterative search ended after iteration {loop_count}")
    logger.debug(
            f"Total input tokens: {total_input_tokens:,.0f}, Total completion tokens: {total_completion_tokens:,.0f}"
            )

    # ------------------------------------------------------------------------------------
    # Step 4: Generate graph and answer
    # ------------------------------------------------------------------------------------

    retrieved_graph_id = uuid.uuid1()
    graph_path = create_rag_instance_graph(retrieved_node_dict, retrieved_graph_id, user_query)
    retrieved_relevant_information = []
    for node in retrieved_node_dict.values():
        retrieved_relevant_information.append(ontology.get_node_structure(node))
    # retrieved_relevant_information = [str(obj) for obj in retrieved_node_dict.values()]
    logger.debug(retrieved_relevant_information)
    return retrieved_relevant_information, graph_path, history, total_input_tokens, total_completion_tokens, loop_count, runtime


def execute_query(query, ontology):
    # found_node_instances_list = re.findall(r'\w+', gpt_response)
    pattern = r"(?<=\[|,|\s)([^,\]\s]+)(?=,|\]|\s)"
    matches = re.findall(pattern, query)
    matches = [m.strip('"') for m in matches]
    return list(ontology.get_nodes(matches).values())


def rag(question, ontology, llm_client, logger=default_logger):
    logger.debug("-- RAG startet ---")
    logger.debug(f"Question: {question}")

    found_node_class_list, found_node_instances_list = prepare_objects(
            llm_client=llm_client,
            ontology=ontology,
            user_query=question,
            logger=logger
            )

    retrieved_relevant_information, retrieved_node_dict = iterate_ontology(
            llm_client=llm_client,
            ontology=ontology,
            user_query=question,
            relevant_classes=found_node_class_list,
            starting_nodes_id=found_node_instances_list,
            logger=logger
            )
    logger.debug(f"Retrieved relevant information: {retrieved_relevant_information}")
    logger.debug(f"Retrieved graph: {retrieved_node_dict}")
    llm_client.clear_history()
    response_msg, prompt_tokens, completion_tokens, reasoning_tokens, runtime = llm_client.send_request(
            question,
            retrieved_information=retrieved_relevant_information,
            logger=logger
            )
    summary = f"runtime: {llm_client.total_runtime:.2f} seconds, prompt tokens: {llm_client.total_prompt_tokens:,.0f}, completion tokens: {llm_client.total_completion_tokens:,.0f}, reasoning tokens: {llm_client.total_reasoning_tokens:,.0f}"
    logger.info(summary)
    logger.info(f"Assistend response: {response_msg}")
    # logger.debug(
    #         f"Total input tokens: {total_input_tokens:,.0f}, Total completion tokens: {total_completion_tokens:,.0f}"
    #         )
    return response_msg, retrieved_node_dict


import os
from pyvis.network import Network


def create_rag_instance_graph(node_dict, graph_id=None, text="RAG-Graph"):
    """
    Creates an interactive graph visualization for Retrieval Augmented Generation (RAG) instances.

    :param node_dict: Dictionary containing nodes and their connections
    :param graph_id: Unique identifier for the question
    :param text: The question being visualized
    :return: Path to the saved HTML file containing the graph
    """
    graph_id = graph_id or uuid.uuid1()
    net = Network(height="100vh", width="100vw", directed=True, notebook=False)

    # Header blue color
    header_blue = "#457b9d"

    # Add nodes with custom color
    for node in node_dict.values():
        net.add_node(
                node.get_node_id(),
                title=node.get_internal_structure(),
                color=header_blue  # Set node color to header blue
                )

    for node in node_dict.values():
        for connection in node.get_node_connections():
            # Ensure connection is a dictionary and contains necessary keys
            if isinstance(connection, dict) and "target" in connection and "relation" in connection:
                target = connection["target"]
                relation = connection["relation"]

                # Ensure the target node actually exists in rag_dict before adding an edge
                if target in node_dict:
                    net.add_edge(node.get_node_id(), target, label=relation, arrows="to", length=400)

    output_file = f"./rag_{graph_id}.html"
    directory = os.path.dirname(output_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    net.save_graph(output_file)

    # Add custom styling and header
    custom_styles = """
    <style>
        body {
            font-family: 'Barlow Semi Condensed', Arial, sans-serif;
            margin: 0;
            background-color: #f4f4f9;
            color: #333;
        }
        #header {
            font-family: 'Barlow Semi Condensed', Arial, sans-serif;
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #457b9d;
            color: white;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
        }
        #logo {
            height: 50px;
            width: auto;
            margin-right: 15px;
        }
        h1 {
            font-family: 'Barlow Semi Condensed', Arial, sans-serif;
            font-size: 1.8rem;
            margin: 0;
        }
        h2 {
            font-family: 'Barlow Semi Condensed', Arial, sans-serif;
            text-align: center;
            color: #457b9d;
            margin-top: 20px;
            font-weight: 700;
        }
        #graph-container {
            margin: 20px auto;
            max-width: 90%;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: white;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }
    </style>
    """

    custom_header = f"""
    <div id="header">
        <img src="../images/kidz.png" alt="Logo" id="logo">
        <h1>Retrieval Augmented Generation</h1>
    </div>
    <h2>{text}</h2>
    <div id="graph-container">
    """

    # Inject custom styles and header into the graph's HTML
    with open(output_file, 'r') as file:
        html_content = file.read()

    html_content = html_content.replace("<head>", f"<head>\n{custom_styles}", 1)
    html_content = html_content.replace("<body>", f"<body>\n{custom_header}", 1)
    html_content = html_content.replace("</body>", "</div>\n</body>", 1)

    with open(output_file, 'w') as file:
        file.write(html_content)

    return output_file


# ------------------------------------------------------------------------------------------------------
# Support
# ------------------------------------------------------------------------------------------------------

def _resolve_eval_path(path_or_stem: str) -> Path:
    """
    Resolve a path to an evaluations file. Accepts either a full path or a stem like
    "data/question_evaluations" and will try common extensions.

    Tries, in order:
    - as given
    - with ".json"
    - with ".jsonl"
    - with ".txt"

    If the argument is a relative path, it is also tried relative to the project root.
    Project root is inferred as two levels above this file (i.e., repository root).
    """
    candidates = []

    # Helper to add variants for a base path
    def add_variants(p: Path):
        candidates.append(p)
        if p.suffix == "":
            candidates.extend([p.with_suffix(".json"), p.with_suffix(".jsonl"), p.with_suffix(".txt")])

    given = Path(path_or_stem)
    add_variants(given)

    # Also try relative to project root
    project_root = Path(__file__).resolve().parents[2]
    add_variants(project_root / given)

    for c in candidates:
        if c.exists() and c.is_file():
            return c

    raise FileNotFoundError(f"Could not resolve evaluations file for: {path_or_stem}")


def _iter_records_from_file(file_path: Path):
    """
    Yield dict records from the given file path. Supports:
    - JSON array file (e.g., [ {..}, {..} ])
    - JSONL/line-delimited JSON (one object per line)
    - A .txt that contains either of the above
    """
    text = file_path.read_text(encoding="utf-8")
    # First try to parse as a full JSON document
    try:
        data = json.loads(text)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
                else:
                    default_logger.error("Non-dict item encountered in JSON array; skipping.")
        elif isinstance(data, dict):
            # Single object file, yield it
            yield data
        else:
            # Fallback to JSONL parsing
            raise ValueError("Top-level JSON was not list/dict; trying JSONL")
        return
    except Exception:
        pass

    # Try JSON Lines (ignore empty/comment lines)
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj
            else:
                default_logger.error("Non-dict JSONL line encountered; skipping.")
        except json.JSONDecodeError:
            # Not valid JSON per line; skip silently to be permissive
            continue


def read_records_by_unique_id(path_or_stem: str, logger=default_logger) -> dict:
    """
    Read an evaluations-like JSON file and return a dict keyed by `unique_id` with the
    remaining attributes as the value.

    The function accepts either a full file path or a stem (e.g., "data/question_evaluations").
    It will look for .json, .jsonl, or .txt variations, and supports JSON array or
    JSON Lines formats.

    Returns
    -------
    dict
        Mapping of unique_id -> {all other attributes}
    """
    file_path = _resolve_eval_path(path_or_stem)
    result = {}
    duplicates = []
    missing_id_count = 0

    for rec in _iter_records_from_file(file_path):
        if not isinstance(rec, dict):
            continue
        uid = rec.get("unique_id")
        if not uid:
            missing_id_count += 1
            continue
        # Copy without unique_id
        value = {k: v for k, v in rec.items() if k != "unique_id"}
        if uid in result:
            duplicates.append(uid)
        result[uid] = value

    if duplicates:
        logger.info(f"Duplicate unique_id entries encountered; last occurrence kept. ids={sorted(set(duplicates))}")
    if missing_id_count:
        logger.info(f"Skipped {missing_id_count} records without 'unique_id'.")

    return result
