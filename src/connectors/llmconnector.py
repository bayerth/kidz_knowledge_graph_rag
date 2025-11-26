import copy
import json
import time
import logging
from pathlib import Path
from google.genai import types


def default_log(log_string):
    """
    The individual function that consumes the log string.
    Replace this print with whatever logic you need (e.g. storing in a list).
    """
    print(f"------------------------ \n{log_string}")


class FunctionHandler(logging.Handler):
    """
    A logging handler that formats the record and passes the string
    to a provided function.
    """

    def __init__(self, sink_function):
        super().__init__()
        self.sink_function = sink_function

    def emit(self, record):
        try:
            msg = self.format(record)
            self.sink_function(msg)
        except Exception:
            self.handleError(record)


def get_print_logger(name="print_logger", msg_producer=default_log, level=logging.DEBUG):
    """Defines a logger that forwards output to a custom function."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Reset handlers to ensure the new format is applied immediately
    if logger.handlers:
        logger.handlers.clear()

    # Use the custom FunctionHandler instead of StreamHandler
    handler = FunctionHandler(msg_producer)

    formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
            )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


logger = get_print_logger("LLM-Connector")


class LLMCClient:
    def __init__(self, client, model, filename=None, logger=logger):
        self.client = client
        self.model = model
        self.logger = logger
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_reasoning_tokens = 0
        self.total_runtime = 0
        self.history = []
        self.filename = filename
        self.response = None

    def get_client(self):
        return self.client

    def get_model(self):
        return self.model

    def get_history(self):
        return copy.deepcopy(self.history)

    def set_history(self, history):
        self.history = copy.deepcopy(history)

    def clear_history(self, reset_counters=False):
        self.history = []
        if reset_counters:
            self.reset_counters()

    def reset_counters(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_reasoning_tokens = 0
        self.total_runtime = 0

    def write_json(self, message, filename=None):
        filename = filename if filename is not None else self.filename
        try:
            Path(filename).write_text(json.dumps(message, encoding="utf-8"))
        except Exception as e:
            logger.error(f"Error writing JSON file: {e}")

    def write_history(self, filename):
        self.write_json(self.history, filename)

    def send_request(self, user_message, system_message=None, ignore_history=False, temperature=0, write_to_file=False,
                     **kwargs
                     ):
        logger.debug(f"-- GPT Request started. Used model: {self.model} --")
        self.logger.debug(f"Prompt: {user_message}")
        start = time.time()
        response_msg, prompt_tokens, completion_tokens, reasoning_tokens, runtime = self.call_client(
                user_message,
                system_message=system_message,
                ignore_history=ignore_history,
                temperature=0,
                **kwargs
                )
        end = time.time()
        runtime = end - start
        self.logger.debug(
                f"runtime: {runtime:.3f}s, prompt_tokens: {prompt_tokens:,.0f}, completion_tokens: {completion_tokens:,.0f}, reasoning_tokens: {reasoning_tokens:,.0f}, "
                )
        self.total_runtime += runtime
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_reasoning_tokens += reasoning_tokens
        return response_msg, prompt_tokens, completion_tokens, reasoning_tokens, runtime

    def call_client(self, user_message, system_message=None, history=None, temperature=0, write_to_file=False, **kwargs
                    ):
        # history must be maintained in the client
        return "not implemented"


# Python
from openai import OpenAI
from xai_sdk.chat import user as xai_user, system as xai_system


class XAIClient(LLMCClient):
    """Client for xAI (Grok) using OpenAI-compatible API.

    Notes
    - Uses OpenAI Python SDK with `base_url` set to `https://api.x.ai`.
    - Default model can be overridden; common choices: "grok-2-latest" or similar.
    - Maintains chat history like the OpenAI client.
    """

    def call_client(
            self,
            user_message,
            system_message=None,
            ignore_history=False,
            temperature=0,
            retrieved_information=None,
            write_to_file=False,
            **kwargs,
            ):
        # Build chat using xai_sdk's chat interface
        try:
            start = time.time()
            chat = self.client.chat.create(model=self.model)

            # Rehydrate prior history unless ignored
            if not ignore_history and self.history:
                self.logger.debug(f"history: {self.history}")
                for msg in self.history:
                    role = msg.get("role")
                    content = msg.get("content", "")
                    if not content:
                        continue
                    if role == "system":
                        chat.append(xai_system(content))
                    elif role == "user":
                        chat.append(xai_user(content))
                    elif role == "assistant":
                        # xai_sdk does not have an explicit assistant appender; append as system note for context
                        chat.append(xai_system(f"Assistant: {content}"))

            # Optional current system message
            if system_message is not None:
                self.logger.debug(f"System message: {system_message}")
                chat.append(xai_system(system_message))
                if not ignore_history:
                    self.history.append({"role": "system", "content": system_message})

            # Compose current user message (optionally enriched)
            if retrieved_information is not None:
                self.logger.debug(f"Retrieved information: {retrieved_information}")
                user_payload = f"{user_message}. Here is ontology retrieved information based on the topic: {retrieved_information}"
            else:
                user_payload = user_message

            chat.append(xai_user(user_payload))

            # Call the model per xai_sdk spec
            response = chat.sample()
            end = time.time()
            self.response = response
            runtime = end - start
            self.logger.debug(f"xAI Request took {runtime:.2f} seconds")

            if write_to_file:
                # Persist a simplified trace of the conversation
                to_save = [] if ignore_history else list(self.history)
                if system_message is not None:
                    to_save.append({"role": "system", "content": system_message})
                to_save.append({"role": "user", "content": user_payload})
                self.write_json(to_save, filename=self.filename)

        except Exception as e:
            self.logger.error(f"xAI API call failed: {e}")
            return None, None, None, 0, 0.0

        response_message = getattr(response, "content", None)
        if not response_message:
            # Fallback for unexpected shapes
            response_message = "No response from Grok."
        self.logger.debug(f"xAI Respond: {response_message}")

        # xai_sdk `sample()` returns an object with `.content`
        prompt_tokens = 0
        completion_tokens = 0
        reasoning_tokens = 0
        try:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            reasoning_tokens = response.usage.reasoning_tokens
        except Exception as e:
            self.logger.error(f"Error while determining token count: {e}")

        # Maintain our local history for future calls
        self.history.append({"role": "assistant", "content": response_message})

        return response_message, prompt_tokens, completion_tokens, reasoning_tokens, runtime


class OpenAIClient(LLMCClient):
    def call_client(self, user_message, system_message=None, ignore_history=False, temperature=0,
                    retrieved_information=None, write_to_file=False, **kwargs
                    ):
        message = [] if ignore_history else self.history
        self.logger.debug(f"history: {message}")

        if system_message is not None:
            message.append({"role": "system", "content": system_message})
        self.logger.debug(f"System message: {system_message}")

        if retrieved_information is not None:
            self.logger.debug(f"Retrieved information: {retrieved_information}")
            message.append(
                    {"role": "user",
                     "content": f"{user_message}. Here is ontology retrieved information based on the topic: {retrieved_information}"}
                    )
        else:
            message.append({"role": "user", "content": user_message})
        self.logger.debug(f"Retrieved information: {retrieved_information}")
        try:
            start = time.time()
            response = self.client.chat.completions.create(
                    temperature=temperature,
                    messages=message,
                    model=self.model,
                    seed=42
                    )
            end = time.time()
            runtime = end - start
            self.logger.debug(f"GPT Request took {runtime} seconds")
            if write_to_file:
                # result = {"message": message, "response": response, "runtime": runtime}
                result = message
                self.write_json(result, filename=self.filename)
        except Exception as e:
            self.logger.error(f"API call failed: {e}")
            return None, None, None, None, None
        prompt_tokens, completion_tokens = 0, 0
        if response and response.choices:
            try:
                prompt_tokens = response.usage.prompt_tokens
                # prompt_tokens = response.get("usage", {}).get("prompt_tokens")
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                prompt_tokens = response.usage.prompt_tokens
                # self.logger.debug(
                #         f"Total tokens: {total_tokens}, Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}"
                #         )
            except Exception as e:
                self.logger.error(f"Error while determining token count: {e}")
            response_message = response.choices[0].message.content
            self.logger.debug(f"GPT Respond: {response_message}")
            self.history.append({"role": "assistant", "content": response_message})
        else:
            response_message = "Error: No valid response from API."
            self.logger.error(response_message)
        return response_message, prompt_tokens, completion_tokens, 0, runtime


class GeminiClient(LLMCClient):
    # todo: incooperate history
    # prompt_template needs: role, constraints, context, task; output_format and instructions are prescribed
    def __init__(self,
                 client,
                 model,
                 filename=None,
                 system_message="You are Gemini 3, a specialized assistant for Data Science and ontologies. You are precise, analytical, and persistent.",
                 constraints="""
                1. use general reasoning, but only consider data from context and user message
                2. if information is not provided, request specific information: <REQUEST> requested data </REQUEST>,
                3. Context may contain text and ontology-description, do not confuse them
                4. Ontology contains classes with attributes, relations, and instances - use them
                4. Output format is specified in the user message
                """,
                 prompt_template=None,
                 logger=logger
                 ):
        super().__init__(client, model, filename, logger)
        self.system_message = system_message
        self.constraints = constraints
        self.prompt_template = prompt_template if prompt_template is not None else """
    <role>
    {role}
    </role>
    <instructions>
    1. **Plan**: Analyze the task the context and user message.
    2. **Execute**: Identify objects, classes, relations from user message, use provided data (ontology) to get instances or connections
    3. **Validate**: Review your output against the user's task.
    4. **Format**: Present the final answer in the requested structure.
    </instructions>
    <constraints>
    {constraints}
    </constraints>
    <output_format>
    short answer, provide only objects as a list or, only if requestet, as json syntax. Embrace list or json syntax with square brackets. 
    </output_format>
    <context>
    {context}
    </context>
    
    <task>
    {task}
    </task>
    """
        self.history.append({"role": "system", "content": system_message + "\n" + constraints})

    def call_client(self, user_message, system_message=None, ignore_history=False, temperature=0,
                    retrieved_information=None, write_to_file=False, constraints=None, **kwargs
                    ):
        # message = [] if ignore_history else self.history

        system_instruction = "" if system_message is None else system_message
        history = "" if ignore_history else "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in self.history[1:-1]]
                )
        logger.debug(f"history: {history}")
        msg = self.prompt_template.format(
                role=self.system_message,
                constraints=self.constraints if constraints is None else constraints,
                context=system_instruction,
                task=history + " " + user_message
                )
        prompt = types.Content(
                role='user',
                parts=[types.Part.from_text(text=msg)]
                )
        print(msg)
        start = time.time()
        response = self.client.models.generate_content(
                model='gemini-2.5-flash-lite', contents=[prompt]
                )
        end = time.time()
        runtime = end - start
        message = response.text
        print(response.text)
        self.logger.debug(f"history: {message}")
        user_history = "" if constraints is None else constraints
        user_history += "\n" + str(retrieved_information) if retrieved_information is not None else ""
        user_history += f"\n{system_instruction}\n{user_message}"
        self.history.append({"role": "user", "content": user_history})
        self.history.append({"role": "assistant", "content": message})
        prompt_tokens, completion_tokens, reasoning_tokens = 0, 0, 0
        if response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count
            reasoning_tokens = response.usage_metadata.thoughts_token_count
            reasoning_tokens = reasoning_tokens if reasoning_tokens is not None else 0
        return message, prompt_tokens, completion_tokens, reasoning_tokens, runtime
