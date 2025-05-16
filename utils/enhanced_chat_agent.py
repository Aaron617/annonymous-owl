from __future__ import annotations

import json
import logging
import textwrap
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Type,
    Union,
)

from openai import (
    AsyncStream,
    Stream,
)
from pydantic import BaseModel, ValidationError

from camel.agents._types import ModelResponse, ToolCallRequest

from camel.agents.base import BaseAgent
from camel.memories import (
    AgentMemory,
    ChatHistoryMemory,
    MemoryRecord,
    ScoreBasedContextCreator,
)
from camel.messages import BaseMessage, FunctionCallingMessage, OpenAIMessage
from camel.models import (
    BaseModelBackend,

)
from camel.prompts import TextPrompt
from camel.responses import ChatAgentResponse
from camel.toolkits import FunctionTool
from camel.types import (
    ModelPlatformType,
    ModelType,
    OpenAIBackendRole,
    RoleType,
)
from camel.types.agents import ToolCallingRecord
from camel.utils import get_model_encoding
from camel.agents.chat_agent import ChatAgent
from retry import retry
import openai

if TYPE_CHECKING:
    from camel.terminators import ResponseTerminator


logger = logging.getLogger(__name__)


class OwlChatAgent(ChatAgent):
    def __init__(
        self, 
        system_message: Optional[Union[BaseMessage, str]] = None,
        model: Optional[
            Union[BaseModelBackend, List[BaseModelBackend]]
        ] = None,
        memory: Optional[AgentMemory] = None,
        message_window_size: Optional[int] = None,
        token_limit: Optional[int] = None,
        output_language: Optional[str] = None,
        tools: Optional[List[Union[FunctionTool, Callable]]] = None,
        external_tools: Optional[
            List[Union[FunctionTool, Callable, Dict[str, Any]]]
        ] = None,
        response_terminators: Optional[List[ResponseTerminator]] = None,
        scheduling_strategy: str = "round_robin",
        single_iteration: bool = False,
        agent_id: Optional[str] = None,
    ):
        super().__init__(
            system_message,
            model,
            memory,
            message_window_size,
            token_limit,
            output_language,
            tools,
            external_tools,
            response_terminators,
            scheduling_strategy,
            single_iteration,
            agent_id
        )

    
    @retry(openai.APIConnectionError, backoff=2, max_delay=60)
    def step(
        self,
        input_message: Union[BaseMessage, str],
        response_format: Optional[Type[BaseModel]] = None,
        max_tool_calls: int = 15
    ) -> ChatAgentResponse:
        
        if isinstance(input_message, str):
            input_message = BaseMessage.make_user_message(
                role_name="User", content=input_message
            )

        # Add user input to memory
        self.update_memory(input_message, OpenAIBackendRole.USER)

        tool_call_records: List[ToolCallingRecord] = []
        external_tool_call_requests: Optional[List[ToolCallRequest]] = None

        while True:
            is_tool_call_limit_reached = False
            try:
                openai_messages, num_tokens = self.memory.get_context()
            except RuntimeError as e:
                return self._step_token_exceed(
                    e.args[1], tool_call_records, "max_tokens_exceeded"
                )
            # Get response from model backend
            response = self._get_model_response(
                openai_messages,
                num_tokens,
                response_format,
                self._get_full_tool_schemas(),
            )

            if self.single_iteration:
                break

            if tool_call_requests := response.tool_call_requests:
                # Process all tool calls
                for tool_call_request in tool_call_requests:
                    if tool_call_request.tool_name in self._external_tool_schemas:
                        if external_tool_call_requests is None:
                            external_tool_call_requests = []
                        external_tool_call_requests.append(tool_call_request)
                    else:
                        tool_call_records.append(self._execute_tool(tool_call_request))
                        if len(tool_call_records) > max_tool_calls:
                            is_tool_call_limit_reached = True
                            break

                # If we found external tool calls or reached the limit, break the loop
                if external_tool_call_requests or is_tool_call_limit_reached:
                    break

                if self.single_iteration:
                    break

                # If we're still here, continue the loop
                continue

            break

        self._format_response_if_needed(response, response_format)
        self._record_final_output(response.output_messages)
        
        if is_tool_call_limit_reached:
            tool_call_msgs = []
            for tool_call in tool_call_records:
                
                result = str(tool_call.result)
                # if result is too long, truncate it
                max_result_length = 800
                if len(result) > max_result_length:
                    result = result[:max_result_length] + "..." + f" (truncated, total length: {len(result)})"
                
                tool_call_msgs.append({
                    "function": tool_call.tool_name,
                    "args": tool_call.args,
                    "result": result
                })
            
            response.output_messages[0].content = f"""
The tool call limit has been reached. Here is the tool calling history so far:
{json.dumps(tool_call_msgs, indent=2)}

Please try other ways to get the information.
"""
            return self._convert_to_chatagent_response(
                response, tool_call_records, num_tokens, external_tool_call_requests
            )
        
        return self._convert_to_chatagent_response(
            response, tool_call_records, num_tokens, external_tool_call_requests
        )
        
        
    async def astep(
        self,
        input_message: Union[BaseMessage, str],
        response_format: Optional[Type[BaseModel]] = None,
        max_tool_calls: int = 15
    ) -> ChatAgentResponse:

        if isinstance(input_message, str):
            input_message = BaseMessage.make_user_message(
                role_name="User", content=input_message
            )


        self.update_memory(input_message, OpenAIBackendRole.USER)

        tool_call_records: List[ToolCallingRecord] = []
        external_tool_call_requests: Optional[List[ToolCallRequest]] = None
        while True:
            is_tool_call_limit_reached = False
            try:
                openai_messages, num_tokens = self.memory.get_context()
            except RuntimeError as e:
                return self._step_token_exceed(
                    e.args[1], tool_call_records, "max_tokens_exceeded"
                )

            response = await self._aget_model_response(
                openai_messages,
                num_tokens,
                response_format,
                self._get_full_tool_schemas(),
            )

            if self.single_iteration:
                break

            if tool_call_requests := response.tool_call_requests:
                # Process all tool calls
                for tool_call_request in tool_call_requests:
                    if tool_call_request.tool_name in self._external_tool_schemas:
                        if external_tool_call_requests is None:
                            external_tool_call_requests = []
                        external_tool_call_requests.append(tool_call_request)
                    else:
                        tool_call_record = await self._aexecute_tool(tool_call_request)
                        tool_call_records.append(tool_call_record)
                        if len(tool_call_records) > max_tool_calls:
                            is_tool_call_limit_reached = True
                            break

                # If we found external tool calls or reached the limit, break the loop
                if external_tool_call_requests or is_tool_call_limit_reached:
                    break

                if self.single_iteration:
                    break

                # If we're still here, continue the loop
                continue

            break

        await self._aformat_response_if_needed(response, response_format)
        self._record_final_output(response.output_messages)
        
        if is_tool_call_limit_reached:
            tool_call_msgs = []
            for tool_call in tool_call_records:
                
                result = str(tool_call.result)
                # if result is too long, truncate it
                max_result_length = 800
                if len(result) > max_result_length:
                    result = result[:max_result_length] + "..." + f" (truncated, total length: {len(result)})"
                
                tool_call_msgs.append({
                    "function": tool_call.tool_name,
                    "args": tool_call.args,
                    "result": result
                })
            debug_content = f"""
The tool call limit has been reached. Here is the tool calling history so far:
{json.dumps(tool_call_msgs, indent=2)}

Please try other ways to get the information.
"""
            response.output_messages[0].content = debug_content

            return self._convert_to_chatagent_response(
                response, tool_call_records, num_tokens, external_tool_call_requests
            )

        return self._convert_to_chatagent_response(
            response, tool_call_records, num_tokens, external_tool_call_requests
        )



class OwlWorkforceChatAgent(ChatAgent):
    def __init__(
        self, 
        system_message: Optional[Union[BaseMessage, str]] = None,
        model: Optional[
            Union[BaseModelBackend, List[BaseModelBackend]]
        ] = None,
        memory: Optional[AgentMemory] = None,
        message_window_size: Optional[int] = None,
        token_limit: Optional[int] = None,
        output_language: Optional[str] = None,
        tools: Optional[List[Union[FunctionTool, Callable]]] = None,
        external_tools: Optional[
            List[Union[FunctionTool, Callable, Dict[str, Any]]]
        ] = None,
        response_terminators: Optional[List[ResponseTerminator]] = None,
        scheduling_strategy: str = "round_robin",
        single_iteration: bool = False,
        agent_id: Optional[str] = None,
    ):
        super().__init__(
            system_message,
            model,
            memory,
            message_window_size,
            token_limit,
            output_language,
            tools,
            external_tools,
            response_terminators,
            scheduling_strategy,
            single_iteration,
            agent_id
        )
    
    
    @retry(openai.APIConnectionError, backoff=2, max_delay=60)
    def step(
        self,
        input_message: Union[BaseMessage, str],
        response_format: Optional[Type[BaseModel]] = None,
        max_tool_calls: int = 15
    ) -> ChatAgentResponse:
        
        if isinstance(input_message, str):
            input_message = BaseMessage.make_user_message(
                role_name="User", content=input_message
            )

        # Add user input to memory
        self.update_memory(input_message, OpenAIBackendRole.USER)

        tool_call_records: List[ToolCallingRecord] = []
        external_tool_call_requests: Optional[List[ToolCallRequest]] = None

        while True:
            is_tool_call_limit_reached = False
            try:
                openai_messages, num_tokens = self.memory.get_context()
            except RuntimeError as e:
                return self._step_token_exceed(
                    e.args[1], tool_call_records, "max_tokens_exceeded"
                )
            # Get response from model backend
            response = self._get_model_response(
                openai_messages,
                num_tokens,
                response_format,
                self._get_full_tool_schemas(),
            )

            if self.single_iteration:
                break

            if tool_call_requests := response.tool_call_requests:
                # Process all tool calls
                for tool_call_request in tool_call_requests:
                    if tool_call_request.tool_name in self._external_tool_schemas:
                        if external_tool_call_requests is None:
                            external_tool_call_requests = []
                        external_tool_call_requests.append(tool_call_request)
                    else:
                        tool_call_records.append(self._execute_tool(tool_call_request))
                        if len(tool_call_records) > max_tool_calls:
                            is_tool_call_limit_reached = True
                            break

                # If we found external tool calls or reached the limit, break the loop
                if external_tool_call_requests or is_tool_call_limit_reached:
                    break

                if self.single_iteration:
                    break

                # If we're still here, continue the loop
                continue

            break

        self._format_response_if_needed(response, response_format)
        self._record_final_output(response.output_messages)
        
        if is_tool_call_limit_reached:
            tool_call_msgs = []
            for tool_call in tool_call_records:
                
                result = str(tool_call.result)
                # if result is too long, truncate it
                max_result_length = 800
                if len(result) > max_result_length:
                    result = result[:max_result_length] + "..." + f" (truncated, total length: {len(result)})"
                
                tool_call_msgs.append({
                    "function": tool_call.tool_name,
                    "args": tool_call.args,
                    "result": result
                })
            debug_content = f"""
The tool call limit has been reached. Here is the tool calling history so far:
{json.dumps(tool_call_msgs, indent=2)}

Please try other ways to get the information.
"""
            # the content should be a json object
            response.output_messages[0].content = f"""
{{
    "content": "{debug_content}",
    "failed": true
}}
"""

            return self._convert_to_chatagent_response(
                response, tool_call_records, num_tokens, external_tool_call_requests
            )
        
        return self._convert_to_chatagent_response(
            response, tool_call_records, num_tokens, external_tool_call_requests
        )
    
    
    async def astep(
        self,
        input_message: Union[BaseMessage, str],
        response_format: Optional[Type[BaseModel]] = None,
        max_tool_calls: int = 15
    ) -> ChatAgentResponse:
        r"""Performs a single step in the chat session by generating a response
        to the input message. This agent step can call async function calls.

        Args:
            input_message (Union[BaseMessage, str]): The input message to the
                agent. For BaseMessage input, its `role` field that specifies
                the role at backend may be either `user` or `assistant` but it
                will be set to `user` anyway since for the self agent any
                incoming message is external. For str input, the `role_name`
                would be `User`.
            response_format (Optional[Type[BaseModel]], optional): A pydantic
                model class that includes value types and field descriptions
                used to generate a structured response by LLM. This schema
                helps in defining the expected output format. (default:
                :obj:`None`)
            max_tool_calls (int, optional): Maximum number of tool calls allowed
                before interrupting the process. (default: :obj:`15`)

        Returns:
            ChatAgentResponse: A struct containing the output messages,
                a boolean indicating whether the chat session has terminated,
                and information about the chat session.
        """
        if isinstance(input_message, str):
            input_message = BaseMessage.make_user_message(
                role_name="User", content=input_message
            )

        self.update_memory(input_message, OpenAIBackendRole.USER)

        tool_call_records: List[ToolCallingRecord] = []
        external_tool_call_requests: Optional[List[ToolCallRequest]] = None
        while True:
            is_tool_call_limit_reached = False
            try:
                openai_messages, num_tokens = self.memory.get_context()
            except RuntimeError as e:
                return self._step_token_exceed(
                    e.args[1], tool_call_records, "max_tokens_exceeded"
                )

            response = await self._aget_model_response(
                openai_messages,
                num_tokens,
                response_format,
                self._get_full_tool_schemas(),
            )

            if self.single_iteration:
                break

            if tool_call_requests := response.tool_call_requests:
                # Process all tool calls
                for tool_call_request in tool_call_requests:
                    if tool_call_request.tool_name in self._external_tool_schemas:
                        if external_tool_call_requests is None:
                            external_tool_call_requests = []
                        external_tool_call_requests.append(tool_call_request)
                    else:
                        tool_call_record = await self._aexecute_tool(tool_call_request)
                        tool_call_records.append(tool_call_record)
                        if len(tool_call_records) > max_tool_calls:
                            is_tool_call_limit_reached = True
                            break

                # If we found external tool calls or reached the limit, break the loop
                if external_tool_call_requests or is_tool_call_limit_reached:
                    break

                if self.single_iteration:
                    break

                # If we're still here, continue the loop
                continue

            break

        await self._aformat_response_if_needed(response, response_format)
        self._record_final_output(response.output_messages)

        if is_tool_call_limit_reached:
            tool_call_msgs = []
            for tool_call in tool_call_records:
                
                result = str(tool_call.result)
                # if result is too long, truncate it
                max_result_length = 800
                if len(result) > max_result_length:
                    result = result[:max_result_length] + "..." + f" (truncated, total length: {len(result)})"
                
                tool_call_msgs.append({
                    "function": tool_call.tool_name,
                    "args": tool_call.args,
                    "result": result
                })
            debug_content = f"""
The tool call limit has been reached. Here is the tool calling history so far:
{json.dumps(tool_call_msgs, indent=2)}

Please try other ways to get the information.
"""         
            # request should be a json object
            response_dict = {
                "content": debug_content,
                "failed": True
            }
            response.output_messages[0].content = json.dumps(response_dict)
            
            return self._convert_to_chatagent_response(
                response, tool_call_records, num_tokens, external_tool_call_requests
            )
            
        return self._convert_to_chatagent_response(
            response, tool_call_records, num_tokens, external_tool_call_requests
        )
        
        
        
        
        
