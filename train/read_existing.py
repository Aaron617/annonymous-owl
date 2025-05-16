import argparse
import json
import os
import re

import datasets
from dotenv import load_dotenv

load_dotenv(override=True)
web_agent_system_prompt = """
You are a helpful assistant that can search the web, extract webpage content, simulate browser actions, and provide relevant information to solve the given task.
Keep in mind that:
- Do not be overly confident in your own knowledge. Searching can provide a broader perspective and help validate existing knowledge.  
- If one way fails to provide an answer, try other ways or methods. The answer does exists.
- If the search snippet is unhelpful but the URL comes from an authoritative source, try visit the website for more details.  
- When looking for specific numerical values (e.g., dollar amounts), prioritize reliable sources and avoid relying only on search snippets.  
- When solving tasks that require web searches, check Wikipedia first before exploring other websites.  
- You can also simulate browser actions to get more information or verify the information you have found.
- Browser simulation is also helpful for finding target URLs. Browser simulation operations do not necessarily need to find specific answers, but can also help find web page URLs that contain answers (usually difficult to find through simple web searches). You can find the answer to the question by performing subsequent operations on the URL, such as extracting the content of the webpage.
- Do not solely rely on document tools or browser simulation to find the answer, you should combine document tools and browser simulation to comprehensively process web page information. Some content may need to do browser simulation to get, or some content is rendered by javascript.
- In your response, you should mention the urls you have visited and processed.

Here are some tips that help you perform web search:
- Never add too many keywords in your search query! Some detailed results need to perform browser interaction to get, not using search toolkit.
- If the question is complex, search results typically do not provide precise answers. It is not likely to find the answer directly using search toolkit only, the search query should be concise and focuses on finding official sources rather than direct answers.
  For example, as for the question "What is the maximum length in meters of #9 in the first National Geographic short on YouTube that was ever released according to the Monterey Bay Aquarium website?", your first search term must be coarse-grained like "National Geographic YouTube" to find the youtube website first, and then try other fine-grained search terms step-by-step to find more urls.
- The results you return do not have to directly answer the original question, you only need to collect relevant information.
"""
web_agent_tools = '[{"name": "web_search", "description": "Performs web search about the given query, and return the search result, containing relevant urls and results.\\nIf searching result does not include relevant information, you need to try other ways to solve the task instead of calling this tool again and again.", "strict": true, "parameters": {"properties": {"question": {"type": "string", "description": "The questions which wanting to obtain relevant information through online searches."}}, "required": ["question"], "type": "object", "additionalProperties": false}}, {"name": "extract_document_content", "description": "Extract the content of a given document (or url) and return the processed text.\\nIt may filter out some information, resulting in inaccurate content.", "strict": true, "parameters": {"properties": {"document_path": {"type": "string", "description": "The path of the document to be processed, either a local path or a URL. It can process image, audio files, zip files and webpages, etc."}, "query": {"type": "string", "description": "The query to be used for retrieving the content. If the content is too long, the query will be used to identify which part contains the relevant information (like RAG). The query should be consistent with the current task."}}, "required": ["document_path", "query"], "type": "object", "additionalProperties": false}}, {"name": "browse_url", "description": "A powerful toolkit which can simulate the browser interaction to solve the task which needs multi-step actions.", "strict": true, "parameters": {"properties": {"task_prompt": {"type": "string", "description": "The task prompt to solve."}, "start_url": {"type": "string", "description": "The start URL to visit."}}, "required": ["task_prompt", "start_url"], "type": "object", "additionalProperties": false}}, {"name": "ask_question_about_video", "description": "Ask a question about the video.", "strict": true, "parameters": {"properties": {"video_path": {"type": "string", "description": "The path to the video file."}, "question": {"type": "string", "description": "The question to ask about the video."}}, "required": ["video_path", "question"], "type": "object", "additionalProperties": false}}]'

document_processing_system_prompt = "You are a helpful assistant that can process documents and multimodal data, such as images, audio, and video."
document_processing_tools = '[{"name": "extract_document_content", "description": "Extract the content of a given document (or url) and return the processed text.\\nIt may filter out some information, resulting in inaccurate content.", "strict": true, "parameters": {"properties": {"document_path": {"type": "string", "description": "The path of the document to be processed, either a local path or a URL. It can process image, audio files, zip files and webpages, etc."}, "query": {"type": "string", "description": "The query to be used for retrieving the content. If the content is too long, the query will be used to identify which part contains the relevant information (like RAG). The query should be consistent with the current task."}}, "required": ["document_path", "query"], "type": "object", "additionalProperties": false}}, {"name": "ask_question_about_image", "description": "Answers image questions with optional custom instructions.", "strict": true, "parameters": {"properties": {"image_path": {"type": "string", "description": "Local path or URL to an image file."}, "question": {"type": "string", "description": "Query about the image content."}, "sys_prompt": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "Custom system prompt for the analysis.\\n(default: :obj:`None`)"}}, "required": ["image_path", "question", "sys_prompt"], "type": "object", "additionalProperties": false}}, {"name": "ask_question_about_audio", "description": "Ask any question about the audio and get the answer using\\nmultimodal model.", "strict": true, "parameters": {"properties": {"audio_path": {"type": "string", "description": "The path to the audio file."}, "question": {"type": "string", "description": "The question to ask about the audio."}}, "required": ["audio_path", "question"], "type": "object", "additionalProperties": false}}, {"name": "ask_question_about_video", "description": "Ask a question about the video.", "strict": true, "parameters": {"properties": {"video_path": {"type": "string", "description": "The path to the video file."}, "question": {"type": "string", "description": "The question to ask about the video."}}, "required": ["video_path", "question"], "type": "object", "additionalProperties": false}}, {"name": "execute_code", "description": "Execute a given code snippet.", "strict": true, "parameters": {"properties": {"code": {"type": "string", "description": "The input code to the Code Interpreter tool call."}}, "required": ["code"], "type": "object", "additionalProperties": false}}]'

reasoning_coding_system_prompt = "You are a helpful assistant that specializes in reasoning and coding, and can think step by step to solve the task. When necessary, you can write python code to solve the task. If you have written code, do not forget to execute the code. Never generate codes like 'example code', your code should be able to fully solve the task. You can also leverage multiple libraries, such as requests, BeautifulSoup, re, pandas, etc, to solve the task. For processing excel files, you should write codes to process them."
reasoning_coding_tools = '[{"name": "execute_code", "description": "Execute a given code snippet.", "strict": true, "parameters": {"properties": {"code": {"type": "string", "description": "The input code to the Code Interpreter tool call."}}, "required": ["code"], "type": "object", "additionalProperties": false}}, {"name": "extract_excel_content", "description": "Extract detailed cell information from an Excel file, including\\nmultiple sheets.", "strict": true, "parameters": {"properties": {"document_path": {"type": "string", "description": "The path of the Excel file."}}, "required": ["document_path"], "type": "object", "additionalProperties": false}}, {"name": "extract_document_content", "description": "Extract the content of a given document (or url) and return the processed text.\\nIt may filter out some information, resulting in inaccurate content.", "strict": true, "parameters": {"properties": {"document_path": {"type": "string", "description": "The path of the document to be processed, either a local path or a URL. It can process image, audio files, zip files and webpages, etc."}, "query": {"type": "string", "description": "The query to be used for retrieving the content. If the content is too long, the query will be used to identify which part contains the relevant information (like RAG). The query should be consistent with the current task."}}, "required": ["document_path", "query"], "type": "object", "additionalProperties": false}}]'
functions_dict = {
    web_agent_system_prompt: web_agent_tools,
    document_processing_system_prompt: document_processing_tools,
    reasoning_coding_system_prompt: reasoning_coding_tools,
}


def parse_final_answer(final_answer):
    # parse ```json
    pattern = re.compile(r"```json(?:\w+)?\s*([\s\S]*?)```", re.DOTALL)
    code_blocks = pattern.findall(final_answer.strip())
    if len(code_blocks) > 0:
        final_answer = code_blocks[0]
    return final_answer


def run(args):
    if args.dataset.endswith(".json"):
        with open(f"{args.dataset}", "r") as f:
            data = json.load(f)
    elif args.dataset.endswith(".jsonl"):
        with open(f"{args.dataset}", "r") as f:
            data = [json.loads(line) for line in f.readlines()]
    elif args.dataset.startswith("owl-agent"):
        data = datasets.load_dataset(args.dataset)["train"]
        args.dataset = args.dataset.split("/")[-1] + "_format" + ".json"
    count_all = 0
    countLong = 0
    for item in data:
        tools = {}
        flag1=0
        flag2=0
        if "trajectory" not in item:
            trajectory = item["metadata"]["trajectory"]
            trajectory = [{"trajectory": trajectory, "success": True}]
        else:
            trajectory = item["trajectory"]
        for each_traj in trajectory:
            subtasks = each_traj["trajectory"][-1]
            if not each_traj["success"]:
                flag1 = 1
            subtasks_history = subtasks["subtasks_history"]
            n = len(subtasks_history)
            for i, each_subtask in enumerate(subtasks_history):
                if not each_subtask["trajectory"] or not each_subtask["result"]:
                    continue
                try:
                    final = parse_final_answer(
                        each_subtask["trajectory"][-1]["content"]
                    )
                    failed = json.loads(final)["failed"]
                except json.JSONDecodeError:
                    raise ValueError("parse final answer failed")
                if failed:
                    continue
                each_subtask_history = each_subtask["trajectory"]
                if (
                    each_subtask_history[0]["role"] == "user"
                    and each_subtask_history[1]["role"] == "system"
                ):
                    each_subtask_history[0]["role"] = "system"
                    tmp = each_subtask_history[0]["content"]
                    each_subtask_history[0]["content"] = each_subtask_history[1][
                        "content"
                    ]
                    each_subtask_history[1]["content"] = tmp
                    each_subtask_history[1]["role"] = "user"
                if each_subtask_history[0]["role"] != "system":
                    print("the first role should be system")
                    continue
                    # raise ValueError("the first role should be system")
                system = each_subtask_history[0]["content"]
                if system not in functions_dict:
                    print(f"system prompt not in functions_dict: {system}")
                    continue
                new_subtask_history = []
                if len(each_subtask_history) > 20:
                    flag2 = 1
                    break
                for each_item in each_subtask_history:
                    tmp_item = {}
                    if (
                        each_item["role"] == "assistant"
                        and "tool_calls" in each_item
                        and each_item["tool_calls"]
                    ):
                        tool_name = each_item["tool_calls"][0]["function"]["name"]
                        if tool_name not in tools:
                            tools[tool_name] = 1
                        else:
                            tools[tool_name] += 1
                        tmp_item["role"] = "function_call"
                        assert (
                            len(each_item["tool_calls"]) == 1
                        ), "we only support one tool call"
                        each_item["tool_calls"][0]["function"]["arguments"] = (
                            json.loads(
                                each_item["tool_calls"][0]["function"]["arguments"]
                            )
                        )
                        tmp_item["content"] = json.dumps(
                            each_item["tool_calls"][0]["function"]
                        )
                    elif each_item["role"] == "tool":
                        tmp_item["role"] = "observation"
                        tmp_item["content"] = each_item["content"]
                    else:
                        tmp_item["role"] = each_item["role"]
                        tmp_item["content"] = each_item["content"]
                        if "tool_calls" in each_item:
                            del each_item["tool_calls"]
                            del each_item["tool_call_id"]
                    new_subtask_history.append(tmp_item)

        count_all += flag1
        countLong += flag2
    # print count
    print(f"count_all: {count_all}, countLong: {countLong}, {countLong/count_all:.2%}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoints
    parser.add_argument("--best_of_n", type=int, default=4)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="gaia")
    args = parser.parse_args()
    os.makedirs("dataset/sft", exist_ok=True)
    run(args)
