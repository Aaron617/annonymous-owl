import argparse
import json

from camel.toolkits import (
    VideoAnalysisToolkit,
    SearchToolkit,
    CodeExecutionToolkit,
    ImageAnalysisToolkit,
    DocumentProcessingToolkit,
    AudioAnalysisToolkit,
    AsyncBrowserToolkit,
    ExcelToolkit,
    FunctionTool,
)
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType


import os
from typing import List, Dict, Any
from loguru import logger
from utils import OwlWorkforceChatAgent, OwlWorkforce
from utils.gaia import GAIABenchmark
import shutil

from dotenv import load_dotenv

from utils.imm import IMMBenchmark

load_dotenv(override=True)


def construct_agent_list() -> List[Dict[str, Any]]:
    web_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
        model_config_dict={"temperature": 0},
    )

    document_processing_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
        model_config_dict={"temperature": 0},
    )

    multimodal_processing_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
        model_config_dict={"temperature": 0},
    )

    reasoning_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.O3_MINI,
        model_config_dict={"temperature": 0},
    )

    image_analysis_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
        model_config_dict={"temperature": 0},
    )

    video_analysis_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
        model_config_dict={"temperature": 0},
    )

    audio_reasoning_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.O3_MINI,
        model_config_dict={"temperature": 0},
    )

    web_agent_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
        model_config_dict={"temperature": 0},
    )

    planning_agent_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.O3_MINI,
        model_config_dict={"temperature": 0},
    )

    search_toolkit = SearchToolkit()
    document_processing_toolkit = DocumentProcessingToolkit(cache_dir="tmp")
    image_analysis_toolkit = ImageAnalysisToolkit(model=image_analysis_model)
    video_analysis_toolkit = VideoAnalysisToolkit(download_directory="tmp/video")
    # audio_analysis_toolkit = AudioAnalysisToolkit(cache_dir="tmp/audio", audio_reasoning_model=audio_reasoning_model)
    audio_analysis_toolkit = AudioAnalysisToolkit(cache_dir="tmp/audio", reasoning=True)
    # video_analysis_toolkit = VideoAnalysisToolkit(download_directory="tmp/video", model=video_analysis_model, use_audio_transcription=True)
    code_runner_toolkit = CodeExecutionToolkit(sandbox="subprocess", verbose=True)
    browser_simulator_toolkit = AsyncBrowserToolkit(
        headless=True,
        cache_dir="tmp/browser",
        planning_agent_model=planning_agent_model,
        web_agent_model=web_agent_model,
    )
    excel_toolkit = ExcelToolkit()

    web_agent = OwlWorkforceChatAgent(
        """
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
""",
        model=web_model,
        tools=[
            # FunctionTool(search_toolkit.search_google),
            # FunctionTool(search_toolkit.search_wiki),
            # FunctionTool(search_toolkit.search_wiki_revisions),
            FunctionTool(search_toolkit.web_search),
            FunctionTool(document_processing_toolkit.extract_document_content),
            FunctionTool(browser_simulator_toolkit.browse_url),
            FunctionTool(video_analysis_toolkit.ask_question_about_video),
            # FunctionTool(search_toolkit.search_archived_webpage),
        ],
    )

    document_processing_agent = OwlWorkforceChatAgent(
        "You are a helpful assistant that can process documents and multimodal data, such as images, audio, and video.",
        document_processing_model,
        tools=[
            FunctionTool(document_processing_toolkit.extract_document_content),
            FunctionTool(image_analysis_toolkit.ask_question_about_image),
            FunctionTool(audio_analysis_toolkit.ask_question_about_audio),
            FunctionTool(video_analysis_toolkit.ask_question_about_video),
            FunctionTool(code_runner_toolkit.execute_code),
        ],
    )

    reasoning_coding_agent = OwlWorkforceChatAgent(
        "You are a helpful assistant that specializes in reasoning and coding, and can think step by step to solve the task. When necessary, you can write python code to solve the task. If you have written code, do not forget to execute the code. Never generate codes like 'example code', your code should be able to fully solve the task. You can also leverage multiple libraries, such as requests, BeautifulSoup, re, pandas, etc, to solve the task. For processing excel files, you should write codes to process them.",
        reasoning_model,
        tools=[
            FunctionTool(code_runner_toolkit.execute_code),
            FunctionTool(excel_toolkit.extract_excel_content),
            FunctionTool(document_processing_toolkit.extract_document_content),
        ],
    )

    agent_list = []

    web_agent_dict = {
        "name": "Web Agent",
        "description": "A helpful assistant that can search the web, extract webpage content, simulate browser actions, and retrieve relevant information.",
        "agent": web_agent,
    }

    document_processing_agent_dict = {
        "name": "Document Processing Agent",
        "description": "A helpful assistant that can process a variety of local and remote documents, including pdf, docx, images, audio, and video, etc.",
        "agent": document_processing_agent,
    }

    reasoning_coding_agent_dict = {
        "name": "Reasoning Coding Agent",
        "description": "A helpful assistant that specializes in reasoning, coding, and processing excel files. However, it cannot access the internet to search for information. If the task requires python execution, it should be informed to execute the code after writing it.",
        "agent": reasoning_coding_agent,
    }

    agent_list.append(web_agent_dict)
    agent_list.append(document_processing_agent_dict)
    agent_list.append(reasoning_coding_agent_dict)
    return agent_list


def construct_workforce(model_name, port=25001) -> OwlWorkforce:
    task_agent_kwargs = {
        "model": ModelFactory.create(
            model_platform=ModelPlatformType.VLLM,
            model_type=model_name,
            model_config_dict={"temperature": 0},
            url=f"http://localhost:{port}/v1",
        )
    }

    coordinator_agent_kwargs = {
        "model": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.O3_MINI,
            model_config_dict={"temperature": 0},
        )
    }

    workforce = OwlWorkforce(
        "Gaia Workforce",
        task_agent_kwargs=task_agent_kwargs,
        coordinator_agent_kwargs=coordinator_agent_kwargs,
    )

    agent_list = construct_agent_list()

    for agent_dict in agent_list:
        workforce.add_single_agent_worker(
            agent_dict["description"],
            worker=agent_dict["agent"],
            name=agent_dict["name"],
        )

    return workforce


def evaluate_on_gaia(args):
    LEVEL = args.level
    BON = args.best_of_n

    target_trajectory = "planner_history"

    SAVE_RESULT = True
    MAX_TRIES = 1

    test_idx = []

    if os.path.exists("tmp/"):
        # delete all files in the tmp folder
        shutil.rmtree("tmp/")

    # save path
    SAVE_DIR = "train/dataset/dpo"
    SAVE_ENTIRE_RESULT_PATH = (
        f"{SAVE_DIR}/final_{args.dataset}_workforce_{LEVEL}_config2.json"
    )
    SAVE_DPO_RESULT_PATH = {
        "wo_final_answer": f"{SAVE_DIR}/dpo_{args.dataset}_workforce_{LEVEL}_config2.json",
        "w_final_answer": f"{SAVE_DIR}/dpo_{args.dataset}_workforce_{LEVEL}_config2_w_final_answer.json",
    }
    SAVE_KTO_RESULT_PATH = (
        f"{SAVE_DIR}/kto_{args.dataset}_workforce_{LEVEL}_config2.json"
    )

    os.makedirs(SAVE_DIR, exist_ok=True)

    # directly copy the valid data several times
    def workforce():
        return construct_workforce(args.model_name, args.port)

    final_results = {}
    for i in range(BON):
        SAVE_RESULT_PATH = (
            f"{SAVE_DIR}/{i}_{args.dataset}_workforce_{LEVEL}_config2.json"
        )
        if args.dataset == "gaia":
            benchmark = GAIABenchmark(
                data_dir="data/gaia",
                save_to=SAVE_RESULT_PATH,
            )
        elif args.dataset == "imm":
            benchmark = IMMBenchmark(
                data_dir="train/dataset/data/",
                file_name="imm_data.jsonl",
                benchmark_name="imm",
                save_to=SAVE_RESULT_PATH,
            )
        elif args.dataset == "wtq-csv":
            benchmark = IMMBenchmark(
                data_dir="train/dataset/data/",
                file_name="wtq-csv-500.jsonl",
                benchmark_name="wtq-csv",
                save_to=SAVE_RESULT_PATH,
            )
        elif args.dataset == "wtq-xlsx":
            benchmark = IMMBenchmark(
                data_dir="train/dataset/data/",
                file_name="wtq-xlsx-500.jsonl",
                benchmark_name="wtq-xlsx",
                save_to=SAVE_RESULT_PATH,
            )
        elif args.dataset == "math":
            benchmark = IMMBenchmark(
                data_dir="train/dataset/",
                file_name="gaia_train_scored.jsonl",
                benchmark_name="math",
                save_to=SAVE_RESULT_PATH,
            )
        elif args.dataset == "hotpotqa":
            benchmark = IMMBenchmark(
                data_dir="train/dataset/",
                file_name="gaia_train_scored.jsonl",
                benchmark_name="hotpotqa",
                save_to=SAVE_RESULT_PATH,
            )
        else:
            raise ValueError(f"Dataset {args.dataset} not supported yet")

        result = benchmark.run_workforce_concurrently(
            workforce,
            on="valid",
            level=LEVEL,
            idx=test_idx,
            save_result=SAVE_RESULT,
            max_tries=MAX_TRIES,
            max_replanning_tries=1,
            max_workers=6,
            # subset=996 if args.dataset == "hotpotqa" else None,
        )
        for item in result["results"]:
            if item["score"]:
                key = "chosen"
            else:
                key = "rejected"
            if item["task_id"] not in final_results.keys():
                final_results[item["task_id"]] = {
                    "chosen": [],
                    "rejected": [],
                    "question": item["question"],
                    "answer": item["ground_truth"],
                }
            final_results[item["task_id"]][key].append(item)
        logger.success(f"Correct: {result['correct']}, Total: {result['total']}")
        logger.success(f"Accuracy: {result['accuracy']}")
    # save the entire result with json
    with open(SAVE_ENTIRE_RESULT_PATH, "w") as f:
        json.dump(final_results, f, indent=4)

    # process for dpo
    # TODO: worker id is different, may need to be fixed
    dpo_results = []
    dpo_with_final_answer_results = []
    for item in final_results.values():
        # if the chosen and rejected are not empty, just select all of them
        chosen_num = len(item["chosen"])
        rejected_num = len(item["rejected"])
        if chosen_num > 0 and rejected_num > 0:
            data_num = min(chosen_num, rejected_num)
            try:
                dpo_results.extend(
                    {
                        "chosen": item["chosen"][i]["trajectory"][0]["trajectory"][0][
                            target_trajectory
                        ][:-2],
                        "rejected": item["rejected"][i]["trajectory"][0]["trajectory"][
                            0
                        ][target_trajectory][:-2],
                    }
                    for i in range(data_num)
                )
            except IndexError as e:
                print(f"IndexError: {e}")
                continue
            dpo_with_final_answer_results.extend(
                {
                    "chosen": item["chosen"][i]["trajectory"][0]["trajectory"][0][
                        target_trajectory
                    ],
                    "rejected": item["rejected"][i]["trajectory"][0]["trajectory"][0][
                        target_trajectory
                    ],
                }
                for i in range(data_num)
            )  # TODO: random sample
        else:
            # skip for now
            continue
    # save the dpo results
    with open(SAVE_DPO_RESULT_PATH["wo_final_answer"], "w") as f:
        json.dump(dpo_results, f, indent=4)
    with open(SAVE_DPO_RESULT_PATH["w_final_answer"], "w") as f:
        json.dump(dpo_with_final_answer_results, f, indent=4)

    # process for kto/ppo
    kto_data = []
    for item in final_results.values():
        for preference_data in item["chosen"]:
            kto_data.append(
                {
                    "instruction": preference_data["trajectory"][0]["trajectory"][0][
                        target_trajectory
                    ][:2],
                    "response": preference_data["trajectory"][0]["trajectory"][0][
                        target_trajectory
                    ][2:3],
                    "score": 1,
                }
            )
        for preference_data in item["rejected"]:
            try:
                kto_data.append(
                    {
                        "instruction": preference_data["trajectory"][0]["trajectory"][
                            0
                        ][target_trajectory][:2],
                        "response": preference_data["trajectory"][0]["trajectory"][0][
                            target_trajectory
                        ][2:3],
                        "score": 0,
                    }
                )
            except IndexError as e:
                print(f"IndexError: {e}")
                continue
    # save the kto data
    with open(SAVE_KTO_RESULT_PATH, "w") as f:
        json.dump(kto_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoints
    parser.add_argument("--best_of_n", type=int, default=4)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="gaia")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-32B-Instruct",
        help="The opensource model to use.",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=25001,
        help="The port used to connect to the vLLM server.",
    )
    args = parser.parse_args()
    evaluate_on_gaia(args)
