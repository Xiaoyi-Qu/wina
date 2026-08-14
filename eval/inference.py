import argparse
import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_args(add_evaluation=False):
    parser = argparse.ArgumentParser(description="llm_config")

    ## model & tokenizer
    parser.add_argument('--output-folder', type=str, default=None)
    parser.add_argument('--load', type=str, default=None,
                       help='Directory containing a model checkpoint.')
    parser.add_argument('--tokenizer-model', type=str, default=None)
    ## dataset path
    parser.add_argument('--datapath', type=str, default='')

    ## others
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device-id', type=str, default=None)

    if add_evaluation:
        parser = _add_evaluation_argument(parser)

    args = parser.parse_args()

    return args


def _add_evaluation_argument(parser):
    group = parser.add_argument_group(title='evaluation')

    ## generation
    group.add_argument('--model-type', type=str, required=True)
    group.add_argument('--temperature', type=float, default=0)
    group.add_argument('--topk', type=int, default=1)
    group.add_argument('--topp', type=float, default=1)
    group.add_argument('--max-output-len', type=int, default=2048)
    group.add_argument('--start-idx', type=int, default=-1)
    group.add_argument('--end-idx', type=int, default=-1)
    group.add_argument('--tensor-parallel-size', type=int, default=1)

    ## hf backend options
    group.add_argument('--trust-remote-code', default=False, action='store_true',
                       help='Pass trust_remote_code=True when loading with the HF backend '
                            '(needed for custom model architectures).')
    group.add_argument('--hf-device-map', type=str, default='auto',
                       help="device_map for HF backend, e.g. 'auto', 'cuda:0'.")

    ## sparse model options
    group.add_argument('--allocation-dir', type=str, default='/teamspace/studios/this_studio/wina/allocation_results',
                       help='Directory with histograms/ and lookup/ from allocation.')
    group.add_argument('--sparse-mode', type=str, default='wina',
                       help="Sparse mode: 'wina' or 'teal'.")
    group.add_argument('--mask-by', type=str, default='topk',
                       help="Masking strategy: 'topk' or 'threshold'.")
    group.add_argument('--transform', default=False, action='store_true',
                       help='Must match how you ran grab_acts.')
    group.add_argument('--sparsity', type=float, default=0.0,
                       help='Sparsity level for load_greedy_sparsities.')

    ## inference api
    group.add_argument('--max-workers', type=int, default=16)
    group.add_argument('--eval-dataset-list', nargs='*', type=str)
    group.add_argument('--stop-token-ids', nargs='*', type=int)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--fp16', default=False, action='store_true')
    parser.add_argument('--bf16', default=False, action='store_true')
    return parser


def get_starter_code(header_str):
    if "def " in header_str:
        starter_code = header_str.split("def")[1].split("(")[0].strip()
    else:
        starter_code = header_str

    return starter_code

def preprocess_livecodebench(data_file, model_type):
    with open(data_file, "r") as f:
        data_list = json.load(f)

    instruction = ""
    if model_type == "qwen":
        instruction = "<|im_start|>system\nYou are a helpful and harmless assistant. You should think step-by-step.<|im_end|>\n"
    elif model_type == "mistral":
        instruction = "You are a helpful and harmless assistant. You should think step-by-step.\n"

    prompt_list = []
    qid_list = []
    for item in data_list:
        question = item['question_content'].strip()

        code_instruction_nostartercode = """Write Python code to solve the problem. Please place the solution code in the following format:\n```python\n# Your solution code here\n```"""
        code_instruction_hasstartercode = """Please place the solution code in the following format:\n```python\n# Your solution code here\n```"""

        if item['starter_code'] != "":
            question += "\n\n" + "Solve the problem starting with the provided function header.\n\nFunction header:\n" + "```\n" + item['starter_code'] + "\n```"
            question += "\n\n" + code_instruction_hasstartercode
        else:
            question += "\n\n" + code_instruction_nostartercode

        if model_type == "qwen":
            final_prompt = instruction + "<|im_start|>user\n" + question + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
        elif model_type == "mistral":
            final_prompt = "[INST] " + instruction + question + " [/INST]<think>\n"
        else:
            final_prompt = "<｜User｜>" + question + "<｜Assistant｜><think>\n"

        prompt_list.append(final_prompt)
        qid_list.append(item['question_id'])

    return prompt_list, qid_list

def preprocess_aime(data_file, model_type):

    prompt_list = []
    qid_list = []
    with open(data_file, "r") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            final_question = item['problem'].strip()
            if model_type == "qwen":
                final_prompt = """<|im_start|>system\nYou are a helpful and harmless assistant. You should think step-by-step.<|im_end|>\n<|im_start|>user\n{question}\n\nPlease place your final answer inside \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n<think>\n""".format(question=final_question)
            elif model_type == "mistral":
                final_prompt = """[INST] You are a helpful and harmless assistant. You should think step-by-step.\n{question}\n\nPlease place your final answer inside \\boxed{{}}. [/INST]<think>\n""".format(question=final_question)
            else:
                final_prompt = """<｜begin▁of▁sentence｜><｜User｜>{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<｜Assistant｜><think>\n""".format(question=final_question)
            prompt_list.append(final_prompt)
            qid_list.append(i)
    return prompt_list, qid_list


def load_sparse_model(args):
    import sys 
    sys.path.insert(0, "/teamspace/studios/this_studio/wina")
    from utils.utils import get_sparse_model, get_tokenizer

    torch_dtype = torch.float16 if args.fp16 else torch.bfloat16

    model_path = args.load
    tokenizer_path = args.tokenizer_model if args.tokenizer_model else model_path

    print("load tokenizer from %s" % tokenizer_path)
    print("load model from %s" % model_path)
    print("torch_dtype:", torch_dtype)
    print("allocation_dir:", args.allocation_dir)
    print("sparse_mode:", args.sparse_mode)
    print("mask_by:", args.mask_by)
    print("transform:", args.transform)
    print("sparsity:", args.sparsity)

    tokenizer = get_tokenizer(tokenizer_path)
    tokenizer.padding_side = "left"                # <-- add this
    if tokenizer.pad_token is None:                # <-- and this block
        tokenizer.pad_token = tokenizer.eos_token

    model = get_sparse_model(
        model_path,
        device=args.hf_device_map,
        histogram_path=os.path.join(args.allocation_dir, "histograms"),
        sparse_mode=args.sparse_mode,
        mask_by=args.mask_by,
        transform=args.transform,
        torch_dtype=torch_dtype,
    )

    model.load_greedy_sparsities(
        os.path.join(args.allocation_dir, "lookup"),
        args.sparsity,
    )

    model.eval()

    return model, tokenizer


def hf_generate_batch(model, tokenizer, batch_prompts, args):
    """Generate text for a batch of prompts using the HF model."""
    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=args.max_output_len,
        do_sample=args.temperature > 0,
    )
    if args.temperature > 0:
        gen_kwargs["temperature"] = args.temperature
        if args.topp < 1:
            gen_kwargs["top_p"] = args.topp
        else:
            gen_kwargs["top_k"] = args.topk

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    # slice off prompt tokens
    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, prompt_length:]
    batch_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

    # --- compute per-token entropy via a forward pass ---
    with torch.no_grad():
        # run the full sequence (prompt + generated) through the model
        logits = model(output_ids).logits  # (batch, seq_len, vocab_size)

    # we only need logits at positions that predicted generated tokens
    # logits at position t predict token at position t+1
    # so logits[:, prompt_length-1:-1, :] correspond to generated tokens
    gen_logits = logits[:, prompt_length - 1:-1, :]  # (batch, gen_len, vocab)

    probs = F.softmax(gen_logits, dim=-1)
    log_probs = F.log_softmax(gen_logits, dim=-1)
    entropies = -(probs * log_probs).sum(dim=-1)  # (batch, gen_len)

    return batch_texts, entropies


def get_prompt_list(args):
    ## get input data
    input_datapath = args.datapath
    if "aime" in args.datapath or "amc" in args.datapath or "minervamath" in args.datapath \
        or "math" in args.datapath or "olympiad" in args.datapath:
        prompt_list, qid_list = preprocess_aime(input_datapath, args.model_type)
    elif "livecodebench" in args.datapath:
        prompt_list, qid_list = preprocess_livecodebench(input_datapath, args.model_type)
    else:
        raise ValueError("Invalid dataset name")

    print("number of total prompt_list:", len(prompt_list))
    if args.start_idx != -1 and args.end_idx != -1:
        print("getting data from %d to %d" % (args.start_idx, args.end_idx))
        prompt_list = prompt_list[args.start_idx:args.end_idx]
        if qid_list:
            qid_list = qid_list[args.start_idx:args.end_idx]

    print("number of test samples in the dataset:", len(prompt_list))
    return prompt_list, qid_list


def truncate_at_stop_strings(generated_text):
    for stop_str in ["<|im_end|>", "<|end_of_text|>", "<|eot_id|>", "</s>", "<|endoftext|>"]:
        if stop_str in generated_text:
            idx = generated_text.index(stop_str)
            generated_text = generated_text[:idx]
    return generated_text


def main():
    args = get_args(add_evaluation=True)
    if args.device_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    for key, value in vars(args).items():
        print(f"{key}: {value}")

    # seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ## load sparse model
    model, tokenizer = load_sparse_model(args)

    ## load test data
    prompt_list, qid_list = get_prompt_list(args)

    ## run inference
    print("args.max_output_len:", args.max_output_len)

    output_list = []
    for i in tqdm(range(0, len(prompt_list), args.batch_size)):
        batch_prompts = prompt_list[i:i+args.batch_size]
        if qid_list:
            batch_qids = qid_list[i:i+args.batch_size]

        batch_texts, entropies = hf_generate_batch(model, tokenizer, batch_prompts, args)

        for j, generated_text in enumerate(batch_texts):
            generated_text = truncate_at_stop_strings(generated_text)

            # get per-token entropies for this sample
            # tokenize the generated text to know actual token count
            gen_tokens = tokenizer.encode(generated_text, add_special_tokens=False)
            token_entropies = entropies[j, :len(gen_tokens)].tolist() # type: ignore

            # also store the individual tokens for alignment
            token_strings = tokenizer.convert_ids_to_tokens(gen_tokens)

            output_dict = {"output": generated_text}
            if qid_list:
                output_dict["task_id"] = batch_qids[j]
            output_dict["token_entropies"] = token_entropies
            output_dict["tokens"] = token_strings
            output_dict["mean_entropy"] = sum(token_entropies) / len(token_entropies) if token_entropies else 0.0
            output_dict["token_count"] = len(token_strings)

            output_list.append(output_dict)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    output_name = "%dto%d_seed%d.jsonl" % (args.start_idx, args.end_idx, args.seed) \
                            if args.start_idx != -1 and args.end_idx != -1 else "seed%d.jsonl" % args.seed

    output_datapath = os.path.join(args.output_folder, output_name)

    print("writing to %s" % output_datapath)
    with open(output_datapath, "w", encoding='utf-8') as f:
        for output in output_list:
            if type(output) == dict:
                f.write(json.dumps(output) + "\n")
            else:
                f.write(output + "\n")

if __name__ == "__main__":
    main()