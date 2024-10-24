import onnxruntime_genai as og
import argparse

def main(args):
    model = og.Model(f'{args.model}')
    tokenizer = og.Tokenizer(model)

    if hasattr(args, 'prompts'):
        prompts = args.prompts
    else:
        prompts = ["I like walking my cute dog",
                   "What is the best restaurant in town?",
                   "Hello, how are you today?"]
    
    if args.chat_template:
        if args.chat_template.count('{') != 1 or args.chat_template.count('}') != 1:
            print("Error, chat template must have exactly one pair of curly braces, e.g. '<|user|>\n{input} <|end|>\n<|assistant|>'")
            exit(1)
        prompts[:] = [f'{args.chat_template.format(input=text)}' for text in prompts]
        
    input_tokens = tokenizer.encode_batch(prompts)

    params = og.GeneratorParams(model)

    search_options = {name:getattr(args, name) for name in ['do_sample', 'max_length', 'min_length', 'top_p', 'top_k', 'temperature', 'repetition_penalty'] if name in args} 


    params.set_search_options(**search_options)
    # Set the batch size for the CUDA graph to the number of prompts if the user didn't specify a batch size
    params.try_graph_capture_with_max_batch_size(len(prompts))
    if args.batch_size_for_cuda_graph:
        params.try_graph_capture_with_max_batch_size(args.batch_size_for_cuda_graph)
    params.input_ids = input_tokens

    # output_tokens = model.generate(params)

    # for i in range(len(prompts)):
    #     # print(f'Prompt #{i}: {prompts[i]}')
    #     # print()
    #     print(tokenizer.decode(output_tokens[i]))
    #     print()
    
    generator = og.Generator(model, params)
    output = ""
    while not generator.is_done():
        generator.compute_logits()
        generator.generate_next_token()

        new_token = generator.get_next_tokens()[0]
        output += tokenizer.decode(new_token)
    print(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="End-to-end token generation loop example for gen-ai")
    parser.add_argument('-m', '--model', type=str, required=True, help='Onnx model folder path (must contain config.json and model.onnx)')
    parser.add_argument('-pr', '--prompts', nargs='*', required=False, help='Input prompts to generate tokens from. Provide this parameter multiple times to batch multiple prompts')
    parser.add_argument('-i', '--min_length', type=int, help='Min number of tokens to generate including the prompt')
    parser.add_argument('-l', '--max_length', type=int, default=2048, help='Max number of tokens to generate including the prompt')
    parser.add_argument('-ds', '--do_random_sampling', action='store_true', help='Do random sampling. When false, greedy or beam search are used to generate the output. Defaults to false')
    parser.add_argument('-p', '--top_p', type=float, help='Top p probability to sample with')
    parser.add_argument('-k', '--top_k', type=int, help='Top k tokens to sample from')
    parser.add_argument('-t', '--temperature', type=float, help='Temperature to sample with')
    parser.add_argument('-r', '--repetition_penalty', type=float, help='Repetition penalty to sample with')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Print verbose output and timing information. Defaults to false')
    parser.add_argument('-b', '--batch_size_for_cuda_graph', type=int, default=1, help='Max batch size for CUDA graph')
    parser.add_argument('-c', '--chat_template', type=str, default='<|user|>\n{input} <|end|>\n<|assistant|>', help='Chat template to use for the prompt. User input will be injected into {input}. If not set, the prompt is used as is.')

    args = parser.parse_args()
    main(args)