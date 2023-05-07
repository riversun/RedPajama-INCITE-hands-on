from transformers import set_seed
from chat_core import process_chat
from chat_prompt import ChatPrompt
from model_loader_for_redpajama_incite import load_hf_model

# Fix seed value for verification.
seed_value = 42
set_seed(seed_value)

model_path = "togethercomputer/RedPajama-INCITE-Chat-3B-v1"

model, tokenizer, device = load_hf_model(model_path=model_path, device="cuda")  # Works if "cpu", but it's slow.

chatPrompt = ChatPrompt()

chatPrompt.set_requester("<human>")
chatPrompt.set_responder("<bot>")

chat_mode = True  # You can do multi-round chats while keeping context.

while True:
    user_input = input("YOU: ")
    if user_input.lower() == "exit":
        break

    if chat_mode:
        chatPrompt.add_requester_msg(user_input)
        chatPrompt.add_responder_msg(None)
        prompt = chatPrompt.create_prompt()
        stop_strs = chatPrompt.get_stop_strs()

    else:
        prompt = user_input
        stop_str = None

    params = {
        "prompt": prompt,
        "temperature": 0.7,
        "max_new_tokens": 512,
        "context_len": 1024,
        "use_top_k_sampling": True,
        "top_k_value": 50,
        "use_top_p_sampling": True,
        "top_p_value": 0.7,
        "stop_strs": stop_strs,
    }

    generator = process_chat(model, tokenizer, device, params)

    prev = ""

    for index, response_text in enumerate(generator):

        if index == 0:
            print("AI : ", end="", flush=True)

        if chat_mode:
            response_text = response_text[chatPrompt.get_skip_len():].strip()
        else:
            # response_text = response_text[len(prompt):].strip()
            pass

        updated_text = response_text[len(prev):]

        print(updated_text, end="", flush=True)

        prev = response_text

    print()

    if chat_mode:
        chatPrompt.set_responder_last_msg(response_text.strip())
