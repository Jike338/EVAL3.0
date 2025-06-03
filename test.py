from vllm import LLM

def run():
    llm = LLM(model="Qwen/Qwen2-VL-72B-Instruct-AWQ")
    # your inference or testing code here
    print("Model initialized successfully.")

if __name__ == "__main__":
    run()
