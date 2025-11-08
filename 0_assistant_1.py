# assistant.py
from llama_cpp import Llama

class Assistant:
    # def __init__(self, model_path=None, mmproj_path=None, gpu_id=1):
    def __init__(self, model_path=None):
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        default_model_path="./qwen2-0_5b-instruct-q4_k_m.gguf"
        self.model_path = model_path if model_path is not None else default_model_path

        # Initialize Llama with Qwen2.5-Omni multimodal support
        self.llm = Llama(
            model_path=self.model_path,
            # n_ctx=32768,
            n_ctx=1024,
            logits_all=True,
            # n_gpu_layers=-1,
            # n_threads=16,
            # n_threads_batch=16,
            seed=1337,
            verbose=True,
        )

    def generate_message(self, instructions="", user_message="", stop_tokens=None, max_tokens=500):
        """Text-only generation."""
        if stop_tokens is None:
            stop_tokens = []
        response = self.llm.create_chat_completion(
            max_tokens=max_tokens,
            stop=stop_tokens,
            messages=[
                {"role": "user", "content": instructions + user_message + "\n"},
            ]
        )
        return response['choices'][0]['message']['content']


# === Simple Run Example ===
if __name__ == '__main__':
    # Initialize assistant
    assistant = Assistant()

    # === TEXT-ONLY TEST ===
    print("\n=== TEXT-ONLY PROMPT ===")
    print(assistant.generate_message(
        user_message="How much wood could a woodchuck chuck if a woodchuck could chuck wood?"
    ))

    input("//////// PRESS ENTER ////////")

