import tiktoken
import numpy as np
from transformers import AutoTokenizer
import vertexai
from vertexai.generative_models import GenerativeModel
from zhipuai import ZhipuAI
class OpenAI:
    def ChatGPT4oLatest(self, input_string):
        if not input_string:
            return 0
    
        try:
            encoder = tiktoken.get_encoding("ChatGPT-4o-latest(2024-09-03)")
            tokens = encoder.encode(input_string)
        except Exception as e:
            print(f"Ocurrio un error al usar tiktoken: {e}")
            tokens = np.nan

        return tokens
    
    def GPT4omini20240718(self, input_string):
        if not input_string:
            return 0
    
        try:
            encoder = tiktoken.get_encoding("GPT-4o-mini-2024-07-18")
            tokens = encoder.encode(input_string)
        except Exception as e:
            print(f"Ocurrio un error al usar tiktoken: {e}")
            tokens = np.nan

        return tokens
    
    def o1_preview(self, input_string):
        if not input_string:
            return 0
    
        try:
            encoder = tiktoken.get_encoding("o1-preview")
            tokens = encoder.encode(input_string)
        except Exception as e:
            print(f"Ocurrio un error al usar tiktoken: {e}")
            tokens = np.nan

        return tokens
    
    def o1_mini(self, input_string):
        if not input_string:
            return 0
    
        try:
            encoder = tiktoken.get_encoding("o1-mini")
            tokens = encoder.encode(input_string)
        except Exception as e:
            print(f"Ocurrio un error al usar tiktoken: {e}")
            tokens = np.nan

        return tokens


class Google:
    
    def Gemini_1_5_Pro_002(self, input_string):
        if not input_string:
            return 0
    
        try:
            model=GenerativeModel("google/gemini-1.5-pro-002")
            tokens=model.count_tokens(input_string)
            
        except Exception as e:
            print(f"Ocurrio un error:{e}")
            tokens = np.nan

        return tokens
    
    def Gemini_1_5_Flash_Exp_0827(self, input_string):
        if not input_string:
            return 0
        
        try:
            model=GenerativeModel("google/gemini-1.5-flash-exp-0827")
            tokens = model.count_tokens(input_string)
        except Exception as e:
            print(f"Ocurrio un error:{e}")
            tokens = np.nan

        return tokens
    
    def Gemma_2_27b_it(self, input_string):
        if not input_string:
            return 0
        try:
            model=GenerativeModel("google/gemma-2.27b-it")
            tokens = model.count_tokens(input_string)
        except Exception as e:
            print(f"Ocurrio un error:{e}")
            tokens = np.nan

class a_01AI:

    def Yi_Lightning(self, input_string):
        
        if not input_string:
            return 0  # Return 0 for empty input
    
        try:
            tokens = example_tokenizer(input_string)  # Replace with actual tokenizer logic
        except Exception:
            tokens = input_string.split()  # Fallback to simple split if tokenizer fails

        return len(tokens)
    
    def Yi_Lightning_lite(self, input_string):
        if not input_string:
            return 0
    
    def Yi_Large_preview(self, input_string):
        if not input_string:
            return 0
    def Yi_Large(self, input_string):
        if not input_string:
            return 0

class zhipu_AI:

    def GLM_4_Plus(self, input_string):
        
        if not input_string:
            return 0  # Return 0 for empty input
    
        try:
            pass
        except Exception as e:
            print(f"{e}")

        return 0
    def GLM_4_0520(self, input_string): 
        if not input_string:
            return 0
        
        try:
            pass
        except Exception as e:
            print(f"{e}")
        
        return tokens

class Anthropic:
    
    def Claude3_5Sonnet(self, input_string):
        
        if not input_string:
            return 0  # Return 0 for empty input
    
        try:
            tokens = example_tokenizer(input_string)  # Replace with actual tokenizer logic
        except Exception:
            tokens = input_string.split()  # Fallback to simple split if tokenizer fails

        return len(tokens)

    def Claude3Opus(self, input_string):
        
        if not input_string:
            return 0
    
    def Claude3Sonnet(self, input_string):
        
        if not input_string:
            return 0

class Meta:
    
    def Meta_Llama_3_1_405b_Instruct_bf16(self, input_string):
        
        if not input_string:
            return 0  # Return 0 for empty input
    
        try:
            tokens = example_tokenizer(input_string)  # Replace with actual tokenizer logic
        except Exception:
            tokens = input_string.split()  # Fallback to simple split if tokenizer fails

        return len(tokens)
    
    def Meta_Llama_3_1_405b_Instruct_fp8(self, input_string):
        
        if not input_string:
            return 0
        
    def Meta_Llama_3_1_70b_Instruct(self, input_string):
        
        if not input_string:
            return 0

class Alibaba:
            
    def Qwen_Max_0919(self, input_string):
        if not input_string:
            return 0  # Return 0 for empty input
        try:
            tokenizer = AutoTokenizer.from_pretrained("alibaba/qwen-max-0919")
            tokens = tokenizer.encode(input_string)
            return tokens
        except Exception:
            tokens = np.nan
        return tokens
    
    def Qwen2_5_72b_Instruct(self, input_string):
        if not input_string:
            return 0
        try:
            tokenizer = AutoTokenizer.from_pretrained("alibaba/qwen2.5-72b-instruct")
            tokens = tokenizer.encode(input_string)
            return tokens
        except Exception:
            tokens = np.nan
        return tokens

    
    def Qwen_Plus_0828(self, input_string):
        if not input_string:
            return 0
        try:
            tokenizer = AutoTokenizer.from_pretrained("alibaba/qwen-plus-0828")
            tokens = tokenizer.encode(input_string)
            return tokens
        except Exception:
            tokens = np.nan
        return tokens

class DeepSeek:
    def Deepseek_v2_5(self, input_string):
        
        if not input_string:
            return 0  # Return 0 for empty input
    
        try:
            tokens = example_tokenizer(input_string)  # Replace with actual tokenizer logic
        except Exception:
            tokens = input_string.split()  # Fallback to simple split if tokenizer fails

        return len(tokens)
    
    def Deepseek_Coder_v2_0724(self, input_string):
        
        if not input_string:
            return 0

class DeepSeekAI:
    def Deepseek_v2_API_0628(self, input_string):
        
        if not input_string:
            return 0  # Return 0 for empty input
    
        try:
            tokens = example_tokenizer(input_string)  # Replace with actual tokenizer logic
        except Exception:
            tokens = input_string.split()  # Fallback to simple split if tokenizer fails

        return len(tokens)

class Mistral:
    def Mistral_Large_2407(self, input_string):
        
        if not input_string:
            return 0  # Return 0 for empty input
    
        try:
            tokens = example_tokenizer(input_string)  # Replace with actual tokenizer logic
        except Exception:
            tokens = input_string.split()  # Fallback to simple split if tokenizer fails

        return len(tokens)

class NexusFlow:
    def Athene_70b(self, input_string):
        
        if not input_string:
            return 0  # Return 0 for empty input
    
        try:
            tokens = example_tokenizer(input_string)  # Replace with actual tokenizer logic
        except Exception:
            tokens = input_string.split()  # Fallback to simple split if tokenizer fails

        return len(tokens)

class RekaAI:

    def Reka_Core_20240722(self, input_string):
        if not input_string:
            return 0
        
    def Reka_Flash_20240722(self, input_string):
        if not input_string:
            return 0

class AI21Labs:

    def Jamba_1_5_Large(self, input_string):
        
        if not input_string:
            return 0  # Return 0 for empty input
    
        try:
            tokens = example_tokenizer(input_string)  # Replace with actual tokenizer logic
        except Exception:
            tokens = input_string.split()  # Fallback to simple split if tokenizer fails

        return len(tokens)


class Princeton:
    def Gemma_2_9b_it_SimPO(self, input_string):
        
        if not input_string:
            return 0  # Return 0 for empty input
    
        try:
            model=GenerativeModel("princeton/gemma-2.9b-it-simpo")
            tokens = model.count_tokens  # Replace with actual tokenizer logic
        except Exception as e:
            print(f"Ocurrio un error:{e}")
            tokens = np.nan

        return tokens

class Cohere:
    def CommandR__08_2024_(self, input_string):
        
        if not input_string:
            return 0  # Return 0 for empty input
    
        try:
            tokens = example_tokenizer(input_string)  # Replace with actual tokenizer logic
        except Exception:
            tokens = input_string.split()  # Fallback to simple split if tokenizer fails

        return len(tokens)

class Nvidia:
    def Nemotron_4_340B_Instruct(self, input_string):
        
        if not input_string:
            return 0  # Return 0 for empty input
    
        try:
            pass
        except Exception:
            pass

        return 0

