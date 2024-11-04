
MODELS_DICT = {
        # Nano(1B) Models
         "stablelm1_6": {"full": "stabilityai/stablelm-2-1_6b-chat",
                          "gguf": "https://huggingface.co/second-state/stablelm-2-zephyr-1.6b-GGUF/resolve/main/stablelm-2-zephyr-1_6b-Q5_K_M.gguf",
#                           "gptq": None,
                         },
    
        "llama_3.2_1b_instruct": {"full": "meta-llama/Llama-3.2-1B-Instruct",
                                    "gguf": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K_L.gguf",
                                    "gguf_large": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf",
            #                         "awq": ""
                                },

        # Mini(3B)  Models
         "phi3.5_mini_instruct": {"full": "microsoft/Phi-3.5-mini-instruct",
                  "gguf": "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q6_K_L.gguf",
                  "gguf_large": "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q8_0.gguf",
#                   "awq": None
                         },

         "phi3_mini_instruct_graph": {"full": "EmergentMethods/Phi-3-mini-4k-instruct-graph",
                                     },

         "llama_3.2_3b": {"full": "meta-llama/Llama-3.2-3B-Instruct",
                            "gguf": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K_L.gguf",
                            "gguf_large": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q8_0.gguf",
        #                     "awq": ""
                                },
    
        # Small(7-8)B Models
         "llama_3.1_8b": {"full": "meta-llama/Llama-3.1-8B-Instruct",
                    "gguf": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf",
                    "gguf_large": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
                    "awq": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
                        },
          "mistral_7b_instruct_v0.3":{"full": "mistralai/Mistral-7B-Instruct-v0.3",
                                       "gguf": "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q5_K_M.gguf",
                                       "gguf_large": "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q8_0.gguf",
                                       "awq": "solidrust/Mistral-7B-Instruct-v0.3-AWQ"
                        },
          "mistra_8b_instruct":{"full": "mistralai/Ministral-8B-Instruct-2410",
                                # "gguf": "",
                                # "gguf_large": "",
                        },
          
          "qwen2.5_7b_instruct": {"full": "Qwen/Qwen2-7B-Instruct",
                                "gguf": "https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/resolve/main/qwen2-7b-instruct-q6_k.gguf",
                                "gguf_large": "https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/resolve/main/qwen2-7b-instruct-q8_0.gguf",
                               },
          "gemma_2_9b_instruct": {"full": "google/gemma-2-9b-it",
                                  "gguf": "https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q6_K_L.gguf",
                                  "ggguf_large": "https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q8_0.gguf"
                  },
          "interlm_2.5_instruct": {"full": "internlm/internlm2_5-7b-chat",
                                   "gguf": "https://huggingface.co/internlm/internlm2_5-7b-chat-gguf/resolve/main/internlm2_5-7b-chat-q6_k.gguf",
                                   "gguf_large": "https://huggingface.co/internlm/internlm2_5-7b-chat-gguf/resolve/main/internlm2_5-7b-chat-q8_0.gguf",
                  },
    
        # Code Models
        "deepseek2.5_coder": {"full": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
#                          "gguf": "",
#                          "gguf_large": "",
#                           "awq":                        
                         },

        # Vision Models
    
         "phi3.5_mini_vl": {"full": "microsoft/Phi-3.5-vision-instruct",
                                  "gguf": "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q6_K_L.gguf",
                                  "gguf_large": "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q8_0.gguf"
                            },
        
        "paligemma_3b_224": {"full": "google/paligemma-3b-pt-224",
                            "gguf": "https://huggingface.co/abetlen/paligemma-3b-mix-224-gguf/resolve/main/paligemma-3b-mix-224-text-model-q4_k_m.gguf",
                            "gguf_large": "https://huggingface.co/abetlen/paligemma-3b-mix-224-gguf/resolve/main/paligemma-3b-mix-224-text-model-q8_0.gguf"
                            },
    
        "pixtral": {"full": "meta-llama/Llama-3.2-11B-Vision-Instruct",
                    "gguf": "https://huggingface.co/leafspark/Pixtral-12B-2409-hf-text-only-GGUF/resolve/main/Pixtral-12B-2409-hf-text-only.Q4_K_L.gguf",
                    "gguf_large": "https://huggingface.co/leafspark/Pixtral-12B-2409-hf-text-only-GGUF/resolve/main/Pixtral-12B-2409-hf-text-only.Q8_0_L.gguf",           
                    },
    
        "molmo_7b_preview": {"full": "allenai/Molmo-7B-D-0924",
#                              "gguf":,
#                              "gguf_large":
                             },
    
        "qwen2_vl_7b_instruct": {"full": "Qwen/Qwen2-VL-7B-Instruct",
        #                          "gguf": "",
        #                          "gguf_large": "",
                                  "awq": "Qwen/Qwen2-VL-7B-Instruct-AWQ"                        
                         },
    
         "minicpm_llama3_8b_vl": {"full": "openbmb/MiniCPM-Llama3-V-2_5",
                                     "gguf": "https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf/resolve/main/ggml-model-Q4_K_M.gguf",
#                                      "gguf_large": "",
                                      "gguf_xlarge": "https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf/resolve/main/ggml-model-F16.gguf"
                        },   
         }
