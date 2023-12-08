import os
import sys

from gpt4all import GPT4All

from common import Funcs as F

# https://docs.gpt4all.io/gpt4all_python.html
from gpt_class import GPT_impl

# Feature-Todo: https://docs.gpt4all.io/gpt4all_python_embedding.html

if __name__ == "__main__":
    print("Welcome Back!\n")

    checkGPU = False
    if checkGPU:
        import tensorflow as tf

        gpus = tf.config.experimental.list_physical_devices("GPU")

        if len(gpus) == 0:
            raise ValueError("Keine GPU-Geräte verfügbar.")
        print(gpus)

    listall = False
    if listall:
        print("Available:")
        avail_list = GPT4All.list_models()
        for params in avail_list:
            print("-"*100)#'filename' = {str} 'wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin'
            print("Name:", params["name"])
            print("filename:", params["filename"])
            print("md5sum:", params["md5sum"])
            print("filesize:", (int(params["filesize"])/1024/1024), "MB")
            print("ramrequired:", params["ramrequired"])
            print("type:", params["type"])
            print("=" * 100)

    # herausfinden ob windows und dann das nutzen: "starcoder-q4_0.gguf.bin"
    if F.isWindows():
        # todo: zum Laufen bekommen:
        #  F.set_nvidia_gpu_usage()
        # llmodel_lib.llmodel_available_gpu_devices
        # ValueError: Unable to retrieve list of all GPU devices
        #transformer = GPT_impl(model_name="wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin")#, allow_downloads=True
        transformer = GPT_impl(model_name="ggml-model-gpt4all-falcon-q4_0.bin", allow_downloads=True)
        #transformer = GPT_impl(model_name="starcoderbase-3b-ggml.bin", allow_downloads=True)
        # Name: Starcoder (Small)
        # filename: starcoderbase-3b-ggml.bin
        # filesize: 7155.534317016602 MB
        # ramrequired: 8
        # type: Starcoder
        # ====================================================================================================
        # transformer = GPT_impl(model_name="starcoderbase-7b-ggml.bin", allow_downloads=True)
        # ----------------------------------------------------------------------------------------------------
        # Name: Starcoder
        # filename: starcoderbase-7b-ggml.bin
        # filesize: 17033.050552368164 MB
        # ramrequired: 16
        # type: Starcoder
        # ====================================================================================================
        # transformer = GPT_impl(model_name="llama-2-7b-chat.ggmlv3.q4_0.bin", allow_downloads=True)
        # ----------------------------------------------------------------------------------------------------
        # Name: Llama-2-7B Chat
        # filename: llama-2-7b-chat.ggmlv3.q4_0.bin
        # filesize: 3616.0709228515625 MB
        # ramrequired: 8
        # type: LLaMA2

        # transformer = GPT_impl(model_name="ggml-gpt4all-j-v1.3-groovy.bin")
        #transformer = GPT_impl(model_name="ggml-model-gpt4all-falcon-q4_0.bin")
        #transformer = GPT_impl(model_name="ggml-replit-code-v1-3b.bin")
        #transformer = GPT_impl(model_name="nous-hermes-13b.ggmlv3.q4_0.bin")
        #transformer = GPT_impl(model_name="nous-hermes-llama2-13b.Q4_0.gguf.bin")
        #transformer = GPT_impl(model_name="orca-mini-3b.ggmlv3.q4_0.bin")
        #transformer = GPT_impl(model_name="starcoder-q4_0.gguf.bin"")
        #transformer = GPT_impl(model_name="llama-2-7b-chat.ggmlv3.q4_0.bin")
    elif F.isMac():
        transformer = GPT_impl(model_name="ggml-model-gpt4all-falcon-q4_0.bin") # kann Perl, Python
        #transformer = GPT_impl(model_name="ggml-replit-code-v1-3b.bin") # kann nicht programmieren... und Prompt klappt nicht
        #transformer = GPT_impl(model_name="llama-2-7b-chat.ggmlv3.q4_0.bin") # braucht lange
        #transformer = GPT_impl(model_name="nous-hermes-13b.ggmlv3.q4_0.bin") # hat einen Bug?
        #transformer = GPT_impl(model_name="orca-mini-3b.ggmlv3.q4_0.bin") # braucht lange
        # todo: Model nicht mehr verfügbar?? --> https://observablehq.com/@simonw/gpt4all-models ?
        # transformer = GPT_impl(model_name="rift-coder-v0-7b-q4_0.gguf", allow_downloads = True) # -> keine Verbindung
    else:
        sys.exit(0)

    transformer.start_chat_with_init_question("Please provide me an implementation of bubble sort algorithm in Python "
                                              "language. Use an array with integers in range of 1 to 49 but let the "
                                              "first number be a number with length = 1. Just pic 8 random Numbers "
                                              "for the array. Call the function to sort that array. then tell me what "
                                              "output will be shown, if I would print the sorted array.")


#
#
# ====================================================================================================
# ----------------------------------------------------------------------------------------------------
# Name: GPT4All Falcon
# filename:
# filesize: 3873.48291015625 MB
# ramrequired: 8
# type: Falcon
# ====================================================================================================
# ----------------------------------------------------------------------------------------------------
# Name: Hermes
# filename: nous-hermes-13b.ggmlv3.q4_0.bin
# filesize: 7759.8353271484375 MB
# ramrequired: 16
# type: LLaMA
# ====================================================================================================
# ----------------------------------------------------------------------------------------------------
# Name: Snoozy
# filename: GPT4All-13B-snoozy.ggmlv3.q4_0.bin
# filesize: 7759.8292236328125 MB
# ramrequired: 16
# type: LLaMA
# ====================================================================================================
# ----------------------------------------------------------------------------------------------------
# Name: Mini Orca
# filename: orca-mini-7b.ggmlv3.q4_0.bin
# filesize: 3616.0938720703125 MB
# ramrequired: 8
# type: OpenLLaMa
# ====================================================================================================
# ----------------------------------------------------------------------------------------------------
# Name: Mini Orca (Small)
# filename: orca-mini-3b.ggmlv3.q4_0.bin
# filesize: 1839.109619140625 MB
# ramrequired: 4
# type: OpenLLaMa
# ====================================================================================================
# ----------------------------------------------------------------------------------------------------
# Name: Mini Orca (Large)
# filename: orca-mini-13b.ggmlv3.q4_0.bin
# filesize: 6984.0709228515625 MB
# ramrequired: 16
# type: OpenLLaMa
# ====================================================================================================
# ----------------------------------------------------------------------------------------------------
# Name: Wizard Uncensored
# filename: wizardLM-13B-Uncensored.ggmlv3.q4_0.bin
# filesize: 7759.8353271484375 MB
# ramrequired: 16
# type: LLaMA
# ====================================================================================================
# ----------------------------------------------------------------------------------------------------
# Name: Replit
# filename: ggml-replit-code-v1-3b.bin
# filesize: 4961.058476448059 MB
# ramrequired: 4
# type: Replit
# ====================================================================================================
# ----------------------------------------------------------------------------------------------------
# Name: Bert
# filename: ggml-all-MiniLM-L6-v2-f16.bin
# filesize: 43.412367820739746 MB
# ramrequired: 1
# type: Bert
# ====================================================================================================
# ----------------------------------------------------------------------------------------------------
# Name: Starcoder (Small)
# filename: starcoderbase-3b-ggml.bin
# filesize: 7155.534317016602 MB
# ramrequired: 8
# type: Starcoder
# ====================================================================================================
# ----------------------------------------------------------------------------------------------------
# Name: Starcoder
# filename: starcoderbase-7b-ggml.bin
# filesize: 17033.050552368164 MB
# ramrequired: 16
# type: Starcoder
# ====================================================================================================
# ----------------------------------------------------------------------------------------------------
# Name: Llama-2-7B Chat
# filename: llama-2-7b-chat.ggmlv3.q4_0.bin
# filesize: 3616.0709228515625 MB
# ramrequired: 8
# type: LLaMA2
# ====================================================================================================
