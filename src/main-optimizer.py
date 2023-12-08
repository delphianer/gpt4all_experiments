# This is a test that should optimize the python that calls itself .. just as an idea

from gpt4all import GPT4All

# https://docs.gpt4all.io/gpt4all_python.html

# modelname = "orca-mini-3b.ggmlv3.q4_0.bin" # can only use 2048 tokens (context window size) (main.py is 3795)
# modelname = "llama-2-7b-chat.ggmlv3.q4_0.bin" # can only use 2048 tokens (context window size) (main.py is 3795)
modelname = "ggml-model-gpt4all-falcon-q4_0.bin" # can only use 2048 tokens (context window size) (main.py is 3795)

# https://docs.gpt4all.io/gpt4all_python.html#streaming-generations

model = GPT4All(modelname, "../models")


def logQuestionAndAnswer(questionFileName, questionData, token, answer):
    with open('../gpt_q_and_a-optimizer.txt', mode="a", encoding="UTF-8") as f:
        f.write("questionFileName: " + questionFileName + "\n")
        f.write("Original:")
        f.write("-" * 100)
        f.write("\n".join(questionData))
        f.write("-" * 100)
        f.write("Token: " + str(token) + "\n")
        f.write("Antwort")
        f.write("#" * 100)
        f.write("\n")
        f.write(answer)
        f.write("\n")
        f.write("#" * 100)


def getGPTanswer(questionFileName, token, answerFileName):
    with open(questionFileName, "r", encoding='UTF-8') as f:
        questionData = f.readlines()
    output = model.generate(
        "This is a python source code. "+
        "You are"+
        " a master class senior programmer."+
        " Give back the optimized python source code as clean code."+
        "\r\n".join(questionData)
        , max_tokens=token
        #, temp=0.9
        #, top_k=10
        )
    with open(answerFileName, "w", encoding='UTF-8') as f:
        f.write(output)
    logQuestionAndAnswer(questionFileName, questionData, token, output)
    return output


if __name__ == '__main__':
    questionFileName = "main-old.py"
    answerFileName = "main-opt.py"
    tokenCnt = 10000
    # questionFileName = "main-opt.py"
    # answerFileName = "main-opt-2.py"
    # tokenCnt = 20000
    # questionFileName = "main-opt-2.py"
    # answerFileName = "main-opt-3.py"
    # tokenCnt = 40000
    print("\n\nStarting optimizing Sourcecode of", questionFileName, "...")
    answer = getGPTanswer(questionFileName, tokenCnt, answerFileName)
    print("\n\nAnswer:", "\n")
    print("-" * 100, "\n")
    print(answer)
    print("\n", "-" * 100, "\n")
