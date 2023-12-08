from gpt4all import GPT4All

if __name__ == "__main__":
    print("Welcome!\n")

    modelname = "llama-2-7b-chat.ggmlv3.q4_0.bin"

    model = GPT4All(modelname, "../models")
    model.config["language"] = 'de'

    def logQuestionAndAnswer(question, token, answer):
        with open('../gpt_q_and_a.txt', mode="a", encoding="UTF-8") as f:
            f.write("Frage: "+question+"\n")
            f.write("Token: " + str(token) + "\n")
            f.write("Antwort\n>>")
            f.write(answer)
            f.write("<<\n")


    def getGPTanswer(question, token):
        output = model.generate(question
                                , max_tokens=token
                                , temp=0.42
                                , top_k=20
                                , top_p=0.2
                                , repeat_penalty=1.9
                                , repeat_last_n=100
                                , n_batch=16
                                #,streaming=True
                                #,callback=a_function
                                )
        logQuestionAndAnswer(question, token, output)
        return output


    if __name__ == '__main__':
        question = "Die Hauptstadt von Frankreich ist "
        tokenCnt = 3
        while question != '.':
            answer = getGPTanswer(question, tokenCnt)
            print(question, "\n\n", answer)
            print("-" * 100, "\n")
            try:
                tstr = input(f"Wie viele Token verwenden? (leer = {tokenCnt})\n")
                if len(tstr) > 0:
                    tokenCnt = int(tstr)
            except Exception as ex:
                print(ex)
                tokenCnt = 3
            oldQuestion = question
            question = input("Bitte gebe deine Frage ein (leer = Ende und . = letzte Frage wiederholen)\n")
            if question == ".":
                question = oldQuestion