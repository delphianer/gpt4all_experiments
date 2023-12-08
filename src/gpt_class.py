
import os
from pathlib import Path
from gpt4all import GPT4All
from common import Funcs as F


# https://docs.gpt4all.io/gpt4all_python.html
from src.gpt_model_setup import gpt_model_setup


class GPT_impl:

    # modelnames:
    # "orca-mini-3b.ggmlv3.q4_0.bin"
    # "llama-2-7b-chat.ggmlv3.q4_0.bin"
    def __startup_check_ok(self, model_name, model_directory):

        path_to_model = os.path.join(os.path.join(os.path.dirname(__file__), model_directory), model_name)
        path = Path(path_to_model)
        os_is_ok = self.model_config.is_system_available()
        any_error = not path.exists() or not path.is_file() or not os.access(path, os.R_OK) or not os_is_ok
        md5_ok = 'not run'
        if not any_error and self.model_config.has_to_check_md5():
            md5_ok = self.model_config.check_md5sum(path)
            if md5_ok:
                self.model_config.check_md5_was_ok()
            else:
                any_error = False

        if any_error:
            F.i("path.exists=", path.exists())
            F.i("path.is_file=", path.is_file())
            F.i("os.access=", os.access(path, os.R_OK))
            F.i("os_is_ok=", os_is_ok)
            F.i("md5 OK", md5_ok)
            F.i("any_error=", any_error)
            F.i("ACHTUNG: Leider besteht kein Zugriff auf das Model! Bitte prüfe die Pfade und Parameter, "+
                "ob das Model existiert!")
            F.i("Die log-File kann hilfreich sein:", F.log_file_name)
            return False
        return True

    def __init__(self,
                 model_name="llama-2-7b-chat.ggmlv3.q4_0.bin",
                 relativ_model_directory="models",
                 answer_language="de",
                 log_directory="gpt_logs",
                 log_filename='testGPT',
                 default_token=10000,
                 default_temp=0.7,
                 default_top_k=40,
                 default_top_p=0.2,
                 use_config_default=True,
                 allow_downloads=False):
        self.model_config = gpt_model_setup(model_name)
        logdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gpt_logs")
        F.initLogging(logdir, "testGPT", True)
        F.i("Start")
        F.i("Model used: " + model_name)
        F.i("relativ model directory used: " + relativ_model_directory)
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), relativ_model_directory)
        F.i("absolute model directory used: " + model_dir)
        if self.__startup_check_ok(model_name, model_dir) or allow_downloads:
            if F.nvidia_gpu_usage:
                self.model = GPT4All(model_name, model_dir, allow_download=allow_downloads, device="nvidia")
            else:
                self.model = GPT4All(model_name, model_dir, allow_download=allow_downloads)
            F.i("\n\nModel initialized\n\n")
            self.model.config["language"] = answer_language
            self.initDone = True
        else:
            self.initDone = False
        # todo: Weitere Config-Params setzen
        self.log_directory = log_directory
        self.log_filename = log_filename
        if use_config_default:
            self.default_token = self.model_config.get_default_token()
            self.current_token = self.model_config.get_default_token()
            self.default_temp = self.model_config.get_default_temp()
            self.default_top_k = self.model_config.get_default_top_k()
            self.default_top_p = self.model_config.get_default_top_p()
        else:
            self.default_token = default_token
            self.current_token = default_token
            self.default_temp = default_temp
            self.default_top_k = default_top_k
            self.default_top_p = default_top_p
        self.question = "hello"
        self.answer = "N/A"

    def __log_question_and_answer(self):
        with open('../gpt_q_and_a.txt', mode="a", encoding="UTF-8") as f:
            f.write("Frage: " + self.question + "\n")
            f.write("Token: " + str(self.current_token) + "\n")
            f.write("Antwort\n>>")
            f.write(self.answer)
            f.write("<<\n")

    def __get_generated_answer(self, temp, top_k, top_p):
        question = self.model_config.get_introduction_sentence().replace("%1", self.question)
        return self.model.generate(question
                                   , max_tokens=self.current_token
                                   , temp=temp
                                   , top_k=top_k
                                   , top_p=top_p
                                   , repeat_penalty=1.9
                                   , repeat_last_n=100
                                   , n_batch=16
                                   # ,streaming=True // Todo: eigene Methode erstellen
                                   # ,callback=a_function
                                   )

    def __generate_gpt_answer(self):
        self.answer = self.__get_generated_answer(temp=self.default_temp
                                                  , top_k=self.default_top_k
                                                  , top_p=self.default_top_p)
        if len(self.answer) < 2:
            F.i("Antwort zu klein:" + self.answer)
            F.i("Neuer Versuch mit höherer Temperatur:")
            t = self.default_temp * 1.2
            if t > 100:
                t = 99
            self.answer = self.__get_generated_answer(temp=t
                                                      , top_k=self.default_top_k
                                                      , top_p=self.default_top_p)
            # Todo: Falls öffter vorkommt, auch andere Parameter entsprechend anpassen
        self.__log_question_and_answer()

    def __init_is_done(self):
        if self.initDone:
            return True
        else:
            print("Initialisierung fehlgeschlagen - chat ist nicht möglich.")
            return False

    def start_chat_with_init_question(self, initQuestion):
        if self.__init_is_done():
            self.question = initQuestion
            print(self.question, ":\n\n")
            self.__generate_gpt_answer()
            print(self.answer)
            print("-" * 100, "\n")
            self.start_chat()

    def start_chat(self):
        if self.__init_is_done():
            while self.question != '':
                try:
                    tstr = input(f"Wie viele Token verwenden? (leer = ignorieren und . = {self.current_token})\n")
                    if len(tstr) == 0:
                        pass
                    elif tstr == '.':
                        self.current_token = int(tstr)
                except Exception as ex:
                    print(ex)
                    self.current_token = self.default_token
                oldQuestion = self.question
                self.question = input("Bitte gebe deine Frage ein (leer = Ende und . = letzte Frage wiederholen)\n")
                if self.question == ".":
                    self.question = oldQuestion
                if self.question != "":
                    self.__generate_gpt_answer()
                    print(self.question, "\n\n", self.answer)
                    print("-" * 100, "\n")

    # todo: Stream-Parameter verwenden um Antworten "schneller" lesen zu können
