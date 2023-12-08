import datetime
import json
import os.path
import platform

from gpt4all import GPT4All

from common import Funcs as F
import yaml


class gpt_model_setup:
    # Prompt-source: https://github.com/nomic-ai/gpt4all/issues/631
    __default_config = {
        "introduction-sentence": """### System:
You are an artificial assistant that gives facts based answers.
You strive to answer concisely.
You review the answer after you respond to fact check it.
Importantly, think step by step while reviewing the answer.
Append to the message the correctness of the original answer from 0 to 9, where 0 is not correct at all and 9 is perfectly correct.
Enclose the review in double curly braces {{ }}.
If you are able to give the text-result in german language, then do that.
### Human:
%1
### Assistant:""",
        #"introduction-sentence": "### System:\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.",
        "default_token": 10000,
        "default_temp": 0.7,
        "default_top_k": 40,
        "default_top_p": 0.2,
        "systems-available": ["None"]
    }

    def __init__(self, current_model_file_name):
        self.model_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model_setup.yaml')
        self.current_model_file_name = current_model_file_name
        self.model_config = {}
        self.config = {}
        if os.path.exists(self.model_file):
            with open(file=self.model_file, encoding="UTF-8", mode="r") as file:
                self.model_config = yaml.safe_load(file)
            if self.model_config is None or not current_model_file_name in self.model_config:
                self.init_defaults()
            else:
                self.config = self.model_config[self.current_model_file_name + "-config"]
        else:
            self.init_defaults()

    def init_defaults(self):
        self.set_default_config()
        self.load_additional_model_info()
        self.save_current_config()

    def set_default_config(self):
        F.i("setting up new config for "+self.current_model_file_name)
        self.model_config = {}
        self.model_config[self.current_model_file_name] = 'created:' + datetime.datetime.strftime(
            datetime.datetime.now(), "%d.%m.%Y %H:%M:%S")
        self.model_config[self.current_model_file_name + "-config"] = self.__default_config
        self.config = self.model_config[self.current_model_file_name + "-config"]
        self.config["systems-available"].append(platform.system())

    def load_additional_model_info(self):
        # todo: live holen? avail_list = GPT4All.list_models()
        json_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "current_models.json")
        with open(json_file, mode="r", encoding="UTF-8") as f:
            avail_list = json.load(f)
        #print(avail_list)
        for params in avail_list:
            #print(params["filename"], " ?? ", self.current_model_file_name)
            if params["filename"] == self.current_model_file_name:
                self.config['Name']=params['name']
                self.config['filesize'] = params['filesize']
                self.config['md5sum'] = params['md5sum']
                self.config['last_md5sum_run'] = '00000000'
                self.config['ramrequired'] = params['ramrequired']
                self.config['model-order'] = params['order']
                self.config['parameters'] = params['parameters']
                self.config['quant'] = params['quant']
                self.config['type'] = params['type']
                self.config['systemPrompt'] = params['systemPrompt']
                self.config['description'] = params['description']
                F.i("Loaded Data for", params['name'])
                F.i("Info:", params['description'])
                return

    def save_current_config(self):
        F.i("saving config for " + self.current_model_file_name+" in " + self.model_file)
        self.model_config[self.current_model_file_name + "-config"] = self.config
        with open(file=self.model_file, encoding="UTF-8", mode="w") as file:
            yaml.dump(self.model_config, file)


    def check_md5sum(self, filepath):
        key = "md5sum"
        if not self.__key_valid(key):
            F.i("ACHTUNG: MD5 der File ist nicht gesetzt!")
        else:
            if len(self.config[key]) == 0:
                F.i("ACHTUNG: MD5 der File ist nicht gesetzt!")
            else:
                md5 = F.md5sum(filepath)
                if md5 == self.config[key]:
                    F.i("File MD5 OK")
                    return True
                else:
                    msg = "MD5 of File NOT OK: SOLL: " + self.config[key] + " IST:" + md5
                    F.logException(msg)
                    raise Exception(msg)
        return False

    # Abfrage der Parameter:

    def get_introduction_sentence(self):
        return self.config["introduction-sentence"]

    def __key_valid(self, key):
        if len(self.config)==0:
            return False
        if not key in self.config:
            return False
        return True

    def is_system_available(self):
        key = "systems-available"
        if not self.__key_valid(key):
            return False
        system= platform.system()

        debugging=False
        if debugging==True:
            print("#"*100)
            print("Info zur Config:self.config - len=", len(self.config))
            for key in self.config:
                print(key,"=",self.config[key])
            print("#" * 100)

        return system in self.config[key]

    def get_model_name(self):
        key = "Name"
        if not self.__key_valid(key):
            return "(Nicht gesetzt)"
        return self.config[key]

    def get_default_token(self):
        key = "default_token"
        if not self.__key_valid(key):
            return self.__default_config[key]
        return self.config[key]

    def get_default_temp(self):
        key = "default_temp"
        if not self.__key_valid(key):
            return self.__default_config[key]
        return self.config[key]

    def get_default_top_k(self):
        key = "default_top_k"
        if not self.__key_valid(key):
            return self.__default_config[key]
        return self.config[key]

    def get_default_top_p(self):
        key = "default_top_p"
        if not self.__key_valid(key):
            return self.__default_config[key]
        return self.config[key]

    def has_to_check_md5(self):
        key = "last_md5sum_run"
        if not self.__key_valid(key):
            return True
        last_run = int(self.config['last_md5sum_run'])
        today_is = int(datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d"))
        return today_is - last_run > 15

    def check_md5_was_ok(self):
        self.config['last_md5sum_run'] = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d")
        self.save_current_config()

