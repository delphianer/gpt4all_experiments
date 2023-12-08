import logging
import os
import time
import platform
import hashlib


class Funcs:
    error_count = 0
    exception_count = 0
    bad_error_count = 0
    importantMsgs = []
    do_print = False
    log_file_name = ""
    nvidia_gpu_usage = False

    @classmethod
    def isOs(cls, what):

        """Gibt das Betriebssystem zurück."""
        os = platform.system()

        if os == "Windows" and what == os:
            return True
        elif os == "Linux" and what == os:
            return True
        elif os == "Darwin" and what == os:
            return True
        else:
            return False

    @classmethod
    def isWindows(cls):
        return Funcs.isOs("Windows")

    @classmethod
    def isMac(cls):
        return Funcs.isOs("Darwin")

    @classmethod
    def initLogging(cls, logfile_dir_name, logfile_name, do_activate_print):

        # logging-setup
        datetime = time.strftime("%Y-%m-%d")
        logging_directory = os.path.join(os.path.dirname(__file__), logfile_dir_name)
        if not os.path.exists(logging_directory):
            os.makedirs(logging_directory)

        logging_filename = f"{logfile_name}{datetime}_{platform.system()}.log"
        Funcs.log_file_name = os.path.join(logging_directory, logging_filename)
        logging.basicConfig(filename=Funcs.log_file_name,
                            filemode='a',
                            format='%(asctime)s : %(name)s - %(levelname)s - %(message)s',
                            level=logging.DEBUG)
        Funcs.do_print = do_activate_print
        Funcs.print_und_log("Logging initialisiert")
        Funcs.print_und_log("Logfile = " + Funcs.log_file_name)

    @classmethod
    def print_und_log(cls, msg):
        if Funcs.do_print:
            print(msg)
        logging.debug(msg)

    @classmethod
    def i(cls, *msg):
        if Funcs.do_print:
            try:
                msg_out = ""
                for m in msg:
                    msg_out += str(m)
                print(msg_out)
            except Exception as ex:
                print(ex)
                for m in msg:
                    print(m)
        logging.info(msg)

    @classmethod
    def print_and_save(cls, msg):
        Funcs.importantMsgs.append(msg)
        if Funcs.do_print:
            print(msg)

    @classmethod
    def logWarning(cls, msg):
        logging.warning(msg)
        if Funcs.do_print:
            print("Warn:" + msg)
        Funcs.error_count += 1

    @classmethod
    def logError(cls, msg):
        Funcs.print_and_save("-" * 50)
        Funcs.print_and_save("----  ACHTUNG FEHLER: ---- " + msg)
        Funcs.print_and_save("-" * 50)
        logging.error(msg)
        if Funcs.do_print:
            print(msg)
        Funcs.error_count += 1

    @classmethod
    def logException(cls, msg):
        Funcs.print_and_save("##" * 50)
        if type(msg) == "str":
            Funcs.print_and_save("#######  EXCEPTION WURDE AUSGELÖST " + msg)
        Funcs.print_and_save("##" * 50)
        logging.exception(msg)
        if Funcs.do_print:
            print(msg)
        Funcs.exception_count += 1

    @classmethod
    def print_important(cls):
        if Funcs.do_print:
            for msg in Funcs.importantMsgs:
                print(msg)

    @classmethod
    def get_any_error_count(cls):
        return Funcs.error_count + Funcs.exception_count + Funcs.bad_error_count

    @classmethod
    def set_nvidia_gpu_usage(cls):
        Funcs.nvidia_gpu_usage = True



    @classmethod
    def md5sum(cls, filename):
        """Calculates the MD5 checksum of a file.

        Args:
            filename: The path to the file.

        Returns:
            The MD5 checksum of the file.
        """
        hash_md5 = hashlib.md5()
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

