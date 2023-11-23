import platform


class Env:
    tmp_dir = ""
    os = ""

    def __init__(self):
        self.os = platform.platform()
        if self.os.startswith("mac"):
            self.os = "mac"
            self.tmp_dir = "."
        elif self.os.startswith("Linux"):
            self.os = "linux"
            self.tmp_dir = "/dev/shm/"
        else:
            self.os = "unknown"

    def is_mac(self):
        return self.os == "mac"

    def is_linux(self):
        return self.os == "linux"


ENV = Env()
