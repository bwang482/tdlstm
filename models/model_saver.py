import json


class ModelSaver:
    @classmethod
    def from_dict(cls, dict):
        param_dict = {k: dict[k] for k in cls.PARAMS}
        return cls(**param_dict)

    def to_dict(self):
        return {k: getattr(self, k) for k in self.PARAMS}

    @classmethod
    def load_from_file(cls, path):
        with open(path, "r") as f:
            data = json.loads(f.read())
        return cls.from_dict(data)

    def save_to_file(self, path):
        with open(path, "w") as f:
            f.write(json.dumps(self.to_dict()))

    def print_params(self):
        print("")
        print(json.dumps(self.to_dict(), indent=2))
        print("")