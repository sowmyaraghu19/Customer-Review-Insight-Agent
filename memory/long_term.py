import os
import json

class LongTermMemory:
    def __init__(self, file="memory/user_profile.json"):
        os.makedirs("memory", exist_ok=True)
        self.file = file

        if not os.path.exists(self.file):
            with open(self.file, "w") as f:
                json.dump({"past_queries": []}, f)

    def load(self):
        return json.load(open(self.file))

    def save(self, data):
        json.dump(data, open(self.file, "w"), indent=2)

    def add_query(self, query):
        data = self.load()
        data["past_queries"].append(query)
        self.save(data)
