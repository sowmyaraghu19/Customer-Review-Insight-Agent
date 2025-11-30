class ShortTermMemory:
    def __init__(self, data=None):
        self.data = data or {}

    def update(self, **kwargs):
        self.data.update(kwargs)

    def get_all(self):
        return self.data
