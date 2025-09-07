
# Placeholder adapter; integrate with OSWorld env when available.
class OSWorldEnv:
    def reset(self):
        return {'screen':'<pixels>', 'state':{}}
    def step(self, action):
        return {'obs':'<pixels>', 'reward':0.0, 'done':False, 'info':{}}
