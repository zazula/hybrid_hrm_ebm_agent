
# Placeholder adapter; integrate with WebArena.
class WebArenaEnv:
    def reset(self): return {'url':'http://localhost', 'dom':'<html/>'}
