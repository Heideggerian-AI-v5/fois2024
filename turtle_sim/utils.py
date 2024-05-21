import threading

def makeStartStopFns(gvar, fn):
    def _start(gvar, fn, *args, **kwargs):
        if gvar.get("thread") is None:
            gvar["keepOn"] = True
            gvar["thread"] = threading.Thread(target=lambda : fn(*args, **kwargs), daemon=True)
            gvar["thread"].start()
    def _stop(gvar):
        if gvar.get("thread") is not None:
            gvar["keepOn"] = False
            gvar["thread"].join()
            gvar["thread"] = None
    return (lambda *args, **kwargs: _start(gvar, fn, *args, **kwargs)), (lambda : _stop(gvar))
