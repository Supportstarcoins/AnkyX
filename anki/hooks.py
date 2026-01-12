from addons_manager import gui_hooks


def addHook(name: str, func) -> None:
    hook = getattr(gui_hooks, name, None)
    if hook is None:
        return
    hook.append(func)


def runHook(name: str, *args, **kwargs) -> None:
    hook = getattr(gui_hooks, name, None)
    if hook is None:
        return
    hook(*args, **kwargs)
