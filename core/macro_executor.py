"""
macro_executor.py — Flexible, extensible macro action engine.

A macro is a dict with a ``type`` and type-specific payload. Supported types
(chosen from what modern launchers/macro tools provide the most value with):

    hotkey   {"keys": ["ctrl", "alt", "t"]}      simultaneous key combo
    press    {"keys": ["playpause"]}             one or more single key presses
    text     {"value": "hello world"}            type a string of text
    url      {"value": "https://..."}            open a web URL / URI scheme
                                                  (http, https, mailto:, slack://, ...)
    open     {"value": "~/Documents"}            open a file/folder in default app
    launch   {"value": "code ."}                 launch an app / run a program (no shell)
    shell    {"value": "notify-send hi"}         run a shell command (pipes, globs, &&)
    sequence {"actions": [ {..}, {"type":"delay","ms":200}, {..} ]}
                                                  chain several actions with delays

pyautogui (keyboard simulation) is imported lazily so the DSP core still runs in
headless environments; URL/open/launch/shell use only the standard library.
"""

import os
import sys
import shlex
import subprocess
import webbrowser
import time

import config

try:
    import pyautogui
    pyautogui.PAUSE = 0.03
    pyautogui.FAILSAFE = False
    _PYAUTOGUI_ERROR = None
except Exception as _e:  # pragma: no cover - depends on host display stack
    pyautogui = None
    _PYAUTOGUI_ERROR = _e


# Action types the GUI can offer + a short human hint for the value field.
MACRO_TYPE_HINTS = {
    "hotkey": "keys, e.g.  ctrl+alt+t",
    "press":  "key name(s), e.g.  playpause  or  volumemute",
    "text":   "text to type, e.g.  my.email@example.com",
    "url":    "URL / URI, e.g.  https://chatgpt.com  or  mailto:me@x.com",
    "open":   "file or folder path, e.g.  ~/Downloads",
    "launch": "program + args, e.g.  code .   or  firefox",
    "shell":  "shell command, e.g.  notify-send 'Snap!' && date",
}


class MacroExecutor:
    def __init__(self, dry_run=False, logger=None):
        """
        Parameters
        ----------
        dry_run : if True, actions are validated and described but not performed
                  (used by the test suite and the GUI "test" button).
        logger  : optional callable(str) for user-facing status messages.
        """
        self.macros = config.MACRO_DEFINITIONS
        self.dry_run = dry_run
        self.logger = logger or (lambda msg: None)

    # ────────────────────────────────────────────
    def execute(self, macro_name: str) -> bool:
        macro = self.macros.get(macro_name)
        if macro is None:
            self.logger(f"Unknown macro: {macro_name}")
            return False
        try:
            return self._run_action(macro)
        except Exception as e:
            self.logger(f"Macro '{macro_name}' failed: {e}")
            return False

    # ────────────────────────────────────────────
    def _run_action(self, action: dict) -> bool:
        atype = action.get("type", "hotkey")

        if atype == "sequence":
            ok = True
            for sub in action.get("actions", []):
                if sub.get("type") == "delay":
                    if not self.dry_run:
                        time.sleep(sub.get("ms", 100) / 1000.0)
                    continue
                ok = self._run_action(sub) and ok
            return ok

        keys = list(action.get("keys", []))
        value = action.get("value", "")

        if self.dry_run:
            self.logger(f"[dry-run] {atype}: {keys or value}")
            return self._validate(atype, keys, value)

        if atype == "hotkey":
            return self._need_pyautogui() and (pyautogui.hotkey(*keys) or True)
        if atype == "press":
            if not self._need_pyautogui():
                return False
            for k in keys:
                pyautogui.press(k)
            return True
        if atype == "text":
            return self._need_pyautogui() and (pyautogui.write(value, interval=0.01) or True)
        if atype == "url":
            return self._open_uri(value, is_url=True)
        if atype == "open":
            return self._open_uri(os.path.expanduser(value), is_url=False)
        if atype == "launch":
            return self._spawn(shlex.split(value), use_shell=False)
        if atype == "shell":
            return self._spawn(value, use_shell=True)

        self.logger(f"Unsupported macro type: {atype}")
        return False

    # ────────────────────────────────────────────
    @staticmethod
    def _validate(atype, keys, value) -> bool:
        if atype in ("hotkey", "press"):
            return len(keys) > 0
        if atype in ("text", "url", "open", "launch", "shell"):
            return bool(str(value).strip())
        return atype == "sequence"

    def _need_pyautogui(self) -> bool:
        if pyautogui is None:
            self.logger(f"Keyboard simulation unavailable ({_PYAUTOGUI_ERROR}).")
            return False
        return True

    def _open_uri(self, target: str, is_url: bool) -> bool:
        """Open a URL/URI or a file/folder with the OS default handler."""
        if not target:
            return False
        try:
            if sys.platform == "win32":
                os.startfile(target)  # noqa: S606 - intended default-open
            elif sys.platform == "darwin":
                self._detached(["open", target])
            else:  # Linux / *nix: freedesktop xdg-open handles URLs, schemes, files
                try:
                    self._detached(["xdg-open", target])
                except FileNotFoundError:
                    if is_url:
                        webbrowser.open(target)
                    else:
                        raise
            return True
        except Exception as e:
            # Last-ditch fallback for URLs.
            if is_url:
                try:
                    webbrowser.open(target)
                    return True
                except Exception:
                    pass
            self.logger(f"Could not open '{target}': {e}")
            return False

    def _spawn(self, args, use_shell: bool) -> bool:
        """Launch a detached process so the app never blocks on it."""
        try:
            self._detached(args, use_shell=use_shell)
            return True
        except Exception as e:
            self.logger(f"Could not run {args!r}: {e}")
            return False

    @staticmethod
    def _detached(args, use_shell=False):
        kwargs = dict(
            shell=use_shell,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        if os.name == "posix":
            kwargs["start_new_session"] = True  # don't die with the parent
        subprocess.Popen(args, **kwargs)
