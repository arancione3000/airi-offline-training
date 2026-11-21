"""Verify the public download, chat, and offline restart in Android."""
import re
import subprocess
from time import monotonic, sleep
from xml.etree import ElementTree

PACKAGE = 'com.airi.chat'
ACTIVITY = f'{PACKAGE}/.MainActivity'


def adb(*args):
    return subprocess.check_output(['adb', *args], timeout=30, text=True)


def find_node(attribute, value, timeout=40):
    end = monotonic() + timeout
    while monotonic() < end:
        try:
            adb('shell', 'uiautomator', 'dump', '/sdcard/airi-smoke.xml')
            xml = adb('exec-out', 'cat', '/sdcard/airi-smoke.xml')
            for node in ElementTree.fromstring(xml).iter('node'):
                if value in node.get(attribute, ''):
                    return node
        except (subprocess.SubprocessError, ElementTree.ParseError):
            pass
        sleep(2)
    raise RuntimeError(f'Screen element not found: {attribute} {value}')


def tap(node):
    x1, y1, x2, y2 = map(int, re.findall(r'\d+', node.attrib['bounds']))
    adb('shell', 'input', 'tap', str((x1 + x2) // 2), str((y1 + y2) // 2))


def chat():
    tap(find_node('class', 'android.widget.EditText'))
    adb('shell', 'input', 'text', 'Ciao')
    tap(find_node('text', 'Invia'))
    print(find_node('text', 'Ciao! Come stai?', timeout=90).get('text'), flush=True)


def main():
    print(adb('install', '/tmp/airi-chat-1.0.apk'))
    print(adb('install', '-r', 'mobile/release/Airi-Chat.apk'))
    print(adb('shell', 'am', 'start', '-W', '-n', ACTIVITY))
    print(find_node('text', 'Sul telefono', timeout=210).get('text'), flush=True)
    chat()
    adb('shell', 'am', 'force-stop', PACKAGE)
    adb('shell', 'cmd', 'connectivity', 'airplane-mode', 'enable')
    adb('shell', 'svc', 'wifi', 'disable')
    adb('shell', 'svc', 'data', 'disable')
    assert adb('shell', 'settings', 'get', 'global', 'airplane_mode_on').strip() == '1'
    print(adb('shell', 'am', 'start', '-W', '-n', ACTIVITY))
    print(find_node('text', 'Sul telefono', timeout=45).get('text'), flush=True)
    chat()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print(adb('logcat', '-d', '-s', 'Airi:W', 'AndroidRuntime:E'))
        raise
