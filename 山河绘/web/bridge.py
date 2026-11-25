"""Authenticated localhost IPC; current-run progress and commands stay in RAM."""
import copy
from collections import deque
from multiprocessing.connection import Listener, Client
import threading


class MemoryBridge:
    def __init__(self):
        import secrets
        self.key = secrets.token_bytes(32)
        self.listener = Listener(('127.0.0.1', 0), authkey=self.key)
        self.address = self.listener.address
        self.lock = threading.Lock()
        self.state = {}
        self.commands = deque()
        self.closed = False
        threading.Thread(target=self.serve, daemon=True).start()

    def environment(self):
        return {'SHH_BRIDGE_PORT': str(self.address[1]), 'SHH_BRIDGE_KEY': self.key.hex()}

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.state)

    def command(self, value):
        with self.lock:
            if self.closed:
                raise ValueError('绘画执行器已退出，不能继续本次任务。')
            self.commands.append(copy.deepcopy(value))

    def serve(self):
        try:
            with self.listener.accept() as connection:
                while True:
                    packet = connection.recv()
                    with self.lock:
                        if packet['type'] == 'status':
                            self.state = packet['state']
                            response = {'ok': True}
                        elif packet['type'] == 'poll':
                            response = {'command': self.commands.popleft() if self.commands else None}
                        else:
                            raise ValueError('Unknown bridge packet')
                    connection.send(response)
        except (EOFError, OSError, ValueError):
            pass
        finally:
            with self.lock:
                self.closed = True
            self.listener.close()

    def close(self):
        self.listener.close()


class WorkerChannel:
    def __init__(self, port, key):
        self.connection = Client(('127.0.0.1', int(port)), authkey=bytes.fromhex(key))

    def exchange(self, packet):
        self.connection.send(packet)
        if not self.connection.poll(10):
            raise RuntimeError('网页同步连接未响应，执行退出，请核查现场。')
        return self.connection.recv()

    def publish(self, state):
        self.exchange({'type': 'status', 'state': state})

    def poll(self):
        return self.exchange({'type': 'poll'}).get('command')

    def close(self):
        self.connection.close()
