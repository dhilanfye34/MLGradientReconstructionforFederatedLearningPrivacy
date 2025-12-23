import torch, pickle

def average_state_dicts(states: list[dict[str, torch.Tensor]]):
    # Element-wise mean of a list of state_dicts
    avg = {}
    for k in states[0]:
        avg[k] = torch.stack([sd[k] for sd in states]).mean(0)
    return avg

def send_pkl(sock, obj: object):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(len(data).to_bytes(8,'big'))
    sock.sendall(data)

def recv_pkl(sock):
    size_bytes = sock.recv(8)
    if not size_bytes:
        raise EOFError("socket closed before size header")
    size = int.from_bytes(size_bytes, 'big')
    buf, CHUNK = b'', 4096
    while len(buf) < size:
        chunk = sock.recv(min(CHUNK, size-len(buf)))
        if not chunk:
            raise EOFError("socket closed mid-payload")
        buf += chunk
    return pickle.loads(buf)