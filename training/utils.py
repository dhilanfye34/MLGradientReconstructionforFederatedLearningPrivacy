import torch, pickle

def average_state_dicts(states: list[dict[str, torch.Tensor]]):
    """Element-wise mean of a list of state_dicts."""
    avg = {}
    for k in states[0]:
        avg[k] = torch.stack([sd[k] for sd in states]).mean(0)
    return avg

def send_pkl(sock, obj: object):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(len(data).to_bytes(8,'big'))
    sock.sendall(data)

def recv_pkl(sock):
    size = int.from_bytes(sock.recv(8),'big')
    buf, CHUNK = b'', 4096
    while len(buf) < size:
        buf += sock.recv(min(CHUNK, size-len(buf)))
    return pickle.loads(buf)
