import hashlib

def hash_value(value):
    return hashlib.sha256(value.encode()).hexdigest()