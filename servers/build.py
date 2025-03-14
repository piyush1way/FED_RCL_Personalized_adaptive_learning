from utils import get_numclasses
from utils.registry import Registry
import models

SERVER_REGISTRY = Registry("SERVER")
SERVER_REGISTRY.__doc__ = """
Registry for local updater
"""

__all__ = ['get_server_type', 'build_server']


def get_server_type(args):
    if args.verbose:
        print(SERVER_REGISTRY)

    # Ensure args.server.type is set correctly, default to "Server"
    server_type_name = getattr(args.server, 'type', 'Server')  
    print("=> Getting server type '{}'".format(server_type_name))

    # Fetch the server type from the registry
    server_type = SERVER_REGISTRY.get(server_type_name)

    if server_type is None:
        raise ValueError(f"Error: Server type '{server_type_name}' not found in SERVER_REGISTRY.")

    return server_type



def build_server(args):
    server_type = get_server_type(args)
    server = server_type(args)
    return server
