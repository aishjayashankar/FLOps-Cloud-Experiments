import tenseal as ts
import os

def create_and_save_context(server_path: str, client_path: str):
    """Generates a CKKS context and saves server (public) and client (private) contexts."""
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 29, 29, 29, 29, 29, 29, 29, 60]
    )
    context.global_scale = 2**29
    # context.generate_galois_keys() # Not needed for element-wise ops, saves huge memory
    
    # Save the context with secret key (for clients)
    with open(client_path, "wb") as f:
        f.write(context.serialize(save_secret_key=True))
        
    # Drop the secret key for the server context
    context.make_context_public()
    with open(server_path, "wb") as f:
        f.write(context.serialize())
        
    return context

def load_context(path: str) -> ts.Context:
    with open(path, "rb") as f:
        data = f.read()
    context = ts.context_from(data)
    return context
