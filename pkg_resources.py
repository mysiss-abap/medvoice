"""
Shim mínimo para evitar el crash de webrtcvad cuando no existe pkg_resources.
webrtcvad solo lo usa para obtener version/package metadata (no es crítico).
"""

class DistributionNotFound(Exception):
    pass

def get_distribution(_name: str):
    # Devuelve un objeto con .version por compatibilidad
    class _D:
        version = "0.0.0"
    return _D()