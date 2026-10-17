"""
Shared HTTP-client helper so every model adapter can run correctly through
Perplexity Computer's secure credential proxy, and unmodified on a normal
machine with no proxy at all.

Why this exists: when a real API call is routed through the sandbox's
credential proxy (api_credentials=[...] on the bash tool), outbound HTTPS to
the target host is transparently intercepted and the correct Authorization
header is injected. The proxy terminates TLS with a dynamically-issued leaf
certificate for the target host. On this sandbox (Python 3.14 / OpenSSL
3.5), Python's ssl module enables the OpenSSL 3.2+ "X509_V_FLAG_X509_STRICT"
check by default, which rejects that leaf certificate with
"Missing Authority Key Identifier" even though the certificate is otherwise
valid and trusted (curl, which does not enable this strict flag, connects
fine with the exact same certs). This is a client-side over-strict check,
not a real trust problem — the CA is the sandbox's own proxy CA, already
placed in SSL_CERT_FILE/NODE_EXTRA_CA_CERTS by the environment.

Fix: when SSL_CERT_FILE is set (proxy environment), build an SSL context
using a bundle of certifi's public-CA roots + the proxy's own CA, with the
X509_STRICT flag cleared, and hand that context to the SDK's underlying
httpx.Client. When SSL_CERT_FILE is not set (e.g. running directly on the
user's own machine with a real provider key, no sandbox proxy), this
returns None and every SDK uses its normal default TLS behavior unchanged.
"""
from __future__ import annotations
import os
import ssl
from functools import lru_cache
from typing import Optional


@lru_cache(maxsize=1)
def _combined_ca_bundle_path() -> Optional[str]:
    cert_file = os.environ.get("SSL_CERT_FILE")
    if not cert_file or not os.path.exists(cert_file):
        return None
    try:
        import certifi
    except ImportError:
        return cert_file
    combined_path = "/tmp/rag_eval_combined_ca.pem"
    try:
        with open(combined_path, "wb") as out:
            with open(certifi.where(), "rb") as f:
                out.write(f.read())
            out.write(b"\n")
            with open(cert_file, "rb") as f:
                out.write(f.read())
        return combined_path
    except OSError:
        return cert_file


def build_proxy_ssl_context() -> Optional[ssl.SSLContext]:
    """
    Returns an SSLContext with the proxy CA trusted and the overly-strict
    X509_STRICT flag cleared, or None if no sandbox proxy CA is configured
    (in which case callers should fall back to normal default TLS).
    """
    bundle = _combined_ca_bundle_path()
    if not bundle:
        return None
    ctx = ssl.create_default_context(cafile=bundle)
    if hasattr(ssl, "VERIFY_X509_STRICT"):
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return ctx


def build_httpx_client(**kwargs):
    """
    Returns an httpx.Client pre-configured for the sandbox proxy if active,
    or None if no proxy CA is configured (caller should not override the
    SDK's default http_client in that case).
    """
    ctx = build_proxy_ssl_context()
    if ctx is None:
        return None
    import httpx  # local import: only needed when actually building a client
    return httpx.Client(verify=ctx, **kwargs)
