"""MCP servers: a client for them (JSON-RPC over stdio, the server process, protocol-era
detection) and the adapter that turns a server's tools into tools of the registry, with the
limits, fingerprint pins and result wrapping that being somebody else's text calls for, and
the servers of a session: started together within a budget, kept apart when one fails, and
ended together."""
