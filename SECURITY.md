# Security Policy

StrideWeave is an early-stage research project and has not received a formal
security audit. Its raw-pointer and external-memory interfaces should be treated
as unsafe boundaries: callers are responsible for pointer validity, allocation
size, ownership, and lifetime.

Please do not disclose suspected vulnerabilities in a public issue. Use GitHub's
private vulnerability reporting for this repository. If that interface is not
available, contact the maintainer at `patrick.krusiec@gmail.com`.
