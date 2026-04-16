# Security Policy

## Supported Versions

This project is currently in MVP stage. Security fixes are applied to the latest `main` branch.

## Reporting a Vulnerability

Please do not open public issues for suspected security vulnerabilities.

Instead, report privately to the repository maintainer with:

- a clear description of the issue
- reproduction steps or proof of concept
- potential impact
- any suggested mitigations

If you do not have a private reporting channel configured in your Git hosting platform, contact the repository owner directly.

## Response Process

We aim to:

1. acknowledge receipt within 3 business days
2. validate and triage within 7 business days when possible
3. provide a remediation plan or workaround
4. publish a fix and advisory notes when appropriate

## Disclosure

We follow coordinated disclosure. Please allow time for investigation and patching before public disclosure.

## Scope Notes

Given the nature of this project, common risk areas include:

- prompt injection through untrusted RDF/text input
- unsafe handling of generated RDF output without validation
- dependency vulnerabilities

Users should validate generated outputs and keep dependencies updated.
