---
title: Licensing - AGPL-3.0-or-later
description: stdapi.ai is free and open source under the AGPL-3.0-or-later license. What the license requires, where the third-party licenses are in the container images, and how to update.
keywords: AGPL license, AGPL-3.0-or-later, open source AI gateway, third-party licenses, container image licenses, FFmpeg LGPL, update stdapi.ai
---

# :material-scale-balance: Licensing

stdapi.ai is free and open source software under the [AGPL-3.0-or-later](https://spdx.org/licenses/AGPL-3.0-or-later.html) license. The source code is on [GitHub](https://github.com/stdapi-ai/stdapi.ai). There is one edition, with no paid license and no usage fee.

The AGPL lets you use, modify and redistribute stdapi.ai. If you modify it and let users interact with it over a network, you must offer those users the source code of your modified version under the same license.

One container image is published, under this license: the hardened image `ghcr.io/stdapi-ai/stdapi.ai-community`. The Terraform module deploys it, and you can run it locally.

## :material-help-circle-outline: Frequently Asked Questions

??? question "Where are the third-party licences inside the container image?"

    The image carries the licence and notice files of everything it redistributes, so a legal or compliance review can be done against the artifact itself:

    - **Python dependencies** — each package keeps its `LICENSE`/`NOTICE`/`COPYING`/`AUTHORS` files (and its `METADATA` and SBOM) inside its own `.dist-info` directory: under `/opt/app`, where the build strips those directories down to exactly these files.
    - **FFmpeg** — `/usr/share/licenses/ffmpeg/`, alongside an entry in the image's package inventory so scanners inventory it like any other installed package. The image builds an audio-only FFmpeg from source with no GPL component enabled, which keeps it under the LGPL-2.1-or-later.
    - **Swagger UI** (Apache-2.0) — `/usr/share/licenses/swagger-ui-dist/`, holding its `LICENSE`, its `NOTICE` and the bundled-dependency notice its script carries.
    - **ReDoc** (MIT) — `/usr/share/licenses/redoc/`, holding its `LICENSE` and the bundled-dependency notice its script carries.
    - **stdapi.ai** — `/usr/share/licenses/stdapi.ai/`, holding `LICENSE-AGPL`.

    Swagger UI and ReDoc are the two documentation UIs served at `/docs` and `/redoc`. The image ships them itself, pinned to an exact release and verified against a recorded SHA-256 at build time, so those pages render with no outbound request at all.

??? question "How do I update stdapi.ai?"

    Updates are container image redeployments — no data migration or downtime window required.

    **Using the Terraform module** (recommended): the module deploys the `latest` image tag by default. Set `version_to_deploy` to a version to pin it: changing it and running `tofu apply` updates the ECS task definition to that image and replaces the running tasks one by one, with no downtime.

    **Manual ECS deployment**: update the container image tag in your ECS task definition and trigger a new deployment.

    Security patches are released as new container image versions. Watch [GitHub releases](https://github.com/stdapi-ai/stdapi.ai/releases) to be notified.

---

## :material-email-outline: Questions

For licensing questions, email [support@stdapi.ai](mailto:support@stdapi.ai) or see [Contact](contact.md).

## :material-arrow-right: Next Steps

<div class="grid cards" markdown>

- :material-rocket-launch: [**Getting Started**](operations_getting_started.md) - Deploy with Terraform
- :material-cash-multiple: [**Cost Management**](operations_cost_management.md) - Infrastructure and AI usage cost breakdown
- :material-docker: [**Local Development**](operations_getting_started_local.md) - Run the image locally

</div>
