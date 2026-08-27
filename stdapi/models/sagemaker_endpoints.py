"""Amazon SageMaker AI endpoints published as chat models.

An operator names the endpoints they run in ``aws_sagemaker_endpoints``, and
this module turns each entry into an ordinary catalogue model served by the
transport in :mod:`stdapi.aws_sagemaker`. Nothing is discovered: SageMaker AI
publishes neither which container an endpoint runs -- and only some serve the
OpenAI Chat Completions API -- nor what the model behind it should be called,
so an endpoint reaches the catalogue because the operator declared it.

The entries land in the Bedrock catalogue rather than in ``EXTRA_MODELS``:
every text route resolves its model with ``validate_model(bedrock_only=True)``,
which reads ``_MODELS`` alone, so an entry only in ``EXTRA_MODELS`` would be
listed and then answer 404.
"""

from typing import TYPE_CHECKING

from stdapi.config import SETTINGS
from stdapi.models import SAGEMAKER_ENDPOINT_MODELS, SAGEMAKER_SERVICE, ModelDetails

if TYPE_CHECKING:
    from stdapi.config import SageMakerEndpointConfig


def _model_from_endpoint(
    model_id: str, endpoint: SageMakerEndpointConfig
) -> ModelDetails:
    """Build the catalogue entry publishing one declared endpoint.

    Args:
        model_id: Model ID clients name the endpoint with.
        endpoint: The operator's declaration for it.

    Returns:
        The model details to publish.
    """
    return ModelDetails(
        id=model_id,
        name=endpoint.name or model_id,
        provider=endpoint.provider,
        service=SAGEMAKER_SERVICE,
        input_modalities=[modality.upper() for modality in endpoint.input_modalities],
        output_modalities=["TEXT"],
        # Both supported containers stream, and the route is the same one.
        response_streaming=True,
        # Batch inference takes a Bedrock foundation model or inference profile.
        batch=False,
        regions=[endpoint.region],
        sagemaker_endpoint=endpoint,
    )


def merge_sagemaker_endpoint_models(
    all_models: dict[str, ModelDetails], collisions: dict[str, str]
) -> None:
    """Merge the declared SageMaker AI endpoints into the resolved catalogue.

    A model already in the catalogue keeps its entry: an operator who names an
    endpoint after a Bedrock model would otherwise silently replace a
    serverless model, available in every served Region, with one endpoint.

    Args:
        all_models: Resolved bedrock-runtime models, updated in place.
        collisions: Accumulator reporting a declared model ID the catalogue
            already publishes, so the operator sees the entry was ignored.
    """
    SAGEMAKER_ENDPOINT_MODELS.clear()
    for model_id, endpoint in SETTINGS.aws_sagemaker_endpoints.items():
        if model_id in all_models:
            # Keyed by the colliding ID: one key per region would report only
            # the last of several ignored declarations.
            collisions[f"{model_id} (SageMaker AI endpoints)"] = (
                f"aws_sagemaker_endpoints declares '{model_id}', which the model "
                "catalogue already publishes: the endpoint is ignored. Give it a "
                "model ID of its own."
            )
            continue
        model = _model_from_endpoint(model_id, endpoint)
        all_models[model_id] = model
        SAGEMAKER_ENDPOINT_MODELS[model_id] = model
