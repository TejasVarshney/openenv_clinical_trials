"""FastAPI app entrypoint for the Clinical Trial Matcher OpenEnv server."""

from __future__ import annotations

import os

from openenv.core.env_server.http_server import create_app

try:
    from .api_models import ClinicalTrialMatcherAction, ClinicalTrialMatcherObservation
    from .clinical_trial_environment import ClinicalTrialEnvironment
except ImportError:
    from server.api_models import ClinicalTrialMatcherAction, ClinicalTrialMatcherObservation
    from server.clinical_trial_environment import ClinicalTrialEnvironment


app = create_app(
    ClinicalTrialEnvironment,
    ClinicalTrialMatcherAction,
    ClinicalTrialMatcherObservation,
    env_name="clinical_trial_matcher",
    max_concurrent_envs=2,
)


def main() -> None:
    """Run the environment server with a single command entrypoint."""
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
