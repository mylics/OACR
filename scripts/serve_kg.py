from __future__ import annotations

import argparse

import uvicorn

from src.io import load_yaml_config
from src.kg.freebase import (
    FreebaseKnowledgeGraphService,
    FreebaseSPARQLClient,
)
from src.kg.server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=("Wrap a Freebase SPARQL endpoint with four KG operations."))
    parser.add_argument(
        "--config",
        default="configs/freebase.yaml",
        help="Freebase SPARQL and KG-server YAML configuration.",
    )
    args = parser.parse_args()
    config = load_yaml_config(args.config)

    client = FreebaseSPARQLClient(
        config["sparql_url"],
        default_graph=config.get("default_graph"),
        request_timeout=int(config["sparql_request_timeout"]),
        relation_limit=int(config["relation_limit"]),
        username_env=config.get("sparql_username_env"),
        password_env=config.get("sparql_password_env"),
        bearer_token_env=config.get("sparql_bearer_token_env"),
    )
    service = FreebaseKnowledgeGraphService(
        client,
        top_k=int(config["top_k"]),
        per_relation_limit=int(config["per_relation_limit"]),
        entity_fetch_limit=int(config["entity_fetch_limit"]),
        cvt_expand_limit=int(config["cvt_expand_limit"]),
    )
    uvicorn.run(
        create_app(service),
        host=str(config["host"]),
        port=int(config["port"]),
        access_log=False,
    )


if __name__ == "__main__":
    main()
