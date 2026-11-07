"""Exercise the routing and completion decisions used by v-10."""

import argparse
import json

from engine import evaluate_response, evaluate_routing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="Latest user request")
    parser.add_argument("--answer", help="Candidate answer to check")
    parser.add_argument(
        "--history",
        default="[]",
        help="Prior conversation turns as a JSON array",
    )
    args = parser.parse_args()
    try:
        history = json.loads(args.history)
    except json.JSONDecodeError as exc:
        parser.error(f"--history must be valid JSON: {exc}")
    if not isinstance(history, list):
        parser.error("--history must be a JSON array")

    routing = evaluate_routing({"prompt": args.prompt, "history": history})
    print(f"IS_HISTORY_REQUIRED={routing.history_required}")
    print(f"IS_REDIS_HITS_REQUIRED={routing.redis_hits_required}")

    if args.answer is not None:
        response = evaluate_response(
            {"prompt": args.prompt, "answer": args.answer}
        )
        print(f"IS_RESPONSE_ANSWERS_QUERY={response.answers_query}")
        print(f"SHOULD_STOP={response.should_stop}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
