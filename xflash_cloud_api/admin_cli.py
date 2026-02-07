import argparse
import secrets

import db


def cmd_create_user(args: argparse.Namespace) -> None:
    api_key = secrets.token_urlsafe(32)
    user_id = db.create_user(api_key, args.plan, args.credits, is_active=True)
    print(f"user_id={user_id}")
    print(f"api_key={api_key}")


def cmd_add_credits(args: argparse.Namespace) -> None:
    db.update_credits(args.user_id, args.credits)
    print(f"updated user_id={args.user_id} credits_delta={args.credits}")


def cmd_set_plan(args: argparse.Namespace) -> None:
    db.set_plan(args.user_id, args.plan)
    print(f"updated user_id={args.user_id} plan={args.plan}")


def cmd_set_active(args: argparse.Namespace) -> None:
    db.set_active(args.user_id, args.active)
    print(f"updated user_id={args.user_id} active={args.active}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="X-FLASH Cloud API admin CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_user = subparsers.add_parser("create-user", help="Create a user")
    create_user.add_argument("--plan", required=True, choices=["free", "pro"])
    create_user.add_argument("--credits", type=int, required=True)
    create_user.set_defaults(func=cmd_create_user)

    add_credits = subparsers.add_parser("add-credits", help="Add credits")
    add_credits.add_argument("--user-id", type=int, required=True)
    add_credits.add_argument("--credits", type=int, required=True)
    add_credits.set_defaults(func=cmd_add_credits)

    set_plan = subparsers.add_parser("set-plan", help="Set plan")
    set_plan.add_argument("--user-id", type=int, required=True)
    set_plan.add_argument("--plan", required=True, choices=["free", "pro"])
    set_plan.set_defaults(func=cmd_set_plan)

    set_active = subparsers.add_parser("set-active", help="Activate/deactivate user")
    set_active.add_argument("--user-id", type=int, required=True)
    set_active.add_argument("--active", type=lambda x: x.lower() == "true", required=True)
    set_active.set_defaults(func=cmd_set_active)

    return parser


def main() -> None:
    db.init_db()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
